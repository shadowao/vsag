
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pyramid.h"

#include "data_cell/flatten_interface.h"
#include "empty_index_binary_set.h"
#include "impl/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "io/memory_io_parameter.h"
#include "utils/slow_task_timer.h"

namespace vsag {

constexpr static const int64_t INIT_CAPACITY = 10;
constexpr static const int64_t MAX_CAPACITY_EXTEND = 10000;

std::vector<std::string>
split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    if (str.empty()) {
        throw std::runtime_error("fail to parse empty path");
    }

    while (end != std::string::npos) {
        std::string token = str.substr(start, end - start);
        if (token.empty()) {
            throw std::runtime_error("fail to parse path:" + str);
        }
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    std::string last_token = str.substr(start);
    if (last_token.empty()) {
        throw std::runtime_error("fail to parse path:" + str);
    }
    tokens.push_back(str.substr(start, end - start));
    return tokens;
}

IndexNode::IndexNode(IndexCommonParam* common_param, GraphInterfaceParamPtr graph_param)
    : ids_(common_param->allocator_.get()),
      children_(common_param->allocator_.get()),
      common_param_(common_param),
      graph_param_(std::move(graph_param)) {
}

void
IndexNode::BuildGraph(ODescent& odescent) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Build an index when the level corresponding to the current node requires indexing
    if (has_index_ && not ids_.empty()) {
        InitGraph();
        entry_point_ = ids_[0];
        odescent.Build(ids_);
        odescent.SaveGraph(graph_);
        Vector<InnerIdType>(common_param_->allocator_.get()).swap(ids_);
    }
    for (auto& item : children_) {
        item.second->BuildGraph(odescent);
    }
}

void
IndexNode::AddChild(const std::string& key) {
    // AddChild is not thread-safe; ensure thread safety in calls to it.
    children_[key] = std::make_shared<IndexNode>(common_param_, graph_param_);
    children_[key]->level_ = level_ + 1;
}

std::shared_ptr<IndexNode>
IndexNode::GetChild(const std::string& key, bool need_init) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = children_.find(key);
    if (result != children_.end()) {
        return result->second;
    }
    if (not need_init) {
        return nullptr;
    }
    AddChild(key);
    return children_[key];
}

void
IndexNode::Deserialize(StreamReader& reader) {
    // deserialize `entry_point_`
    StreamReader::ReadObj(reader, entry_point_);
    // deserialize `level_`
    StreamReader::ReadObj(reader, level_);
    // serialize `has_index_`
    StreamReader::ReadObj(reader, has_index_);
    // deserialize `graph`
    if (has_index_) {
        InitGraph();
        graph_->Deserialize(reader);
    }
    // deserialize `children`
    size_t children_size = 0;
    StreamReader::ReadObj(reader, children_size);
    for (int i = 0; i < children_size; ++i) {
        std::string key = StreamReader::ReadString(reader);
        AddChild(key);
        children_[key]->Deserialize(reader);
    }
    StreamReader::ReadObj(reader, has_index_);
}

void
IndexNode::Serialize(StreamWriter& writer) const {
    // serialize `entry_point_`
    StreamWriter::WriteObj(writer, entry_point_);
    // serialize `level_`
    StreamWriter::WriteObj(writer, level_);
    // serialize `has_index_`
    StreamWriter::WriteObj(writer, has_index_);
    // serialize `graph_`
    if (has_index_) {
        graph_->Serialize(writer);
    }
    // serialize `children`
    size_t children_size = children_.size();
    StreamWriter::WriteObj(writer, children_size);
    for (const auto& item : children_) {
        // calculate size of `key`
        StreamWriter::WriteString(writer, item.first);
        // calculate size of `content`
        item.second->Serialize(writer);
    }
    StreamWriter::WriteObj(writer, has_index_);
}
void
IndexNode::InitGraph() {
    graph_ = GraphInterface::MakeInstance(graph_param_, *common_param_, true);
}

MaxHeap
IndexNode::SearchGraph(const SearchFunc& search_func) const {
    if (graph_ != nullptr && graph_->TotalCount() > 0) {
        return search_func(this);
    }
    MaxHeap search_result(common_param_->allocator_.get());
    for (const auto& [key, node] : children_) {
        MaxHeap child_search_result = node->SearchGraph(search_func);
        while (not child_search_result.empty()) {
            auto result = child_search_result.top();
            child_search_result.pop();
            search_result.push(result);
        }
    }
    return search_result;
}

tl::expected<std::vector<int64_t>, Error>
Pyramid::build(const DatasetPtr& base) {
    const auto* path = base->GetPaths();
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto& no_build_levels = pyramid_param_.no_build_levels;

    resize(data_num);
    std::memcpy(labels_.label_table_.data(), data_ids, sizeof(LabelType) * data_num);

    flatten_interface_ptr_->Train(data_vectors, data_num);
    flatten_interface_ptr_->BatchInsertVector(data_vectors, data_num);

    ODescent graph_builder(pyramid_param_.odescent_param,
                           flatten_interface_ptr_,
                           common_param_.allocator_.get(),
                           common_param_.thread_pool_.get());
    for (int i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        std::shared_ptr<IndexNode> node = root_;
        for (auto& path_slice : path_slices) {
            node = node->GetChild(path_slice, true);
            node->ids_.push_back(i);
            node->has_index_ =
                std::find(no_build_levels.begin(), no_build_levels.end(), node->level_) ==
                no_build_levels.end();
        }
    }
    root_->BuildGraph(graph_builder);
    cur_element_count_ = data_num;
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::knn_search(const DatasetPtr& query,
                    int64_t k,
                    const std::string& parameters,
                    const BitsetPtr& invalid) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;
    if (invalid != nullptr) {
        auto filter_adpater = std::make_shared<UniqueFilter>(invalid);
        search_param.is_inner_id_allowed =
            std::make_shared<CommonInnerIdFilter>(filter_adpater, labels_);
    }
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    return this->search_impl(query, k, search_func);
}

tl::expected<DatasetPtr, Error>
Pyramid::knn_search(const DatasetPtr& query,
                    int64_t k,
                    const std::string& parameters,
                    const std::function<bool(int64_t)>& filter) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;
    if (filter != nullptr) {
        auto filter_adpater = std::make_shared<UniqueFilter>(filter);
        search_param.is_inner_id_allowed =
            std::make_shared<CommonInnerIdFilter>(filter_adpater, labels_);
    }
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    return this->search_impl(query, k, search_func);
}

tl::expected<DatasetPtr, Error>
Pyramid::range_search(const DatasetPtr& query,
                      float radius,
                      const std::string& parameters,
                      int64_t limited_size) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    int64_t final_limit = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;
    SAFE_CALL(return this->search_impl(query, final_limit, search_func);)
}

tl::expected<DatasetPtr, Error>
Pyramid::range_search(const DatasetPtr& query,
                      float radius,
                      const std::string& parameters,
                      const BitsetPtr& invalid,
                      int64_t limited_size) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    if (invalid != nullptr) {
        auto filter_adpater = std::make_shared<UniqueFilter>(invalid);
        search_param.is_inner_id_allowed =
            std::make_shared<CommonInnerIdFilter>(filter_adpater, labels_);
    }
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    int64_t final_limit = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;
    return this->search_impl(query, final_limit, search_func);
}

tl::expected<DatasetPtr, Error>
Pyramid::range_search(const DatasetPtr& query,
                      float radius,
                      const std::string& parameters,
                      const std::function<bool(int64_t)>& filter,
                      int64_t limited_size) const {
    auto parsed_param = PyramidSearchParameters::FromJson(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_param.ef_search;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    if (filter != nullptr) {
        auto filter_adpater = std::make_shared<UniqueFilter>(filter);
        search_param.is_inner_id_allowed =
            std::make_shared<CommonInnerIdFilter>(filter_adpater, labels_);
    }
    SearchFunc search_func = [&](const IndexNode* node) {
        search_param.ep = node->entry_point_;
        std::lock_guard<std::mutex> lock(node->mutex_);
        auto vl = pool_->TakeOne();
        auto results = searcher_->Search(
            node->graph_, flatten_interface_ptr_, vl, query->GetFloat32Vectors(), search_param);
        pool_->ReturnOne(vl);
        return results;
    };
    int64_t final_limit = limited_size == -1 ? std::numeric_limits<int64_t>::max() : limited_size;
    return this->search_impl(query, final_limit, search_func);
}

tl::expected<DatasetPtr, Error>
Pyramid::search_impl(const DatasetPtr& query, int64_t limit, const SearchFunc& search_func) const {
    const auto* path = query->GetPaths();
    std::string current_path = path[0];
    auto path_slices = split(current_path, PART_SLASH);
    std::shared_ptr<IndexNode> node = root_;
    for (auto& path_slice : path_slices) {
        node = node->GetChild(path_slice, false);
        if (node == nullptr) {
            auto ret = Dataset::Make();
            ret->Dim(0)->NumElements(1);
            return ret;
        }
    }
    auto search_result = node->SearchGraph(search_func);
    while (search_result.size() > limit) {
        search_result.pop();
    }

    // return result
    auto result = Dataset::Make();
    auto target_size = static_cast<int64_t>(search_result.size());
    if (target_size == 0) {
        result->Dim(0)->NumElements(1);
        return result;
    }
    result->Dim(target_size)->NumElements(1)->Owner(true, common_param_.allocator_.get());
    auto* ids = (int64_t*)common_param_.allocator_->Allocate(sizeof(int64_t) * target_size);
    result->Ids(ids);
    auto* dists = (float*)common_param_.allocator_->Allocate(sizeof(float) * target_size);
    result->Distances(dists);
    for (auto j = target_size - 1; j >= 0; --j) {
        if (j < target_size) {
            dists[j] = search_result.top().first;
            ids[j] = labels_.GetLabelById(search_result.top().second);
        }
        search_result.pop();
    }
    return result;
}

tl::expected<BinarySet, Error>
Pyramid::Serialize() const {
    if (GetNumElements() == 0) {
        return EmptyIndexBinarySet::Make("EMPTY_PYRAMID");
    }
    SlowTaskTimer t("Pyramid Serialize");
    size_t num_bytes = this->cal_serialize_size();
    try {
        std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
        auto* buffer = reinterpret_cast<char*>(const_cast<int8_t*>(bin.get()));
        BufferStreamWriter writer(buffer);
        this->Serialize(writer);
        Binary b{
            .data = bin,
            .size = num_bytes,
        };
        BinarySet bs;
        bs.Set(INDEX_PYRAMID, b);

        return bs;
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to Serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
Pyramid::Deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("pyramid Deserialize");
    if (this->GetNumElements() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to Deserialize: index is not empty");
    }

    // check if binary set is an empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        return {};
    }

    Binary b = binary_set.Get(INDEX_PYRAMID);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor);
        this->Deserialize(reader);
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("pyramid Deserialize");
    if (this->GetNumElements() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to Deserialize: index is not empty");
    }

    try {
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            reader_set.Get(INDEX_PYRAMID)->Read(offset, len, dest);
        };
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor);
        this->Deserialize(reader);
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }

    return {};
}

int64_t
Pyramid::GetNumElements() const {
    return flatten_interface_ptr_->TotalCount();
}

int64_t
Pyramid::GetMemoryUsage() const {
    return 0;
}

void
Pyramid::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, labels_.label_table_);
    flatten_interface_ptr_->Serialize(writer);
    root_->Serialize(writer);
}

void
Pyramid::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, labels_.label_table_);
    flatten_interface_ptr_->Deserialize(reader);
    root_->Deserialize(reader);
    pool_ = std::make_unique<VisitedListPool>(1,
                                              common_param_.allocator_.get(),
                                              flatten_interface_ptr_->TotalCount(),
                                              common_param_.allocator_.get());
}

uint64_t
Pyramid::cal_serialize_size() const {
    auto cal_size_func = [](uint64_t cursor, uint64_t size, void* buf) { return; };
    WriteFuncStreamWriter writer(cal_size_func, 0);
    this->Serialize(writer);
    return writer.cursor_;
}

tl::expected<void, Error>
Pyramid::Serialize(std::ostream& out_stream) {
    try {
        IOStreamWriter writer(out_stream);
        this->Serialize(writer);
        return {};
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to Serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
Pyramid::Deserialize(std::istream& in_stream) {
    SlowTaskTimer t("pyramid Deserialize");
    if (this->GetNumElements() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to Deserialize: index is not empty");
    }
    try {
        IOStreamReader reader(in_stream);
        this->Deserialize(reader);
        return {};
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to Deserialize(bad alloc): ", e.what());
    }
}
tl::expected<std::vector<int64_t>, Error>
Pyramid::add(const DatasetPtr& base) {
    const auto* path = base->GetPaths();
    int64_t data_num = base->GetNumElements();
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    const auto& no_build_levels = pyramid_param_.no_build_levels;
    int64_t local_cur_element_count = 0;
    {
        std::lock_guard lock(cur_element_count_mutex_);
        local_cur_element_count = cur_element_count_;
        if (max_capacity_ == 0) {
            auto new_capacity = std::max(INIT_CAPACITY, data_num);
            resize(new_capacity);
        } else if (max_capacity_ < data_num + cur_element_count_) {
            auto new_capacity = std::min(MAX_CAPACITY_EXTEND, max_capacity_);
            new_capacity = std::max(data_num + cur_element_count_ - max_capacity_, new_capacity) +
                           max_capacity_;
            resize(new_capacity);
        }
        cur_element_count_ += data_num;
        flatten_interface_ptr_->BatchInsertVector(data_vectors, data_num);
    }
    std::shared_lock<std::shared_mutex> lock(resize_mutex_);

    std::memcpy(labels_.label_table_.data() + local_cur_element_count,
                data_ids,
                sizeof(LabelType) * data_num);

    InnerSearchParam search_param;
    search_param.ef = pyramid_param_.ef_construction;
    search_param.topk = pyramid_param_.odescent_param->max_degree;
    search_param.search_mode = KNN_SEARCH;
    auto empty_mutex = std::make_shared<EmptyMutex>();
    for (auto i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        std::shared_ptr<IndexNode> node = root_;
        auto inner_id = static_cast<InnerIdType>(i + local_cur_element_count);
        for (auto& path_slice : path_slices) {
            node = node->GetChild(path_slice, true);
            std::lock_guard<std::mutex> graph_lock(node->mutex_);
            // add one point
            if (node->graph_ == nullptr) {
                node->has_index_ =
                    std::find(no_build_levels.begin(), no_build_levels.end(), node->level_) ==
                    no_build_levels.end();
                node->InitGraph();
            }
            if (node->graph_->TotalCount() == 0) {
                if (node->has_index_) {
                    node->graph_->InsertNeighborsById(
                        inner_id, Vector<InnerIdType>(common_param_.allocator_.get()));
                    node->entry_point_ = inner_id;
                }
            } else {
                search_param.ep = node->entry_point_;
                auto vl = pool_->TakeOne();
                auto results = searcher_->Search(node->graph_,
                                                 flatten_interface_ptr_,
                                                 vl,
                                                 data_vectors + common_param_.dim_ * i,
                                                 search_param);
                pool_->ReturnOne(vl);
                mutually_connect_new_element(inner_id,
                                             results,
                                             node->graph_,
                                             flatten_interface_ptr_,
                                             empty_mutex,
                                             common_param_.allocator_.get());
            }
        }
    }
    return {};
}

void
Pyramid::resize(int64_t new_max_capacity) {
    std::unique_lock<std::shared_mutex> lock(resize_mutex_);
    if (new_max_capacity <= max_capacity_) {
        return;
    }
    pool_ = std::make_unique<VisitedListPool>(
        1, common_param_.allocator_.get(), new_max_capacity, common_param_.allocator_.get());
    labels_.label_table_.resize(new_max_capacity);
    flatten_interface_ptr_->Resize(new_max_capacity);
    max_capacity_ = new_max_capacity;
}

}  // namespace vsag
