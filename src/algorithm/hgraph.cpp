
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

#include "hgraph.h"

#include <data_cell/compressed_graph_datacell_parameter.h>
#include <fmt/format-inl.h>

#include <memory>
#include <stdexcept>

#include "common.h"
#include "data_cell/graph_datacell_parameter.h"
#include "data_cell/sparse_graph_datacell.h"
#include "dataset_impl.h"
#include "impl/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "index/index_impl.h"
#include "index/iterator_filter.h"
#include "logger.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "typing.h"
#include "utils/standard_heap.h"
#include "utils/util_functions.h"
#include "vsag/options.h"

namespace vsag {

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      route_graphs_(common_param.allocator_.get()),
      use_reorder_(hgraph_param->use_reorder),
      use_elp_optimizer_(hgraph_param->use_elp_optimizer),
      ignore_reorder_(hgraph_param->ignore_reorder),
      build_by_base_(hgraph_param->build_by_base),
      use_attribute_filter_(hgraph_param->use_attribute_filter),
      ef_construct_(hgraph_param->ef_construction),
      build_thread_count_(hgraph_param->build_thread_count),
      odescent_param_(hgraph_param->odescent_param),
      graph_type_(hgraph_param->graph_type),
      extra_info_size_(common_param.extra_info_size_),
      use_old_serial_format_(common_param.use_old_serial_format_),
      deleted_ids_(allocator_) {
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (use_reorder_) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->searcher_ = std::make_shared<BasicSearcher>(common_param, neighbors_mutex_);

    this->bottom_graph_ =
        GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    auto graph_param =
        std::dynamic_pointer_cast<GraphDataCellParameter>(hgraph_param->bottom_graph_param);
    sparse_datacell_param_ = std::make_shared<SparseGraphDatacellParameter>();
    sparse_datacell_param_->max_degree_ = hgraph_param->bottom_graph_param->max_degree_ / 2;
    if (graph_param != nullptr) {
        sparse_datacell_param_->remove_flag_bit_ = graph_param->remove_flag_bit_;
        sparse_datacell_param_->support_delete_ = graph_param->support_remove_;
    } else {
        sparse_datacell_param_->support_delete_ = false;
    }
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    if (extra_info_size_ > 0) {
        this->extra_infos_ =
            ExtraInfoInterface::MakeInstance(hgraph_param->extra_info_param, common_param);
    }

    auto step_block_size = Options::Instance().block_size_limit();
    auto block_size_per_vector = this->basic_flatten_codes_->code_size_;
    block_size_per_vector =
        std::max(block_size_per_vector,
                 static_cast<uint32_t>(this->bottom_graph_->maximum_degree_ * sizeof(InnerIdType)));
    if (use_reorder_) {
        block_size_per_vector =
            std::max(block_size_per_vector, this->high_precise_codes_->code_size_);
    }
    if (extra_infos_ != nullptr) {
        block_size_per_vector =
            std::max(block_size_per_vector, static_cast<uint32_t>(this->extra_info_size_));
    }
    auto increase_count = step_block_size / block_size_per_vector;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));

    resize(bottom_graph_->max_capacity_);
    this->build_pool_ = common_param.thread_pool_;
    if (this->build_thread_count_ > 1 && this->build_pool_ == nullptr) {
        this->build_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        this->build_pool_->SetPoolSize(build_thread_count_);
    }

    UnorderedMap<std::string, float> default_param(common_param.allocator_.get());
    default_param.insert(
        {PREFETCH_DEPTH_CODE, (this->basic_flatten_codes_->code_size_ + 63.0) / 64.0});
    this->basic_flatten_codes_->SetRuntimeParameters(default_param);

    if (use_elp_optimizer_) {
        optimizer_ = std::make_shared<Optimizer<BasicSearcher>>(common_param);
    }
    if (use_attribute_filter_) {
        this->attr_filter_index_ =
            AttributeInvertedInterface::MakeInstance(allocator_, false /*have_bucket*/);
    }
}
void
HGraph::Train(const DatasetPtr& base) {
    const auto* base_data = get_data(base);
    this->basic_flatten_codes_->Train(base_data, base->GetNumElements());
    if (use_reorder_) {
        this->high_precise_codes_->Train(base_data, base->GetNumElements());
    }
}

std::vector<int64_t>
HGraph::Build(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    this->Train(data);
    std::vector<int64_t> ret;
    if (graph_type_ == GRAPH_TYPE_NSW) {
        ret = this->Add(data);
    } else {
        ret = this->build_by_odescent(data);
    }
    if (use_elp_optimizer_) {
        elp_optimize();
    }
    return ret;
}

std::vector<int64_t>
HGraph::build_by_odescent(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* extra_infos = data->GetExtraInfos();
    auto inner_ids = this->get_unique_inner_ids(total);
    Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
    InnerIdType cur_size = 0;
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label)) {
            failed_ids.emplace_back(label);
            continue;
        }
        InnerIdType inner_id = inner_ids.at(cur_size);
        cur_size++;
        this->label_table_->Insert(inner_id, label);
        this->basic_flatten_codes_->InsertVector(vectors + dim_ * i, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(vectors + dim_ * i, inner_id);
        }
        auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(Vector<InnerIdType>(allocator_));
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    this->resize(total_count_);
    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(odescent_param_, build_data, allocator_, this->build_pool_.get());
        odescent_builder.Build();
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& route_graph_id : route_graph_ids) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->build_pool_.get());
        auto graph = this->generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph);
        this->route_graphs_.emplace_back(graph);
    }
    return failed_ids;
}

std::vector<int64_t>
HGraph::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    }
    CHECK_ARGUMENT(get_data(data) != nullptr, "base.float_vector is nullptr");

    {
        std::lock_guard lock(this->add_mutex_);
        if (this->total_count_ == 0) {
            this->Train(data);
        }
    }

    auto add_func = [&](const void* data,
                        int level,
                        InnerIdType inner_id,
                        const char* extra_info,
                        const AttributeSet* attrs) -> void {
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->InsertExtraInfo(extra_info, inner_id);
        }
        if (attrs != nullptr and this->use_attribute_filter_) {
            this->attr_filter_index_->Insert(*attrs, inner_id);
        }
        this->add_one_point(data, level, inner_id);
    };

    std::vector<std::future<void>> futures;
    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    Vector<std::pair<InnerIdType, LabelType>> inner_ids(allocator_);
    for (int64_t j = 0; j < total; ++j) {
        auto label = labels[j];
        InnerIdType inner_id;
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
            {
                std::lock_guard lock(this->add_mutex_);
                inner_id = this->get_unique_inner_ids(1).at(0);
                uint64_t new_count = total_count_;
                this->resize(new_count);
            }
            this->label_table_->Insert(inner_id, label);
            inner_ids.emplace_back(inner_id, j);
        }
    }
    for (auto& [inner_id, local_idx] : inner_ids) {
        int level;
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            level = this->get_random_level() - 1;
        }
        const auto* extra_info = extra_infos + local_idx * extra_info_size_;
        const AttributeSet* cur_attr_set = nullptr;
        if (attr_sets != nullptr) {
            cur_attr_set = attr_sets + local_idx;
        }
        if (this->build_pool_ != nullptr) {
            auto future = this->build_pool_->GeneralEnqueue(
                add_func, get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
            futures.emplace_back(std::move(future));
        } else {
            add_func(get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
        }
    }
    if (this->build_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    return failed_ids;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, nullptr);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator) const {
    int64_t query_dim = query->GetDim();
    Allocator* search_allocator = allocator == nullptr ? allocator_ : allocator;
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;
    search_param.search_alloc = search_allocator;
    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param);
        search_param.ep = result->Top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    auto ef_search_threshold = std::max(AMPLIFICATION_FACTOR * k, 1000L);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        if (params.use_extra_info_filter) {
            ft = std::make_shared<ExtraInfoWrapperFilter>(filter, this->extra_infos_);
        } else {
            ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
        }
    }

    search_param.ef = std::max(params.ef_search, k);
    search_param.is_inner_id_allowed = ft;
    search_param.topk = static_cast<int64_t>(search_param.ef);
    auto search_result = this->search_one_graph(
        raw_query, this->bottom_graph_, this->basic_flatten_codes_, search_param);

    if (use_reorder_) {
        this->reorder(raw_query, this->high_precise_codes_, search_result, k);
    }

    while (search_result->Size() > k) {
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = CreateFastDataset(count, search_allocator);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos = (char*)search_allocator->Allocate(extra_info_size_ * search_result->Size());
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    return std::move(dataset_results);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator,
                  IteratorContext*& iter_ctx,
                  bool is_last_filter) const {
    Allocator* search_allocator = allocator == nullptr ? allocator_ : allocator;
    if (GetNumElements() == 0) {
        return DatasetImpl::MakeEmptyDataset();
    }
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    auto params = HGraphSearchParameters::FromJson(parameters);

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        if (params.use_extra_info_filter) {
            ft = std::make_shared<ExtraInfoWrapperFilter>(filter, this->extra_infos_);
        } else {
            ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
        }
    }

    if (iter_ctx == nullptr) {
        auto cur_count = this->bottom_graph_->TotalCount();
        auto* new_ctx = new IteratorFilterContext();
        if (auto ret = new_ctx->init(cur_count, params.ef_search, search_allocator);
            not ret.has_value()) {
            throw vsag::VsagException(ErrorType::INTERNAL_ERROR,
                                      "failed to init IteratorFilterContext");
        }
        iter_ctx = new_ctx;
    }

    auto* iter_filter_ctx = static_cast<IteratorFilterContext*>(iter_ctx);
    DistHeapPtr search_result = std::make_shared<StandardHeap<true, false>>(search_allocator, -1);
    const auto* query_data = get_data(query);
    if (is_last_filter) {
        while (!iter_filter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_filter_ctx->GetTopID();
            float cur_dist = iter_filter_ctx->GetTopDist();
            search_result->Push(cur_dist, cur_inner_id);
            iter_filter_ctx->PopDiscard();
        }
    } else {
        InnerSearchParam search_param;
        search_param.ep = this->entry_point_id_;
        search_param.topk = 1;
        search_param.ef = 1;
        search_param.is_inner_id_allowed = nullptr;
        search_param.search_alloc = search_allocator;
        if (iter_filter_ctx->IsFirstUsed()) {
            for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
                auto result = this->search_one_graph(
                    query_data, this->route_graphs_[i], this->basic_flatten_codes_, search_param);
                search_param.ep = result->Top().second;
            }
        }

        search_param.ef = std::max(params.ef_search, k);
        search_param.is_inner_id_allowed = ft;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        search_result = this->search_one_graph(query_data,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               iter_filter_ctx);
    }

    if (use_reorder_) {
        this->reorder(query_data, this->high_precise_codes_, search_result, k);
    }

    while (search_result->Size() > k) {
        auto curr = search_result->Top();
        iter_filter_ctx->AddDiscardNode(curr.first, curr.second);
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = CreateFastDataset(count, search_allocator);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos = (char*)search_allocator->Allocate(extra_info_size_ * search_result->Size());
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        iter_filter_ctx->SetPoint(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    iter_filter_ctx->SetOFFFirstUsed();
    return std::move(dataset_results);
}

uint64_t
HGraph::EstimateMemory(uint64_t num_elements) const {
    uint64_t estimate_memory = 0;
    auto block_size = Options::Instance().block_size_limit();
    auto element_count =
        next_multiple_of_power_of_two(num_elements, this->resize_increase_count_bit_);

    auto block_memory_ceil = [](uint64_t memory, uint64_t block_size) -> uint64_t {
        return static_cast<uint64_t>(
            std::ceil(static_cast<double>(memory) / static_cast<double>(block_size)) *
            static_cast<double>(block_size));
    };

    if (this->basic_flatten_codes_->InMemory()) {
        auto base_memory = this->basic_flatten_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(base_memory, block_size);
    }

    if (bottom_graph_->InMemory()) {
        auto bottom_graph_memory =
            (this->bottom_graph_->maximum_degree_ + 1) * sizeof(InnerIdType) * element_count;
        estimate_memory += block_memory_ceil(bottom_graph_memory, block_size);
    }

    if (use_reorder_ && this->high_precise_codes_->InMemory() && not this->ignore_reorder_) {
        auto precise_memory = this->high_precise_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(precise_memory, block_size);
    }

    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && this->extra_infos_->InMemory()) {
        auto extra_info_memory = this->extra_infos_->ExtraInfoSize() * element_count;
        estimate_memory += block_memory_ceil(extra_info_memory, block_size);
    }

    auto label_map_memory =
        element_count * (sizeof(std::pair<LabelType, InnerIdType>) + 2 * sizeof(void*));
    estimate_memory += label_map_memory;

    auto sparse_graph_memory = (this->mult_ * 0.05 * static_cast<double>(element_count)) *
                               sizeof(InnerIdType) *
                               (static_cast<double>(this->bottom_graph_->maximum_degree_) / 2 + 1);
    estimate_memory += static_cast<uint64_t>(sparse_graph_memory);

    auto other_memory = element_count * (sizeof(LabelType) + sizeof(std::shared_mutex) +
                                         sizeof(std::shared_ptr<std::shared_mutex>));
    estimate_memory += other_memory;

    return estimate_memory;
}

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(sparse_datacell_param_, this->allocator_);
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param) const {
    auto visited_list = this->pool_->TakeOne();
    auto result = this->searcher_->Search(graph, flatten, visited_list, query, inner_search_param);
    this->pool_->ReturnOne(visited_list);
    return result;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         IteratorFilterContext* iter_ctx) const {
    auto visited_list = this->pool_->TakeOne();
    auto result =
        this->searcher_->Search(graph, flatten, visited_list, query, inner_search_param, iter_ctx);
    this->pool_->ReturnOne(visited_list);
    return result;
}

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    std::shared_ptr<InnerIdWrapperFilter> ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check radius
    CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // check limited_size
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param);
        search_param.ep = result->Top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    CHECK_ARGUMENT((1 <= params.ef_search) and (params.ef_search <= 1000),  // NOLINT
                   fmt::format("ef_search({}) must in range[1, 1000]", params.ef_search));
    search_param.ef = std::max(params.ef_search, limited_size);
    search_param.is_inner_id_allowed = ft;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    search_param.range_search_limit_size = static_cast<int>(limited_size);
    auto search_result = this->search_one_graph(
        raw_query, this->bottom_graph_, this->basic_flatten_codes_, search_param);
    if (use_reorder_) {
        this->reorder(raw_query, this->high_precise_codes_, search_result, limited_size);
    }

    if (limited_size > 0) {
        while (search_result->Size() > limited_size) {
            search_result->Pop();
        }
    }

    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = CreateFastDataset(count, allocator_);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos = (char*)allocator_->Allocate(extra_info_size_ * search_result->Size());
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    return std::move(dataset_results);
}

void
HGraph::serialize_basic_info_v0_14(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    uint64_t max_level = this->route_graphs_.size();
    StreamWriter::WriteObj(writer, max_level);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    auto capacity = this->max_capacity_.load();
    StreamWriter::WriteObj(writer, capacity);
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);

    uint64_t size = this->total_count_;
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::deserialize_basic_info_v0_14(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    uint64_t max_level;
    StreamReader::ReadObj(reader, max_level);
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);
    InnerIdType capacity;
    StreamReader::ReadObj(reader, capacity);
    this->max_capacity_.store(capacity);
    StreamReader::ReadVector(reader, this->label_table_->label_table_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
}

#define TO_JSON(json_obj, var) json_obj[#var] = this->var##_;

#define TO_JSON_BASE64(json_obj, var) json_obj[#var] = base64_encode_obj(this->var##_);

#define TO_JSON_ATOMIC(json_obj, var) json_obj[#var] = this->var##_.load();

JsonType
HGraph::serialize_basic_info() const {
    JsonType jsonify_basic_info;
    TO_JSON(jsonify_basic_info, use_reorder);
    TO_JSON(jsonify_basic_info, dim);
    TO_JSON(jsonify_basic_info, metric);
    TO_JSON(jsonify_basic_info, entry_point_id);
    TO_JSON(jsonify_basic_info, ef_construct);
    // logger::debug("mult: {}", this->mult_);
    TO_JSON_BASE64(jsonify_basic_info, mult);
    TO_JSON_ATOMIC(jsonify_basic_info, max_capacity);
    jsonify_basic_info["max_level"] = this->route_graphs_.size();

    return jsonify_basic_info;
}

#define FROM_JSON(json_obj, var) this->var##_ = (json_obj)[#var];

#define FROM_JSON_BASE64(json_obj, var) base64_decode_obj((json_obj)[#var], this->var##_);

#define FROM_JSON_ATOMIC(json_obj, var) this->var##_.store((json_obj)[#var]);

void
HGraph::deserialize_basic_info(JsonType jsonify_basic_info) {
    logger::debug("jsonify_basic_info: {}", jsonify_basic_info.dump());
    FROM_JSON(jsonify_basic_info, use_reorder);
    FROM_JSON(jsonify_basic_info, dim);
    FROM_JSON(jsonify_basic_info, metric);
    FROM_JSON(jsonify_basic_info, entry_point_id);
    FROM_JSON(jsonify_basic_info, ef_construct);
    FROM_JSON_BASE64(jsonify_basic_info, mult);
    // logger::debug("mult: {}", this->mult_);
    FROM_JSON_ATOMIC(jsonify_basic_info, max_capacity);

    uint64_t max_level = jsonify_basic_info["max_level"];
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
}

void
HGraph::serialize_label_info(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);
    uint64_t size = this->label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::deserialize_label_info(StreamReader& reader) const {
    StreamReader::ReadVector(reader, this->label_table_->label_table_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
}

void
HGraph::Serialize(StreamWriter& writer) const {
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }

    // FIXME(wxyu): this option is used for special purposes, like compatibility testing
    if (this->use_old_serial_format_) {
        this->serialize_basic_info_v0_14(writer);
        this->basic_flatten_codes_->Serialize(writer);
        this->bottom_graph_->Serialize(writer);
        if (this->use_reorder_) {
            this->high_precise_codes_->Serialize(writer);
        }
        for (const auto& route_graph : this->route_graphs_) {
            route_graph->Serialize(writer);
        }
        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Serialize(writer);
        }
        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Serialize(writer);
        }
        return;
    }

    this->serialize_label_info(writer);
    this->basic_flatten_codes_->Serialize(writer);
    this->bottom_graph_->Serialize(writer);
    if (this->use_reorder_) {
        this->high_precise_codes_->Serialize(writer);
    }
    for (const auto& route_graph : this->route_graphs_) {
        route_graph->Serialize(writer);
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        this->extra_infos_->Serialize(writer);
    }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }

    // serialize footer (introduced since v0.15)
    auto jsonify_basic_info = this->serialize_basic_info();
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("basic_info", jsonify_basic_info);
    logger::debug(jsonify_basic_info.dump());

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
HGraph::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        this->deserialize_basic_info_v0_14(reader);

        this->basic_flatten_codes_->Deserialize(reader);
        this->bottom_graph_->Deserialize(reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(reader);
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        // metadata should NOT be nullptr if footer is not nullptr
        this->deserialize_basic_info(metadata->Get("basic_info"));
        this->deserialize_label_info(reader);

        this->basic_flatten_codes_->Deserialize(reader);
        this->bottom_graph_->Deserialize(reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(reader);
        }
    }

    // post serialize procedure
    if (use_elp_optimizer_) {
        elp_optimize();
    }
}

std::string
HGraph::GetMemoryUsageDetail() const {
    JsonType memory_usage;
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }
    memory_usage["basic_flatten_codes"] = this->basic_flatten_codes_->CalcSerializeSize();
    memory_usage["bottom_graph"] = this->bottom_graph_->CalcSerializeSize();
    if (this->use_reorder_) {
        memory_usage["high_precise_codes"] = this->high_precise_codes_->CalcSerializeSize();
    }
    size_t route_graph_size = 0;
    for (const auto& route_graph : this->route_graphs_) {
        route_graph_size += route_graph->CalcSerializeSize();
    }
    memory_usage["route_graph"] = route_graph_size;
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        memory_usage["extra_infos"] = this->extra_infos_->CalcSerializeSize();
    }
    memory_usage["__total_size__"] = this->CalSerializeSize();
    return memory_usage.dump();
}

float
HGraph::CalcDistanceById(const float* query, int64_t id) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_) {
        flat = this->high_precise_codes_;
    }
    float result = 0.0F;
    auto computer = flat->FactoryComputer(query);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        auto new_id = this->label_table_->GetIdByLabel(id);
        flat->Query(&result, computer, &new_id, 1);
        return result;
    }
}

DatasetPtr
HGraph::CalDistanceById(const float* query, const int64_t* ids, int64_t count) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_) {
        flat = this->high_precise_codes_;
    }
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = (float*)allocator_->Allocate(sizeof(float) * count);
    result->Distances(distances);
    auto computer = flat->FactoryComputer(query);
    Vector<InnerIdType> inner_ids(count, 0, allocator_);
    Vector<InnerIdType> invalid_id_loc(allocator_);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        for (int64_t i = 0; i < count; ++i) {
            try {
                inner_ids[i] = this->label_table_->GetIdByLabel(ids[i]);
            } catch (std::runtime_error& e) {
                logger::debug(fmt::format("failed to find id: {}", ids[i]));
                invalid_id_loc.push_back(i);
            }
        }
        flat->Query(distances, computer, inner_ids.data(), count);
        for (unsigned int i : invalid_id_loc) {
            distances[i] = -1;
        }
    }
    return result;
}

std::pair<int64_t, int64_t>
HGraph::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->total_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->total_count_; ++i) {
        if (not deleted_ids_.empty() && deleted_ids_.count(i) != 0) {
            continue;
        }
        auto label = this->label_table_->label_table_[i];
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

void
HGraph::GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const {
    if (this->extra_infos_ == nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "extra_info is NULL");
    }
    for (int64_t i = 0; i < count; ++i) {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        auto inner_id = this->label_table_->GetIdByLabel(ids[i]);
        this->extra_infos_->GetExtraInfoById(inner_id, extra_infos + i * extra_info_size_);
    }
}

void
HGraph::add_one_point(const void* data, int level, InnerIdType inner_id) {
    this->basic_flatten_codes_->InsertVector(data, inner_id);
    if (use_reorder_) {
        this->high_precise_codes_->InsertVector(data, inner_id);
    }
    std::unique_lock add_lock(add_mutex_);
    if (level >= static_cast<int>(this->route_graphs_.size()) || bottom_graph_->TotalCount() == 0) {
        std::lock_guard<std::shared_mutex> wlock(this->global_mutex_);
        // level maybe a negative number(-1)
        for (auto j = static_cast<int>(this->route_graphs_.size()); j <= level; ++j) {
            this->route_graphs_.emplace_back(this->generate_one_route_graph());
        }
        this->graph_add_one(data, level, inner_id);
        entry_point_id_ = inner_id;
        add_lock.unlock();
    } else {
        add_lock.unlock();
        std::shared_lock<std::shared_mutex> rlock(this->global_mutex_);
        this->graph_add_one(data, level, inner_id);
    }
}

void
HGraph::graph_add_one(const void* data, int level, InnerIdType inner_id) {
    DistHeapPtr result = nullptr;
    InnerSearchParam param{
        .topk = 1,
        .ep = this->entry_point_id_,
        .ef = 1,
        .is_inner_id_allowed = nullptr,
    };

    LockGuard cur_lock(neighbors_mutex_, inner_id);
    auto flatten_codes = basic_flatten_codes_;
    if (use_reorder_ and not build_by_base_) {
        flatten_codes = high_precise_codes_;
    }
    for (auto j = this->route_graphs_.size() - 1; j > level; --j) {
        result = search_one_graph(data, route_graphs_[j], flatten_codes, param);
        param.ep = result->Top().second;
    }

    param.ef = this->ef_construct_;
    param.topk = static_cast<int64_t>(ef_construct_);

    if (bottom_graph_->TotalCount() != 0) {
        result = search_one_graph(data, this->bottom_graph_, flatten_codes, param);
        mutually_connect_new_element(
            inner_id, result, this->bottom_graph_, flatten_codes, neighbors_mutex_, allocator_);
    } else {
        bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
    }

    for (int64_t j = 0; j <= level; ++j) {
        if (route_graphs_[j]->TotalCount() != 0) {
            result = search_one_graph(data, route_graphs_[j], flatten_codes, param);
            mutually_connect_new_element(
                inner_id, result, route_graphs_[j], flatten_codes, neighbors_mutex_, allocator_);
        } else {
            route_graphs_[j]->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        }
    }
}

void
HGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_.load();
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_);
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::lock_guard lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->neighbors_mutex_->Resize(new_size_power_2);
        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size_power_2, allocator_);
        this->label_table_->Resize(new_size_power_2);
        bottom_graph_->Resize(new_size_power_2);
        this->max_capacity_.store(new_size_power_2);
        this->basic_flatten_codes_->Resize(new_size_power_2);
        if (use_reorder_) {
            this->high_precise_codes_->Resize(new_size_power_2);
        }
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->Resize(new_size_power_2);
        }
    }
}
void
HGraph::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_MERGE_INDEX,
    });
    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_KNN_ITERATOR_FILTER_SEARCH,
    });
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_CONCURRENT);
    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
    });
    // other
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
        IndexFeature::SUPPORT_CLONE,
        IndexFeature::SUPPORT_EXPORT_MODEL,
    });

    // About Train
    auto name = this->basic_flatten_codes_->GetQuantizerName();

    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }

    bool have_fp32 = false;
    if (name == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
    }
    if (use_reorder_ and
        this->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
    }
    if (have_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
    }

    // metric
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_L2);
    } else if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_COSINE);
    }

    if (this->extra_infos_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER);
    }
}

void
HGraph::elp_optimize() {
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 80;
    param.topk = 10;
    param.is_inner_id_allowed = nullptr;
    searcher_->SetMockParameters(bottom_graph_, basic_flatten_codes_, pool_, param, dim_);
    // TODO(ZXY): optimize PREFETCH_DEPTH_CODE and add default value for the others
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 10, 1));
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 10, 1));
    optimizer_->Optimize(searcher_);
}

void
HGraph::reorder(const void* query,
                const FlattenInterfacePtr& flatten_interface,
                const DistHeapPtr& candidate_heap,
                int64_t k) const {
    uint64_t size = candidate_heap->Size();
    if (k <= 0) {
        k = static_cast<int64_t>(size);
    }
    Vector<InnerIdType> ids(size, allocator_);
    Vector<float> dists(size, allocator_);
    uint64_t idx = 0;
    while (not candidate_heap->Empty()) {
        ids[idx] = candidate_heap->Top().second;
        ++idx;
        candidate_heap->Pop();
    }
    auto computer = flatten_interface->FactoryComputer(query);
    flatten_interface->Query(dists.data(), computer, ids.data(), size);
    for (uint64_t i = 0; i < size; ++i) {
        if (candidate_heap->Size() < k or dists[i] <= candidate_heap->Top().first) {
            candidate_heap->Push(dists[i], ids[i]);
        }
        if (candidate_heap->Size() > k) {
            candidate_heap->Pop();
        }
    }
}

static const std::string HGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_HGRAPH}",
        "{HGRAPH_USE_REORDER_KEY}": false,
        "{HGRAPH_USE_ENV_OPTIMIZER}": false,
        "{HGRAPH_IGNORE_REORDER_KEY}": false,
        "{HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY}": false,
        "{HGRAPH_USE_ATTRIBUTE_FILTER_KEY}": false,
        "{HGRAPH_GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_TYPE_KEY}": "{GRAPH_TYPE_NSW}",
            "{GRAPH_STORAGE_TYPE_KEY}": "{GRAPH_STORAGE_TYPE_FLAT}",
            "{ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}": 10000,
            "{ODESCENT_PARAMETER_MIN_IN_DEGREE}": 1,
            "{ODESCENT_PARAMETER_ALPHA}": 1.2,
            "{ODESCENT_PARAMETER_GRAPH_ITER_TURN}": 30,
            "{ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}": 0.2,
            "{GRAPH_PARAM_MAX_DEGREE}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 100,
            "{GRAPH_SUPPORT_REMOVE}": false,
            "{REMOVE_FLAG_BIT}": 8
        },
        "{HGRAPH_BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "{PCA_DIM}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY}": 32,
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM}": 1
            }
        },
        "{HGRAPH_PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "{PCA_DIM}": 0,
                "{PRODUCT_QUANTIZATION_DIM}": 1
            }
        },
        "{BUILD_PARAMS_KEY}": {
            "{BUILD_EF_CONSTRUCTION}": 400,
            "{BUILD_THREAD_COUNT}": 100
        },
        "{HGRAPH_EXTRA_INFO_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            }
        }
    })";

ParamPtr
HGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {
            HGRAPH_USE_REORDER,
            {
                HGRAPH_USE_REORDER_KEY,
            },
        },
        {
            HGRAPH_USE_ELP_OPTIMIZER,
            {
                HGRAPH_USE_ELP_OPTIMIZER_KEY,
            },
        },
        {
            HGRAPH_IGNORE_REORDER,
            {
                HGRAPH_IGNORE_REORDER_KEY,
            },
        },
        {
            HGRAPH_BUILD_BY_BASE_QUANTIZATION,
            {
                HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY,
            },
        },
        {
            HGRAPH_USE_ATTRIBUTE_FILTER,
            {
                HGRAPH_USE_ATTRIBUTE_FILTER_KEY,
            },
        },
        {
            HGRAPH_BASE_QUANTIZATION_TYPE,
            {
                HGRAPH_BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                QUANTIZATION_TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_IO_TYPE,
            {
                HGRAPH_BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_TYPE_KEY,
            },
        },
        {
            HGRAPH_PRECISE_IO_TYPE,
            {
                HGRAPH_PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_FILE_PATH,
            {
                HGRAPH_BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH,
            },
        },
        {
            HGRAPH_PRECISE_FILE_PATH,
            {
                HGRAPH_PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH,
            },
        },
        {
            HGRAPH_PRECISE_QUANTIZATION_TYPE,
            {
                HGRAPH_PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                QUANTIZATION_TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_MAX_DEGREE,
            {
                HGRAPH_GRAPH_KEY,
                GRAPH_PARAM_MAX_DEGREE,
            },
        },
        {
            HGRAPH_BUILD_EF_CONSTRUCTION,
            {
                BUILD_PARAMS_KEY,
                BUILD_EF_CONSTRUCTION,
            },
        },
        {
            HGRAPH_INIT_CAPACITY,
            {
                HGRAPH_GRAPH_KEY,
                GRAPH_PARAM_INIT_MAX_CAPACITY,
            },
        },
        {
            HGRAPH_GRAPH_TYPE,
            {
                HGRAPH_GRAPH_KEY,
                GRAPH_TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_STORAGE_TYPE,
            {
                HGRAPH_GRAPH_KEY,
                GRAPH_STORAGE_TYPE_KEY,
            },
        },
        {
            ODESCENT_PARAMETER_ALPHA,
            {
                HGRAPH_GRAPH_KEY,
                ODESCENT_PARAMETER_ALPHA,
            },
        },
        {
            ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            {
                HGRAPH_GRAPH_KEY,
                ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            },
        },
        {
            ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            {
                HGRAPH_GRAPH_KEY,
                ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            },
        },
        {
            ODESCENT_PARAMETER_MIN_IN_DEGREE,
            {
                HGRAPH_GRAPH_KEY,
                ODESCENT_PARAMETER_MIN_IN_DEGREE,
            },
        },
        {
            ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            {
                HGRAPH_GRAPH_KEY,
                ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            },
        },
        {
            HGRAPH_BUILD_THREAD_COUNT,
            {
                BUILD_PARAMS_KEY,
                BUILD_THREAD_COUNT,
            },
        },
        {
            SQ4_UNIFORM_TRUNC_RATE,
            {
                HGRAPH_BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE,
            },
        },
        {
            RABITQ_PCA_DIM,
            {
                HGRAPH_BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PCA_DIM,
            },
        },
        {
            RABITQ_BITS_PER_DIM_QUERY,
            {
                HGRAPH_BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY,
            },
        },
        {
            HGRAPH_BASE_PQ_DIM,
            {
                HGRAPH_BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM,
            },
        },
        {
            RABITQ_USE_FHT,
            {
                HGRAPH_BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                USE_FHT,
            },
        },
        {
            HGRAPH_SUPPORT_REMOVE,
            {HGRAPH_GRAPH_KEY, GRAPH_SUPPORT_REMOVE},
        },
        {
            HGRAPH_REMOVE_FLAG_BIT,
            {HGRAPH_GRAPH_KEY, REMOVE_FLAG_BIT},
        },
    };
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("HGraph not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(HGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->data_type = common_param.data_type_;
    hgraph_parameter->FromJson(inner_json);
    uint64_t max_degree = hgraph_parameter->bottom_graph_param->max_degree_;

    auto max_degree_threshold = std::max(common_param.dim_, 128L);
    CHECK_ARGUMENT(  // NOLINT
        (4 <= max_degree) and (max_degree <= max_degree_threshold),
        fmt::format("max_degree({}) must in range[4, {}]", max_degree, max_degree_threshold));

    auto construction_threshold = std::max(1000UL, AMPLIFICATION_FACTOR * max_degree);
    CHECK_ARGUMENT((max_degree <= hgraph_parameter->ef_construction) and  // NOLINT
                       (hgraph_parameter->ef_construction <= construction_threshold),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), {}]",
                               hgraph_parameter->ef_construction,
                               max_degree,
                               construction_threshold));
    return hgraph_parameter;
}
InnerIndexPtr
HGraph::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<HGraph>(this->create_param_ptr_, param);
    this->basic_flatten_codes_->ExportModel(index->basic_flatten_codes_);
    if (use_reorder_) {
        this->high_precise_codes_->ExportModel(index->high_precise_codes_);
    }
    return index;
}
void
HGraph::GetRawData(vsag::InnerIdType inner_id, uint8_t* data) const {
    if (use_reorder_) {
        high_precise_codes_->GetCodesById(inner_id, data);
    } else {
        basic_flatten_codes_->GetCodesById(inner_id, data);
    }
}

bool
HGraph::Remove(int64_t id) {
    // TODO(inbao): support thread safe remove
    auto inner_id = this->label_table_->GetIdByLabel(id);
    if (inner_id == this->entry_point_id_) {
        bool find_new_ep = false;
        while (not route_graphs_.empty()) {
            auto& upper_graph = route_graphs_.back();
            Vector<InnerIdType> neighbors(allocator_);
            upper_graph->GetNeighbors(this->entry_point_id_, neighbors);
            for (const auto& nb_id : neighbors) {
                if (inner_id == nb_id) {
                    continue;
                }
                this->entry_point_id_ = nb_id;
                find_new_ep = true;
                break;
            }
            if (find_new_ep) {
                break;
            }
            route_graphs_.pop_back();
        }
    }
    for (int level = static_cast<int>(route_graphs_.size()) - 1; level >= 0; --level) {
        this->route_graphs_[level]->DeleteNeighborsById(inner_id);
    }
    this->bottom_graph_->DeleteNeighborsById(inner_id);
    this->label_table_->Remove(id);
    this->deleted_ids_.insert(inner_id);
    delete_count_++;
    return true;
}

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, this->total_count_);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (use_reorder_) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, this->total_count_);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, this->total_count_);
        if (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < other_index->route_graphs_.size(); ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], this->total_count_);
        }
        this->total_count_ += other_index->GetNumElements();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(odescent_param_, build_data, allocator_, this->build_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->build_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
}

void
HGraph::SetImmutable() {
    if (this->immutable_) {
        return;
    }
    std::lock_guard<std::shared_mutex> wlock(this->global_mutex_);
    this->neighbors_mutex_.reset();
    this->neighbors_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->neighbors_mutex_);
    this->immutable_ = true;
}

}  // namespace vsag
