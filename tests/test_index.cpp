
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

#include "test_index.h"

#include "fixtures/memory_record_allocator.h"
#include "fixtures/test_logger.h"
#include "fixtures/test_reader.h"
#include "fixtures/thread_pool.h"
#include "index/hnsw.h"
#include "simd/fp32_simd.h"
#include "vsag/engine.h"
#include "vsag/resource.h"
#include "vsag/search_param.h"

namespace fixtures {
static int64_t
Intersection(const int64_t* x, int64_t x_count, const int64_t* y, int64_t y_count) {
    std::unordered_set<int64_t> set_x(x, x + x_count);
    int result = 0;

    for (int i = 0; i < y_count; ++i) {
        if (set_x.count(y[i])) {
            ++result;
        }
    }
    return result;
}

void
TestIndex::TestBuildIndex(const IndexPtr& index,
                          const TestDatasetPtr& dataset,
                          bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_BUILD)) {
        return;
    }
    auto build_index = index->Build(dataset->base_);
    if (expected_success) {
        REQUIRE(build_index.has_value());
        // check the number of vectors in index
        REQUIRE(index->GetNumElements() == dataset->base_->GetNumElements());
    } else {
        REQUIRE(build_index.has_value() == expected_success);
    }
}

void
TestIndex::TestAddIndex(const IndexPtr& index,
                        const TestDatasetPtr& dataset,
                        bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
        return;
    }
    auto add_index = index->Add(dataset->base_);
    if (expected_success) {
        REQUIRE(add_index.has_value());
        // check the number of vectors in index
        REQUIRE(index->GetNumElements() == dataset->base_->GetNumElements());
    } else {
        REQUIRE(not add_index.has_value());
    }
}

void
TestIndex::TestUpdateId(const IndexPtr& index,
                        const TestDatasetPtr& dataset,
                        const std::string& search_param,
                        bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_UPDATE_ID_CONCURRENT)) {
        return;
    }
    auto ids = dataset->base_->GetIds();
    auto num_vectors = dataset->base_->GetNumElements();
    auto dim = dataset->base_->GetDim();
    auto gt_topK = dataset->top_k;
    auto base = dataset->base_->GetFloat32Vectors();

    std::unordered_map<int64_t, int64_t> update_id_map;
    std::unordered_map<int64_t, int64_t> reverse_id_map;
    int64_t max_id = num_vectors;
    for (int i = 0; i < num_vectors; i++) {
        if (ids[i] > max_id) {
            max_id = ids[i];
        }
    }
    for (int i = 0; i < num_vectors; i++) {
        update_id_map[ids[i]] = ids[i] + 2 * max_id;
    }

    std::vector<int> correct_num = {0, 0};
    for (int round = 0; round < 2; round++) {
        // round 0 for update, round 1 for validate update results
        for (int i = 0; i < num_vectors; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(base + i * dim)->Owner(false);

            auto result = index->KnnSearch(query, gt_topK, search_param);
            REQUIRE(result.has_value());

            if (round == 0) {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }

                auto succ_update_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(succ_update_res.has_value());
                if (expected_success) {
                    if (index->CheckFeature(vsag::IndexFeature::SUPPORT_CHECK_ID_EXIST)) {
                        REQUIRE(index->CheckIdExist(ids[i]) == false);
                        REQUIRE(index->CheckIdExist(update_id_map[ids[i]]) == true);
                    }
                    REQUIRE(succ_update_res.value());
                }

                // old id don't exist
                auto failed_old_res = index->UpdateId(ids[i], update_id_map[ids[i]]);
                REQUIRE(failed_old_res.has_value());
                REQUIRE(not failed_old_res.value());

                // same id
                auto succ_same_res = index->UpdateId(update_id_map[ids[i]], update_id_map[ids[i]]);
                REQUIRE(succ_same_res.has_value());
                REQUIRE(succ_same_res.value());
            } else {
                if (result.value()->GetIds()[0] == update_id_map[ids[i]]) {
                    correct_num[round] += 1;
                }
            }
        }

        for (int i = 0; i < num_vectors; i++) {
            if (round == 0) {
                // new id is used
                auto failed_new_res =
                    index->UpdateId(update_id_map[ids[i]], update_id_map[ids[num_vectors - i - 1]]);
                REQUIRE(failed_new_res.has_value());
                REQUIRE(not failed_new_res.value());
            }
        }
    }

    REQUIRE(correct_num[0] == correct_num[1]);
}

void
TestIndex::TestUpdateVector(const IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_UPDATE_VECTOR_CONCURRENT)) {
        return;
    }
    auto ids = dataset->base_->GetIds();
    auto num_vectors = dataset->base_->GetNumElements();
    auto dim = dataset->base_->GetDim();
    auto gt_topK = dataset->top_k;
    auto base = dataset->base_->GetFloat32Vectors();

    int64_t max_id = num_vectors;
    for (int i = 0; i < num_vectors; i++) {
        if (ids[i] > max_id) {
            max_id = ids[i];
        }
    }

    std::vector<int> correct_num = {0, 0};
    uint32_t success_force_updated = 0, failed_force_updated = 0;
    for (int round = 0; round < 2; round++) {
        // round 0 for update, round 1 for validate update results
        for (int i = 0; i < num_vectors; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(base + i * dim)->Owner(false);

            auto result = index->KnnSearch(query, gt_topK, search_param);
            REQUIRE(result.has_value());

            if (round == 0) {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }

                std::vector<float> update_vecs(dim);
                std::vector<float> far_vecs(dim);
                for (int d = 0; d < dim; d++) {
                    update_vecs[d] = base[i * dim + d] + 0.001f;
                    far_vecs[d] = base[i * dim + d] + 1.0f;
                }
                auto new_base = vsag::Dataset::Make();
                new_base->NumElements(1)
                    ->Dim(dim)
                    ->Float32Vectors(update_vecs.data())
                    ->Owner(false);

                // success case
                auto before_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                auto succ_vec_res = index->UpdateVector(ids[i], new_base);
                REQUIRE(succ_vec_res.has_value());
                if (expected_success) {
                    REQUIRE(succ_vec_res.value());
                }
                auto after_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                REQUIRE(before_update_dist < after_update_dist);

                // update with far vector
                new_base->Float32Vectors(far_vecs.data());
                auto fail_vec_res = index->UpdateVector(ids[i], new_base);
                REQUIRE(fail_vec_res.has_value());
                if (fail_vec_res.value()) {
                    // note that the update should be failed, but for some cases, it success
                    auto force_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                    REQUIRE(after_update_dist < force_update_dist);
                    success_force_updated++;
                } else {
                    failed_force_updated++;
                }

                // force update with far vector
                new_base->Float32Vectors(far_vecs.data());
                auto force_update_res1 = index->UpdateVector(ids[i], new_base, true);
                REQUIRE(force_update_res1.has_value());
                if (expected_success) {
                    REQUIRE(force_update_res1.value());
                    auto force_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                    REQUIRE(after_update_dist < force_update_dist);
                }

                new_base->Float32Vectors(update_vecs.data());
                auto force_update_res2 = index->UpdateVector(ids[i], new_base, true);
                REQUIRE(force_update_res2.has_value());
                if (expected_success) {
                    REQUIRE(force_update_res2.value());
                    auto force_update_dist = *index->CalcDistanceById(base + i * dim, ids[i]);
                    REQUIRE(std::abs(after_update_dist - force_update_dist) < 1e-5);
                }

                // old id don't exist
                auto failed_old_res = index->UpdateVector(ids[i] + 2 * max_id, new_base);
                REQUIRE(failed_old_res.has_value());
                REQUIRE(not failed_old_res.value());
            } else {
                if (result.value()->GetIds()[0] == ids[i]) {
                    correct_num[round] += 1;
                }
            }
        }
    }

    REQUIRE(correct_num[0] == correct_num[1]);
    REQUIRE(success_force_updated < failed_force_updated);
}

void
TestIndex::TestContinueAdd(const IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
        return;
    }
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = std::max(1L, dataset->base_->GetNumElements() / 2);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Paths(dataset->base_->GetPaths())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Build(temp_dataset);
    for (uint64_t j = temp_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Paths(dataset->base_->GetPaths() + j)
            ->SparseVectors(dataset->base_->GetSparseVectors() + j)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        if (expected_success) {
            REQUIRE(add_index.has_value());
            // check the number of vectors in index
            REQUIRE(index->GetNumElements() == (j + 1));
        } else {
            REQUIRE(not add_index.has_value());
        }
    }
}

void
TestIndex::TestTrainAndAdd(const TestIndex::IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           bool expected_success) {
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = std::max(1L, static_cast<int64_t>(dataset->base_->GetNumElements() * 0.8));
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Paths(dataset->base_->GetPaths())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Train(dataset->base_);
    index->Add(temp_dataset);
    for (uint64_t j = temp_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Paths(dataset->base_->GetPaths() + j)
            ->SparseVectors(dataset->base_->GetSparseVectors() + j)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        if (expected_success) {
            REQUIRE(add_index.has_value());
            // check the number of vectors in index
            REQUIRE(index->GetNumElements() == (j + 1));
        } else {
            REQUIRE(not add_index.has_value());
        }
    }
}

void
TestIndex::TestKnnSearchCompare(const IndexPtr& index_weak,
                                const IndexPtr& index_strong,
                                const TestDatasetPtr& dataset,
                                const std::string& search_param,
                                bool expected_success) {
    if (not index_weak->CheckFeature(vsag::SUPPORT_KNN_SEARCH) or
        not index_strong->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
        return;
    }

    double time_cost_weak = 0;
    double time_cost_strong = 0;

    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = dataset->top_k;
    for (auto round = 0; round < 2; round++) {
        for (auto i = 0; i < query_count; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(dim)
                ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
                ->SparseVectors(queries->GetSparseVectors() + i)
                ->Paths(queries->GetPaths() + i)
                ->Owner(false);

            if (round == 0) {
                auto st = std::chrono::high_resolution_clock::now();
                auto res = index_weak->KnnSearch(query, topk, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_weak += std::chrono::duration<double>(ed - st).count();
            } else {
                auto st = std::chrono::high_resolution_clock::now();
                auto res = index_strong->KnnSearch(query, topk, search_param);
                auto ed = std::chrono::high_resolution_clock::now();
                time_cost_strong += std::chrono::duration<double>(ed - st).count();
            }
        }
    }
    if (expected_success) {
        REQUIRE(time_cost_weak > time_cost_strong);
    }
}

void
TestIndex::TestKnnSearch(const IndexPtr& index,
                         const TestDatasetPtr& dataset,
                         const std::string& search_param,
                         float expected_recall,
                         bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestRangeSearch(const IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param,
                           float expected_recall,
                           int64_t limited_size,
                           bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH)) {
        return;
    }
    auto queries = dataset->range_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->range_ground_truth_;
    auto gt_topK = gts->GetDim();
    const auto& radius = dataset->range_radius_;
    float cur_recall = 0.0f;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        auto res = index->RangeSearch(query, radius[i], search_param, limited_size);
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        if (limited_size > 0) {
            REQUIRE(res.value()->GetDim() <= limited_size);
        }
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK - 1, result, res.value()->GetDim());
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK - 1);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

class FilterObj : public vsag::Filter {
public:
    FilterObj(std::function<bool(int64_t)> filter_func,
              std::function<bool(const char*)> ex_filter_func,
              float valid_ratio)
        : filter_func_(std::move(filter_func)),
          ex_filter_func_(std::move(ex_filter_func)),
          valid_ratio_(valid_ratio) {
    }

    bool
    CheckValid(int64_t id) const override {
        return not filter_func_(id);
    }

    bool
    CheckValid(const char* data) const override {
        if (not ex_filter_func_)
            return vsag::Filter::CheckValid(data);
        else
            return not ex_filter_func_(data);
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    std::function<bool(int64_t)> filter_func_{nullptr};
    std::function<bool(const char*)> ex_filter_func_{nullptr};
    float valid_ratio_{1.0F};
};

void
TestIndex::TestKnnSearchIter(const IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             float expected_recall,
                             bool expected_success,
                             bool use_ex_filter) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_ITERATOR_FILTER_SEARCH)) {
        return;
    }
    if (use_ex_filter && not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = use_ex_filter ? dataset->ex_filter_ground_truth_ : dataset->filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    auto filter = std::make_shared<FilterObj>(
        dataset->filter_function_, dataset->ex_filter_function_, dataset->valid_ratio_);
    int64_t first_top = topk / 3;
    int64_t second_top = topk / 3;
    int64_t third_top = topk - first_top - second_top;
    std::vector<int64_t> ids(topk);
    for (auto i = 0; i < query_count; ++i) {
        vsag::IteratorContext* filter_ctx = nullptr;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        auto res = index->KnnSearch(query, first_top, search_param, filter, filter_ctx, false);
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        int64_t get_cnt = res.value()->GetDim();
        REQUIRE(res.value()->GetDim() == first_top);
        memcpy(ids.data(), res.value()->GetIds(), sizeof(int64_t) * first_top);
        auto res2 = index->KnnSearch(query, second_top, search_param, filter, filter_ctx, false);
        REQUIRE(res2.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res2.value()->GetDim() == second_top);
        memcpy(ids.data() + first_top, res2.value()->GetIds(), sizeof(int64_t) * second_top);
        auto res3 = index->KnnSearch(query, third_top, search_param, filter, filter_ctx, false);
        REQUIRE(res3.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res3.value()->GetDim() == third_top);
        memcpy(ids.data() + first_top + second_top,
               res3.value()->GetIds(),
               sizeof(int64_t) * third_top);
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, ids.data(), topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
        delete filter_ctx;
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestFilterSearch(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            const std::string& search_param,
                            float expected_recall,
                            bool expected_success,
                            bool support_filter_obj) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_ID_FILTER)) {
        return;
    }
    auto queries = dataset->filter_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        tl::expected<DatasetPtr, vsag::Error> res;
        res = index->KnnSearch(query, topk, search_param, dataset->filter_function_);
        if (support_filter_obj) {
            auto filter = std::make_shared<FilterObj>(
                dataset->filter_function_, dataset->ex_filter_function_, 1.0F);
            auto obj_res = index->KnnSearch(query, topk, search_param, filter);
            if (expected_success) {
                for (int j = 0; j < topk; ++j) {
                    REQUIRE(obj_res.value()->GetIds()[j] == res.value()->GetIds()[j]);
                }
            }
        }
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        if (index->CheckFeature(vsag::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER)) {
            auto threshold = res.value()->GetDistances()[topk - 1];
            auto range_result =
                index->RangeSearch(query, threshold, search_param, dataset->filter_function_);
            REQUIRE(range_result.value()->GetDim() >= topk);
        }
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestSearchAllocator(const TestIndex::IndexPtr& index,
                               const TestDatasetPtr& dataset,
                               const std::string& search_param,
                               float expected_recall,
                               bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    class ExampleAllocator : public vsag::Allocator {
    public:
        std::string
        Name() override {
            return "example-allocator";
        }

        void*
        Allocate(size_t size) override {
            auto addr = (void*)malloc(size);
            sizes_[addr] = size;
            return addr;
        }

        void
        Deallocate(void* p) override {
            if (sizes_.find(p) == sizes_.end())
                return;
            sizes_.erase(p);
            return free(p);
        }

        void*
        Reallocate(void* p, size_t size) override {
            auto addr = (void*)realloc(p, size);
            sizes_.erase(p);
            sizes_[addr] = size;
            return addr;
        }

    private:
        std::unordered_map<void*, size_t> sizes_;
    };

    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        ExampleAllocator allocator;
        vsag::SearchParam search_params(false, search_param, nullptr, &allocator);
        auto res = index->KnnSearch(query, topk, search_params);
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto dis = res.value()->GetDistances();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
        allocator.Deallocate((void*)result);
        allocator.Deallocate((void*)dis);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestCalcDistanceById(const IndexPtr& index,
                                const TestDatasetPtr& dataset,
                                float error,
                                bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        for (auto j = 0; j < gt_topK; ++j) {
            auto id = gts->GetIds()[i * gt_topK + j];
            auto dist = gts->GetDistances()[i * gt_topK + j];
            auto result = index->CalcDistanceById(query->GetFloat32Vectors(), id);
            if (not expected_success) {
                REQUIRE_FALSE(result.has_value());
                continue;
            }
            REQUIRE(result.has_value());
            REQUIRE(std::abs(dist - result.value()) < error);
        }
    }
}

void
TestIndex::TestBatchCalcDistanceById(const IndexPtr& index,
                                     const TestDatasetPtr& dataset,
                                     float error,
                                     bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_CAL_DISTANCE_BY_ID)) {
        return;
    }
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto result = index->CalDistanceById(
            query->GetFloat32Vectors(), gts->GetIds() + (i * gt_topK), gt_topK);
        if (not expected_success) {
            return;
        }
        for (auto j = 0; j < gt_topK; ++j) {
            REQUIRE(std::abs(gts->GetDistances()[i * gt_topK + j] -
                             result.value()->GetDistances()[j]) < error);
        }
    }
    SECTION("test non-existing id") {
        int64_t test_num = 10;
        std::vector<int64_t> no_exist_ids(test_num);
        for (int i = 0; i < test_num; ++i) {
            no_exist_ids[i] = -i - 1;
        }
        auto result =
            index->CalDistanceById(queries->GetFloat32Vectors(), no_exist_ids.data(), test_num);
        for (int i = 0; i < test_num; ++i) {
            fixtures::dist_t dist = result.value()->GetDistances()[i];
            REQUIRE(dist == -1);
        }
    }
}

void
TestIndex::TestGetMinAndMaxId(const IndexPtr& index,
                              const TestDatasetPtr& dataset,
                              bool expected_success) {
    auto base_count = dataset->base_->GetNumElements();
    auto dim = dataset->base_->GetDim();
    auto get_min_max_res = index->GetMinAndMaxId();
    if (not expected_success) {
        REQUIRE_FALSE(get_min_max_res.has_value());
        return;
    }
    REQUIRE(get_min_max_res.has_value() == (index->GetNumElements() > 0));
    int64_t res_max_id = INT64_MIN;
    int64_t res_min_id = INT64_MAX;
    for (uint64_t j = 0; j < base_count; ++j) {
        res_max_id =
            res_max_id > dataset->base_->GetIds()[j] ? res_max_id : dataset->base_->GetIds()[j];
        res_min_id =
            res_min_id < dataset->base_->GetIds()[j] ? res_min_id : dataset->base_->GetIds()[j];
    }
    get_min_max_res = index->GetMinAndMaxId();
    REQUIRE(get_min_max_res.has_value() == true);
    int64_t min_id = get_min_max_res.value().first;
    int64_t max_id = get_min_max_res.value().second;

    REQUIRE(min_id == res_min_id);
    REQUIRE(max_id == res_max_id);
}

void
TestIndex::TestSerializeFile(const IndexPtr& index_from,
                             const IndexPtr& index_to,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_FILE)) {
        return;
    }
    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    auto serialize_index = index_from->Serialize(outfile);
    REQUIRE(serialize_index.has_value() == expected_success);
    outfile.close();

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    auto deserialize_index = index_to->Deserialize(infile);
    REQUIRE(deserialize_index.has_value() == expected_success);
    infile.close();

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        for (auto j = 0; j < topk; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}
void
TestIndex::TestSearchWithDirtyVector(const TestIndex::IndexPtr& index,
                                     const TestDatasetPtr& dataset,
                                     const std::string& search_param,
                                     bool expected_success) {
    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    auto topk = gt_topK;
    int valid_query_count = static_cast<int64_t>(query_count * 0.9);
    for (auto i = 0; i < valid_query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
    }

    const auto& radius = dataset->range_radius_;
    for (auto i = 0; i < valid_query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        if (std::isnan(radius[i])) {
            continue;
        }
        auto res = index->RangeSearch(query, radius[i], search_param);
        REQUIRE(res.has_value() == expected_success);
    }

    for (auto i = valid_query_count; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value() == expected_success);
    }
}

void
TestIndex::TestSerializeBinarySet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_BINARY_SET)) {
        return;
    }
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);

    auto deserialize_index = index_to->Deserialize(serialize_binary.value());
    REQUIRE(deserialize_index.has_value() == expected_success);

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        for (auto j = 0; j < topk; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestSerializeReaderSet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  const std::string& index_name,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_READER_SET)) {
        return;
    }
    vsag::ReaderSet rs;
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);
    auto binary_set = serialize_binary.value();
    for (const auto& key : binary_set.GetKeys()) {
        rs.Set(key, std::make_shared<TestReader>(binary_set.Get(key)));
    }
    auto deserialize_index = index_to->Deserialize(rs);
    REQUIRE(deserialize_index.has_value() == expected_success);

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        for (auto j = 0; j < topk; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestConcurrentAdd(const TestIndex::IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
        return;
    }
    fixtures::logger::LoggerReplacer _;

    auto base_count = dataset->base_->GetNumElements();
    auto temp_count = static_cast<int64_t>(base_count * 0.8);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Paths(dataset->base_->GetPaths())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->SparseVectors(dataset->base_->GetSparseVectors())
        ->Owner(false);
    index->Build(temp_dataset);
    fixtures::ThreadPool pool(5);
    using RetType = tl::expected<std::vector<int64_t>, vsag::Error>;
    std::vector<std::future<RetType>> futures;

    auto func = [&](uint64_t i) -> RetType {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + i)
            ->NumElements(1)
            ->Paths(dataset->base_->GetPaths() + i)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dim)
            ->SparseVectors(dataset->base_->GetSparseVectors() + i)
            ->Owner(false);
        auto add_index = index->Add(data_one);
        return add_index;
    };

    for (uint64_t j = temp_count; j < base_count; ++j) {
        futures.emplace_back(pool.enqueue(func, j));
    }

    for (auto& res : futures) {
        auto val = res.get();
        REQUIRE(val.has_value() == expected_success);
    }
    REQUIRE(index->GetNumElements() == base_count);
}

void
TestIndex::TestConcurrentKnnSearch(const TestIndex::IndexPtr& index,
                                   const TestDatasetPtr& dataset,
                                   const std::string& search_param,
                                   float expected_recall,
                                   bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_SEARCH_CONCURRENT)) {
        return;
    }
    fixtures::logger::LoggerReplacer _;

    auto queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    std::vector<float> search_results(query_count, 0.0f);
    using RetType = std::pair<tl::expected<DatasetPtr, vsag::Error>, uint64_t>;
    std::vector<std::future<RetType>> futures;
    auto topk = gt_topK;
    fixtures::ThreadPool pool(5);

    auto func = [&](uint64_t i) -> RetType {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        return {res, i};
    };

    for (auto i = 0; i < query_count; ++i) {
        futures.emplace_back(pool.enqueue(func, i));
    }

    for (auto& res1 : futures) {
        auto [res, id] = res1.get();
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * id;
        auto val = Intersection(gt, gt_topK, result, topk);
        search_results[id] = static_cast<float>(val) / static_cast<float>(gt_topK);
    }

    auto cur_recall = std::accumulate(search_results.begin(), search_results.end(), 0.0f);
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestConcurrentDestruct(TestIndex::IndexPtr& index,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param) {
    std::vector<std::future<bool>> futures;
    fixtures::ThreadPool pool(32);
    std::shared_mutex index_mutex;

    auto func = [&](uint64_t i) -> bool {
        auto base = vsag::Dataset::Make();
        base->NumElements(1)
            ->Ids(dataset->base_->GetIds() + i)
            ->Dim(dataset->base_->GetDim())
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + i * dataset->base_->GetDim())
            ->Owner(false);

        if (i == (dataset->base_->GetNumElements() * 3) / 4) {
            std::unique_lock status_lock(index_mutex);
            std::dynamic_pointer_cast<vsag::HNSW>(index)->SetStatus(vsag::VSAGIndexStatus::ALIVE);
            index.reset();
            return true;
        }

        std::shared_lock status_lock(index_mutex);
        if (not index) {
            return false;
        }

        switch (random() % 22) {
            case 0:
                return index->Build(base).has_value();
            case 1:
                return index->Add(base).has_value();
            case 2:
                return index->Remove(*base->GetIds()).has_value();
            case 3:
                return index->UpdateId(*base->GetIds(), *base->GetIds() + 1).has_value();
            case 4:
                return index->UpdateVector(*base->GetIds(), base).has_value();
            case 5:
                return index->KnnSearch(base, 100, search_param).has_value();
            case 6:
                return index->RangeSearch(base, 100, search_param).has_value();
            case 7:
                return index->Feedback(base, 100, search_param).has_value();
            case 8:
                return index->Pretrain({*base->GetIds()}, 100, search_param).has_value();
            case 9:
                return index->CalcDistanceById(base->GetFloat32Vectors(), *base->GetIds())
                    .has_value();
            case 10:
                return index->CalDistanceById(base->GetFloat32Vectors(), base->GetIds(), 1)
                    .has_value();
            case 11:
                return index->GetMinAndMaxId().has_value();
            case 12:
                return index->Serialize().has_value();
            case 13: {
                std::ostringstream oss;
                std::ostream& out = oss;
                return index->Serialize(out).has_value();
            }
            case 14: {
                vsag::BinarySet bs;
                return index->Deserialize(bs).has_value();
            }
            case 15: {
                vsag::ReaderSet rs;
                return index->Deserialize(rs).has_value();
            }
            case 16: {
                std::istringstream iss;
                std::istream& in = iss;
                return index->Deserialize(in).has_value();
            }
            case 17:
                return index->Merge({}).has_value();
            case 18:
                return index->CheckFeature(vsag::IndexFeature::SUPPORT_BUILD);
            case 19:
                return index->CheckIdExist(*base->GetIds());
            case 20:
                return index->GetMemoryUsage() > 0;
            case 21:
                std::dynamic_pointer_cast<vsag::HNSW>(index)->SetStatus(
                    vsag::VSAGIndexStatus::DESTROYED);
                return true;
            default:
                std::dynamic_pointer_cast<vsag::HNSW>(index)->SetStatus(
                    vsag::VSAGIndexStatus::ALIVE);
                return true;
        }
    };

    for (uint64_t i = 0; i < dataset->base_->GetNumElements(); i++) {
        futures.emplace_back(pool.enqueue(func, i));
    }
}

void
TestIndex::TestContinueAddIgnoreRequire(const TestIndex::IndexPtr& index,
                                        const TestDatasetPtr& dataset,
                                        float build_ratio) {
    auto base_count = dataset->base_->GetNumElements();
    int64_t temp_count = static_cast<int64_t>(base_count * build_ratio);
    auto dim = dataset->base_->GetDim();
    auto temp_dataset = vsag::Dataset::Make();
    temp_dataset->Dim(dim)
        ->Ids(dataset->base_->GetIds())
        ->NumElements(temp_count)
        ->Paths(dataset->base_->GetPaths())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    index->Build(temp_dataset);
    for (uint64_t j = temp_count; j < base_count; ++j) {
        auto data_one = vsag::Dataset::Make();
        data_one->Dim(dim)
            ->Ids(dataset->base_->GetIds() + j)
            ->NumElements(1)
            ->Paths(dataset->base_->GetPaths() + j)
            ->Float32Vectors(dataset->base_->GetFloat32Vectors() + j * dim)
            ->Owner(false);
        auto add_index = index->Add(data_one);
    }
}
void
TestIndex::TestDuplicateAdd(const TestIndex::IndexPtr& index, const TestDatasetPtr& dataset) {
    auto double_dataset = vsag::Dataset::Make();
    uint64_t base_count = dataset->base_->GetNumElements();
    uint64_t double_count = base_count * 2;
    auto dim = dataset->base_->GetDim();
    auto new_data = std::shared_ptr<float[]>(new float[double_count * dim]);
    auto new_ids = std::shared_ptr<int64_t[]>(new int64_t[double_count]);
    memcpy(new_data.get(), dataset->base_->GetFloat32Vectors(), base_count * dim * sizeof(float));
    memcpy(new_data.get() + base_count * dim,
           dataset->base_->GetFloat32Vectors(),
           base_count * dim * sizeof(float));
    memcpy(new_ids.get(), dataset->base_->GetIds(), base_count * sizeof(int64_t));
    memcpy(new_ids.get() + base_count, dataset->base_->GetIds(), base_count * sizeof(int64_t));
    double_dataset->Dim(dim)
        ->NumElements(double_count)
        ->Ids(new_ids.get())
        ->Float32Vectors(new_data.get())
        ->Owner(false);

    auto check_func = [&](std::vector<int64_t>& failed_ids) -> void {
        REQUIRE(failed_ids.size() == base_count);
        std::sort(failed_ids.begin(), failed_ids.end());
        for (uint64_t i = 0; i < base_count; ++i) {
            REQUIRE(failed_ids[i] == dataset->base_->GetIds()[i]);
        }
    };

    // add once with duplicate;
    auto add_index = index->Build(double_dataset);
    REQUIRE(add_index.has_value());
    check_func(add_index.value());

    // add twice with duplicate;
    auto add_index_2 = index->Add(dataset->base_);
    REQUIRE(add_index_2.has_value());
    check_func(add_index_2.value());
}
void
TestIndex::TestEstimateMemory(const std::string& index_name,
                              const std::string& build_param,
                              const TestDatasetPtr& dataset) {
    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    {
        vsag::Resource resource(allocator.get(), nullptr);
        vsag::Engine engine(&resource);
        auto index1 = engine.CreateIndex(index_name, build_param).value();
        REQUIRE(index1->GetNumElements() == 0);
        auto index2 = vsag::Factory::CreateIndex(index_name, build_param).value();
        REQUIRE(index2->GetNumElements() == 0);
        fixtures::TempDir dir("index");
        auto path = dir.GenerateRandomFile();
        std::ofstream outf(path, std::ios::binary);
        if (index1->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
            auto data_size = dataset->base_->GetNumElements();
            auto estimate_memory = index1->EstimateMemory(data_size);
            auto build_index = index2->Build(dataset->base_);
            REQUIRE(build_index.has_value());
            index2->Serialize(outf);
            std::ifstream inf(path, std::ios::binary);
            index1->Deserialize(inf);
            auto real_memory = allocator->GetCurrentMemory();
            if (estimate_memory <= static_cast<uint64_t>(real_memory * 0.8) or
                estimate_memory >= static_cast<uint64_t>(real_memory * 1.2)) {
                WARN("estimate_memory failed");
            }

            REQUIRE(estimate_memory >= static_cast<uint64_t>(real_memory * 0.2));
            REQUIRE(estimate_memory <= static_cast<uint64_t>(real_memory * 3.2));
            inf.close();
        }
        outf.close();
    }
}

void
TestIndex::TestCheckIdExist(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_CHECK_ID_EXIST)) {
        return;
    }
    auto data_count = dataset->base_->GetNumElements();
    auto* ids = dataset->base_->GetIds();
    int N = 10;
    for (int i = 0; i < N; ++i) {
        auto good_id = ids[random() % data_count];
        REQUIRE(index->CheckIdExist(good_id) == expected_success);
    }
    std::unordered_set<int64_t> exist_ids(ids, ids + data_count);
    int bad_id = 97;
    while (N > 0) {
        for (; bad_id < data_count * N; ++bad_id) {
            if (exist_ids.count(bad_id) == 0) {
                break;
            }
        }
        REQUIRE(index->CheckIdExist(bad_id) == false);
        --N;
    }
}
TestIndex::IndexPtr
TestIndex::TestMergeIndex(const std::string& name,
                          const std::string& build_param,
                          const TestDatasetPtr& dataset,
                          int32_t split_num,
                          bool expect_success) {
    auto create_index_result = vsag::Factory::CreateIndex(name, build_param);
    REQUIRE(create_index_result.has_value() == expect_success);
    auto index = create_index_result.value();
    if (not index->CheckFeature(vsag::SUPPORT_MERGE_INDEX)) {
        return nullptr;
    }

    auto& raw_data = dataset->base_;
    std::vector<vsag::DatasetPtr> sub_datasets;
    int64_t all_data_num = raw_data->GetNumElements();
    int64_t data_dim = raw_data->GetDim();
    const float* vectors = raw_data->GetFloat32Vectors();  // shape = (all_data_num, data_dim)
    const int64_t* ids = raw_data->GetIds();               // shape = (all_data_num)

    int64_t subset_size = all_data_num / split_num;
    int64_t remaining = all_data_num % split_num;

    int64_t start_index = 0;

    for (int64_t i = 0; i < split_num; ++i) {
        int64_t current_subset_size = subset_size + (i < remaining ? 1 : 0);
        auto subset = vsag::Dataset::Make();
        subset->Float32Vectors(vectors + start_index * data_dim);
        subset->Ids(ids + start_index);
        subset->NumElements(current_subset_size);
        subset->Dim(data_dim);
        subset->Owner(false);
        sub_datasets.push_back(subset);
        start_index += current_subset_size;
    }

    std::vector<vsag::MergeUnit> merge_units;
    for (auto sub_dataset : sub_datasets) {
        auto new_index_result = vsag::Factory::CreateIndex(name, build_param);
        REQUIRE(new_index_result.has_value() == expect_success);
        auto new_index = new_index_result.value();
        new_index->Build(sub_dataset);
        vsag::IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
            return std::make_tuple(true, id);
        };
        merge_units.push_back({new_index, id_map});
    }
    auto merge_result = index->Merge(merge_units);
    REQUIRE(merge_result.has_value());
    return index;
}

TestIndex::IndexPtr
TestIndex::TestMergeIndexWithSameModel(const TestIndex::IndexPtr model,
                                       const TestDatasetPtr& dataset,
                                       int32_t split_num,
                                       bool expect_success) {
    if (not model->CheckFeature(vsag::SUPPORT_MERGE_INDEX)) {
        return nullptr;
    }
    if (not model->CheckFeature(vsag::SUPPORT_CLONE)) {
        return nullptr;
    }
    auto& raw_data = dataset->base_;
    std::vector<vsag::DatasetPtr> sub_datasets;
    int64_t all_data_num = raw_data->GetNumElements();
    int64_t data_dim = raw_data->GetDim();
    const float* vectors = raw_data->GetFloat32Vectors();  // shape = (all_data_num, data_dim)
    const int64_t* ids = raw_data->GetIds();               // shape = (all_data_num)
    int64_t subset_size = all_data_num / split_num;
    int64_t remaining = all_data_num % split_num;

    int64_t start_index = 0;

    for (int64_t i = 0; i < split_num; ++i) {
        int64_t current_subset_size = subset_size + (i < remaining ? 1 : 0);
        auto subset = vsag::Dataset::Make();
        subset->Float32Vectors(vectors + start_index * data_dim);
        subset->Ids(ids + start_index);
        subset->NumElements(current_subset_size);
        subset->Dim(data_dim);
        subset->Owner(false);
        sub_datasets.push_back(subset);
        start_index += current_subset_size;
    }
    std::vector<vsag::MergeUnit> merge_units;
    for (auto sub_dataset : sub_datasets) {
        auto new_index_result = model->Clone();
        REQUIRE(new_index_result.has_value() == expect_success);
        auto new_index = new_index_result.value();
        new_index->Add(sub_dataset);
        vsag::IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
            return std::make_tuple(true, id);
        };
        merge_units.push_back({new_index, id_map});
    }
    auto index_result = model->Clone();
    REQUIRE(index_result.has_value() == expect_success);
    auto index = index_result.value();
    auto merge_result = index->Merge(merge_units);
    REQUIRE(merge_result.has_value());
    return index;
}

void
TestIndex::TestGetExtraInfoById(const TestIndex::IndexPtr& index,
                                const TestDatasetPtr& dataset,
                                int64_t extra_info_size) {
    if (not index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
        return;
    }
    int64_t count = dataset->count_;
    std::vector<int64_t> ids(count);
    memcpy(ids.data(), dataset->base_->GetIds(), count * sizeof(int64_t));
    std::shuffle(ids.begin(), ids.end(), std::default_random_engine());
    std::vector<char> extra_infos(count * extra_info_size);
    auto result = index->GetExtraInfoByIds(ids.data(), count, extra_infos.data());
    REQUIRE(result.has_value());
    for (int64_t i = 0; i < count; ++i) {
        REQUIRE(
            memcmp(extra_infos.data() + i * extra_info_size,
                   dataset->base_->GetExtraInfos() + (ids[i] - dataset->ID_BIAS) * extra_info_size,
                   extra_info_size) == 0);
    }
}

void
TestIndex::TestKnnSearchExFilter(const IndexPtr& index,
                                 const TestDatasetPtr& dataset,
                                 const std::string& search_param,
                                 float expected_recall,
                                 bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)) {
        return;
    }
    auto queries = dataset->filter_query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ex_filter_ground_truth_;
    auto gt_topK = dataset->top_k;
    auto extra_info_size = dataset->base_->GetExtraInfoSize();
    float cur_recall = 0.0f;
    auto topk = gt_topK;
    auto f = std::make_shared<FilterObj>(dataset->filter_function_, nullptr, dataset->valid_ratio_);
    auto filter = std::make_shared<FilterObj>(
        dataset->filter_function_, dataset->ex_filter_function_, dataset->valid_ratio_);
    for (auto i = 0; i < query_count; ++i) {
        auto query_recall = 0.0f;
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Paths(queries->GetPaths() + i)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param, filter);
        if (not expected_success) {
            if (res.has_value()) {
                REQUIRE(res.value()->GetDim() == 0);
            }
        } else {
            REQUIRE(res.has_value() == expected_success);
        }
        if (!expected_success) {
            return;
        }
        REQUIRE(res.has_value() == true);
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        if (extra_info_size > 0) {
            const char* extra_infos = res.value()->GetExtraInfos();
            REQUIRE(f->CheckValid(extra_infos) == true);
            REQUIRE(extra_infos != nullptr);
            int64_t num = res.value()->GetNumElements();
            for (int j = 0; j < num; ++j) {
                REQUIRE((extra_infos + j * extra_info_size) != nullptr);
            }
        }
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
    }
    if (cur_recall <= expected_recall * query_count) {
        WARN(fmt::format("cur_result({}) <= expected_recall * query_count({})",
                         cur_recall,
                         expected_recall * query_count));
    }
    REQUIRE(cur_recall > expected_recall * query_count * RECALL_THRESHOLD);
}

void
TestIndex::TestClone(const TestIndex::IndexPtr& index,
                     const TestDatasetPtr& dataset,
                     const std::string& search_param) {
    if (not index->CheckFeature(vsag::SUPPORT_CLONE)) {
        return;
    }
    auto index_clone_result = index->Clone();
    REQUIRE(index_clone_result.has_value() == true);
    auto& index_clone = index_clone_result.value();

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res_from = index->KnnSearch(query, topk, search_param);
        auto res_to = index_clone->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        for (auto j = 0; j < topk; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestExportModel(const TestIndex::IndexPtr& index,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param) {
    if (not index->CheckFeature(vsag::SUPPORT_EXPORT_MODEL)) {
        return;
    }
    auto index_model_result = index->ExportModel();
    REQUIRE(index_model_result.has_value() == true);
    auto& index_model = index_model_result.value();
    tl::expected<std::vector<int64_t>, vsag::Error> add_index;
    if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
        add_index = index_model->Add(dataset->base_);
        REQUIRE(add_index.has_value());
    } else if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
        add_index = index_model->Build(dataset->base_);
        REQUIRE(add_index.has_value());
    } else {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float recall1 = 0.0F;
    float recall2 = 0.0F;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Paths(queries->GetPaths() + i)
            ->SparseVectors(queries->GetSparseVectors() + i)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res1 = index->KnnSearch(query, topk, search_param);
        REQUIRE(res1.has_value());
        auto result1 = res1.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result1, topk);
        recall1 += static_cast<float>(val) / static_cast<float>(gt_topK);

        auto res2 = index_model->KnnSearch(query, topk, search_param);
        REQUIRE(res2.has_value());
        auto result2 = res2.value()->GetIds();
        val = Intersection(gt, gt_topK, result2, topk);
        recall2 += static_cast<float>(val) / static_cast<float>(gt_topK);
    }

    REQUIRE(std::abs(recall1 - recall2) < 0.01F * query_count);
}

}  // namespace fixtures
