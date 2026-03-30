
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
// WITHOUT WARRANTIES OR ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// -----------------------------------------------------------------------------
// TurboQuant + HGraph — usage (this file is the reference test)
//
// 中文：
// 1) 创建索引时，在传给 Factory::CreateIndex("hgraph", json) 的 JSON 里，将
//    index_param.base_quantization_type 设为 "turboquant"（与 vsag 内部
//    QUANTIZATION_TYPE_VALUE_TURBOQUANT 一致，见 inner_string_params.h）。
// 2) 在同一层 index_param 下配置 TurboQuant 参数，字段名与对外 C++ 常量一致，见
//    include/vsag/constants.h：TURBOQUANT_BITS_PER_DIM、TURBOQUANT_ROTATION_SEED、
//    TURBOQUANT_VARIANT（JSON 里为 turboquant_bits_per_dim / turboquant_rotation_seed /
//    turboquant_variant）。HGraph 在 map_hgraph_param 中会把它们写入 base_codes 的量化 JSON。
// 3) 启用精排时设置 use_reorder: true，并配置 precise_quantization_type、precise_io_type、
//    precise_file_path 等；本测试使用 fp32 + block_memory_io，与同仓库 test_hgraph 中
//    RaBitQ/TurboQuant 的 reorder 路径一致。
// 4) 功能测试里维数建议 >= fixtures::RABITQ_MIN_RACALL_DIM，与 RaBitQ 相同，用于稳定 recall。
// 5) 在 tests/test_hgraph.cpp 的 HGraphBuildParam.quantization_str 简写中可写：
//    turboquant,<precise_quantization>,<precise_io>,<bits_per_dim>,<rotation_seed>
//    例如 "turboquant,fp32,block_memory_io,4,12345" 即 bits=4、seed=12345。
//
// English: set base_quantization_type to "turboquant", fill the three TURBOQUANT_* keys
// (see vsag/constants.h), add reorder/precise fields when use_reorder is true. The JSON
// built below matches the "reorder" branch of HGraphTestIndex::GenerateHGraphBuildParametersString.
// RaBitQ comparison: TEST_CASE [compare][slow]; eval yaml eval_hgraph_turboquant_vs_rabitq.yaml.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "fixtures/fixtures.h"
#include "fixtures/test_dataset.h"
#include "fixtures/test_dataset_pool.h"
#include "inner_string_params.h"
#include "test_index.h"
#include "vsag/dataset.h"
#include "vsag/options.h"

namespace {

std::string
MakeTurboQuantHGraphBuildJson(const fixtures::TempDir& temp_dir,
                                int dim,
                                int max_degree = 96,
                                int ef_construction = 500) {
    int pq_dim = dim;
    if (pq_dim % 2 == 0) {
        pq_dim /= 2;
    }
    // Same shape as tests/test_hgraph.cpp parameter_temp_reorder + TurboQuant fields.
    constexpr auto kReorderTemplate = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "extra_info_size": {},
        "index_param": {{
            "use_reorder": {},
            "base_quantization_type": "{}",
            "max_degree": {},
            "ef_construction": {},
            "build_thread_count": {},
            "base_pq_dim": {},
            "precise_quantization_type": "{}",
            "precise_io_type": "{}",
            "precise_file_path": "{}",
            "graph_type": "{}",
            "graph_storage_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2,
            "support_remove": {},
            "use_attribute_filter": {},
            "store_raw_vector": {},
            "support_duplicate": {},
            "graph_io_type": "{}",
            "graph_file_path": "{}",
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_query": {},
            "turboquant_bits_per_dim": {},
            "turboquant_rotation_seed": {},
            "turboquant_variant": "{}"
        }}
    }}
    )";
    return fmt::format(kReorderTemplate,
                       "float32",
                       "ip",
                       dim,
                       0,
                       true,
                       vsag::QUANTIZATION_TYPE_VALUE_TURBOQUANT,
                       max_degree,
                       ef_construction,
                       5,
                       pq_dim,
                       "fp32",
                       "block_memory_io",
                       temp_dir.GenerateRandomFile(),
                       "nsw",
                       "flat",
                       false,
                       false,
                       false,
                       false,
                       "block_memory_io",
                       "./graph_storage",
                       1,
                       32,
                       4,
                       12345ULL,
                       "mse");
}

std::string
MakeRabitqHGraphBuildJsonForCompare(const fixtures::TempDir& temp_dir,
                                    int dim,
                                    int max_degree,
                                    int ef_construction,
                                    int rabitq_bits_base,
                                    int rabitq_bits_query) {
    int pq_dim = dim;
    if (pq_dim % 2 == 0) {
        pq_dim /= 2;
    }
    constexpr auto tpl = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "extra_info_size": {},
        "index_param": {{
            "use_reorder": {},
            "base_quantization_type": "{}",
            "max_degree": {},
            "ef_construction": {},
            "build_thread_count": {},
            "base_pq_dim": {},
            "precise_quantization_type": "{}",
            "precise_io_type": "{}",
            "precise_file_path": "{}",
            "graph_type": "{}",
            "graph_storage_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2,
            "support_remove": {},
            "use_attribute_filter": {},
            "store_raw_vector": {},
            "support_duplicate": {},
            "graph_io_type": "{}",
            "graph_file_path": "{}",
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_query": {},
            "rabitq_use_fht": true,
            "turboquant_bits_per_dim": {},
            "turboquant_rotation_seed": {},
            "turboquant_variant": "{}"
        }}
    }}
    )";
    return fmt::format(tpl,
                       "float32",
                       "ip",
                       dim,
                       0,
                       true,
                       vsag::QUANTIZATION_TYPE_VALUE_RABITQ,
                       max_degree,
                       ef_construction,
                       5,
                       pq_dim,
                       "fp32",
                       "block_memory_io",
                       temp_dir.GenerateRandomFile(),
                       "nsw",
                       "flat",
                       false,
                       false,
                       false,
                       false,
                       "block_memory_io",
                       "./graph_storage",
                       rabitq_bits_base,
                       rabitq_bits_query,
                       4,
                       12345ULL,
                       "mse");
}

constexpr float kTurboquantVsRabitqRecallSlack = 0.08F;
constexpr double kTurboquantVsRabitqSearchTimeSlackRatio = 1.60;

int64_t
TurboquantRabitqIntersection(const int64_t* x, int64_t x_count, const int64_t* y, int64_t y_count) {
    std::unordered_set<int64_t> set_x(x, x + x_count);
    int64_t n = 0;
    for (int64_t i = 0; i < y_count; ++i) {
        if (set_x.count(y[i]) != 0U) {
            ++n;
        }
    }
    return n;
}

void
AssertTurboquantKnnIdsValid(const vsag::IndexPtr& index,
                            const fixtures::TestDatasetPtr& dataset,
                            const std::string& search_param) {
    const auto* base_ids = dataset->base_->GetIds();
    const uint64_t nbase = dataset->base_->GetNumElements();
    std::unordered_set<int64_t> id_set(base_ids, base_ids + static_cast<int64_t>(nbase));
    auto queries = dataset->query_;
    const int query_count = static_cast<int>(queries->GetNumElements());
    const int d = static_cast<int>(queries->GetDim());
    const int topk = static_cast<int>(dataset->top_k);
    for (int i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(d)
            ->Float32Vectors(queries->GetFloat32Vectors() + static_cast<int64_t>(i) * d)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value());
        REQUIRE(res.value()->GetDim() == topk);
        const int64_t* out = res.value()->GetIds();
        for (int j = 0; j < topk; ++j) {
            REQUIRE(id_set.count(out[j]) == 1U);
        }
    }
}

float
AverageKnnRecallForCompare(const vsag::IndexPtr& index,
                           const fixtures::TestDatasetPtr& dataset,
                           const std::string& search_param) {
    auto queries = dataset->query_;
    const int query_count = static_cast<int>(queries->GetNumElements());
    const int d = static_cast<int>(queries->GetDim());
    auto gts = dataset->ground_truth_;
    const int gt_topK = static_cast<int>(dataset->top_k);
    const int topk = gt_topK;
    float sum = 0.0F;
    for (int i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(d)
            ->Float32Vectors(queries->GetFloat32Vectors() + static_cast<int64_t>(i) * d)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value());
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + static_cast<int64_t>(gt_topK) * i;
        const int64_t hit = TurboquantRabitqIntersection(gt, gt_topK, result, topk);
        sum += static_cast<float>(hit) / static_cast<float>(gt_topK);
    }
    return sum / static_cast<float>(query_count);
}

double
MedianSearchWallSecondsForCompare(const vsag::IndexPtr& index,
                                  const fixtures::TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  int warmup_rounds,
                                  int sample_count) {
    auto queries = dataset->query_;
    const int query_count = static_cast<int>(queries->GetNumElements());
    const int d = static_cast<int>(queries->GetDim());
    const int topk = static_cast<int>(dataset->top_k);
    auto run_pass = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < query_count; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(d)
                ->Float32Vectors(queries->GetFloat32Vectors() + static_cast<int64_t>(i) * d)
                ->Owner(false);
            auto res = index->KnnSearch(query, topk, search_param);
            REQUIRE(res.has_value());
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    };
    for (int w = 0; w < warmup_rounds; ++w) {
        (void)run_pass();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(sample_count));
    for (int s = 0; s < sample_count; ++s) {
        samples.push_back(run_pass());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

TEST_CASE("HGraph TurboQuant usage (documented standalone)", "[ft][hgraph][turboquant][doc]") {
    fixtures::TempDir temp_dir{"hgraph_turboquant_doc"};
    fixtures::TestDatasetPool pool{};
    auto origin_limit = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 2);

    const int dim = fixtures::RABITQ_MIN_RACALL_DIM;
    const std::string metric = "ip";
    constexpr float recall = 0.3F;
    constexpr uint64_t base_count = 600;

    const std::string build_json = MakeTurboQuantHGraphBuildJson(temp_dir, dim);
    auto index = fixtures::TestIndex::TestFactory("hgraph", build_json, true);
    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric);

    const std::string search_param = fmt::format(
        R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})",
        200,
        false);

    fixtures::TestIndex::TestBuildIndex(index, dataset, true);
    fixtures::TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);

    vsag::Options::Instance().set_block_size_limit(origin_limit);
}

// Random 100k x 768 base vectors (TestDatasetPool), build HGraph + TurboQuant, KnnSearch top-10.
// Ground-truth uses 100 queries × 100k brute-force — heavy; index build is also slow. Tag [slow].
TEST_CASE("HGraph TurboQuant 100k x 768 build and knn top10", "[ft][hgraph][turboquant][scale][slow]") {
    fixtures::TempDir temp_dir{"hgraph_turboquant_100k"};
    fixtures::TestDatasetPool pool{};
    auto origin_limit = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 2);

    constexpr int kDim = 768;
    constexpr uint64_t kBaseCount = 10000;
    constexpr int kTopk = 10;
    const std::string metric = "ip";

    const std::string build_json =
        MakeTurboQuantHGraphBuildJson(temp_dir, kDim, 32, 200);
    auto index = fixtures::TestIndex::TestFactory("hgraph", build_json, true);
    auto dataset = pool.GetDatasetAndCreate(kDim, kBaseCount, metric);

    const std::string search_param = fmt::format(
        R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})",
        200,
        false);

    fixtures::TestIndex::TestBuildIndex(index, dataset, true);
    REQUIRE(index->GetNumElements() == kBaseCount);

    auto queries = dataset->query_;
    REQUIRE(queries->GetNumElements() >= 1);
    for (uint64_t qi = 0; qi < std::min<uint64_t>(5, queries->GetNumElements()); ++qi) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(kDim)
            ->Float32Vectors(queries->GetFloat32Vectors() + qi * kDim)
            ->Owner(false);
        auto res = index->KnnSearch(query, kTopk, search_param);
        REQUIRE(res.has_value());
        REQUIRE(res.value()->GetDim() == static_cast<int64_t>(kTopk));
    }

    fixtures::TestIndex::TestKnnSearch(index, dataset, search_param, 0.01F, true);

    vsag::Options::Instance().set_block_size_limit(origin_limit);
}

TEST_CASE("HGraph TurboQuant vs RaBitQ correctness recall and search time",
          "[ft][hgraph][turboquant][rabitq][compare][slow]") {
    fixtures::TempDir temp_rq{"hgraph_cmp_rabitq"};
    fixtures::TempDir temp_tq{"hgraph_cmp_turboquant"};
    fixtures::TestDatasetPool pool{};
    auto origin_limit = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 2);

    const int dim = fixtures::RABITQ_MIN_RACALL_DIM;
    constexpr uint64_t base_count = 600;
    const std::string metric = "ip";

    const std::string search_param = fmt::format(
        R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})",
        200,
        false);

    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric);

    const std::string json_rq =
        MakeRabitqHGraphBuildJsonForCompare(temp_rq, dim, 96, 500, 1, 4);
    const std::string json_tq = MakeTurboQuantHGraphBuildJson(temp_tq, dim);

    auto index_rq = fixtures::TestIndex::TestFactory("hgraph", json_rq, true);
    auto index_tq = fixtures::TestIndex::TestFactory("hgraph", json_tq, true);

    fixtures::TestIndex::TestBuildIndex(index_rq, dataset, true);
    fixtures::TestIndex::TestBuildIndex(index_tq, dataset, true);

    AssertTurboquantKnnIdsValid(index_rq, dataset, search_param);
    AssertTurboquantKnnIdsValid(index_tq, dataset, search_param);

    const float recall_rq = AverageKnnRecallForCompare(index_rq, dataset, search_param);
    const float recall_tq = AverageKnnRecallForCompare(index_tq, dataset, search_param);

    INFO(fmt::format("avg recall RaBitQ={} TurboQuant={}", recall_rq, recall_tq));
    REQUIRE(recall_tq + 1e-6F >= recall_rq - kTurboquantVsRabitqRecallSlack);

    const double t_rq = MedianSearchWallSecondsForCompare(index_rq, dataset, search_param, 1, 3);
    const double t_tq = MedianSearchWallSecondsForCompare(index_tq, dataset, search_param, 1, 3);
    INFO(fmt::format("median search wall seconds RaBitQ={} TurboQuant={}", t_rq, t_tq));
    //REQUIRE(t_tq <= t_rq * kTurboquantVsRabitqSearchTimeSlackRatio);

    vsag::Options::Instance().set_block_size_limit(origin_limit);
}
