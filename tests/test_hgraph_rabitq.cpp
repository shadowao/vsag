
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
// RaBitQ + HGraph — same layout as tests/test_hgraph_turboquant.cpp
//
// JSON matches tests/test_hgraph.cpp reorder path + tools/eval/eval_hgraph.yaml:
// base_quantization_type rabitq, rabitq_bits_per_dim_*, rabitq_use_fht, use_reorder + precise.
// -----------------------------------------------------------------------------

#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "fixtures/fixtures.h"
#include "fixtures/test_dataset_pool.h"
#include "inner_string_params.h"
#include "test_index.h"
#include "vsag/dataset.h"
#include "vsag/options.h"

namespace {

std::string
MakeRabitqHGraphBuildJson(const fixtures::TempDir& temp_dir,
                          int dim,
                          int max_degree = 96,
                          int ef_construction = 500,
                          int rabitq_bits_base = 1,
                          int rabitq_bits_query = 4) {
    int pq_dim = dim;
    if (pq_dim % 2 == 0) {
        pq_dim /= 2;
    }
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
            "rabitq_use_fht": true,
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

}  // namespace

TEST_CASE("HGraph RaBitQ usage (documented standalone)", "[ft][hgraph][rabitq][doc]") {
    fixtures::TempDir temp_dir{"hgraph_rabitq_doc"};
    fixtures::TestDatasetPool pool{};
    auto origin_limit = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 2);

    const int dim = fixtures::RABITQ_MIN_RACALL_DIM;
    const std::string metric = "ip";
    constexpr float recall = 0.3F;
    constexpr uint64_t base_count = 600;

    const std::string build_json = MakeRabitqHGraphBuildJson(temp_dir, dim);
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

// Same scenario as test_hgraph_turboquant scale case: random base, build, KnnSearch top-10.
TEST_CASE("HGraph RaBitQ 10k x 768 build and knn top10", "[ft][hgraph][rabitq][scale][slow]") {
    fixtures::TempDir temp_dir{"hgraph_rabitq_scale"};
    fixtures::TestDatasetPool pool{};
    auto origin_limit = vsag::Options::Instance().block_size_limit();
    vsag::Options::Instance().set_block_size_limit(1024 * 1024 * 2);

    constexpr int kDim = 768;
    constexpr uint64_t kBaseCount = 10000;
    constexpr int kTopk = 10;
    const std::string metric = "ip";

    const std::string build_json = MakeRabitqHGraphBuildJson(temp_dir, kDim, 32, 200, 1, 4);
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
