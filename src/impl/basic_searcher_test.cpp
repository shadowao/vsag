
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

#include "basic_searcher.h"

#include "algorithm/hnswlib/hnswalg.h"
#include "algorithm/hnswlib/space_l2.h"
#include "basic_optimizer.h"
#include "catch2/catch_template_test_macros.hpp"
#include "data_cell/adapter_graph_datacell.h"
#include "data_cell/flatten_datacell.h"
#include "default_allocator.h"
#include "fixtures.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "safe_allocator.h"

using namespace vsag;

TEST_CASE("search with alg_hnsw and optimizer", "[ut][basic_searcher]") {
    // data attr
    uint32_t base_size = 1000;
    uint32_t query_size = 100;
    uint64_t dim = 960;

    // build and search attr
    uint32_t M = 32;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t DEFAULT_MAX_ELEMENT = 1;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // alg_hnsw
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   DEFAULT_MAX_ELEMENT,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();
    for (int64_t i = 0; i < base_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    // graph data cell
    auto graph_data_cell = std::make_shared<AdaptGraphDataCell>(alg_hnsw);
    using GraphTmpl = std::remove_pointer_t<decltype(graph_data_cell.get())>;

    // vector data cell
    auto fp32_param = JsonType::parse("{}");
    auto io_param = JsonType::parse("{}");
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;

    auto vector_data_cell = std::make_shared<
        FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
        fp32_param, io_param, common);
    vector_data_cell->SetQuantizer(
        std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim, allocator.get()));
    vector_data_cell->SetIO(std::make_unique<MemoryIO>(allocator.get()));

    vector_data_cell->Train(base_vectors.data(), base_size);
    vector_data_cell->BatchInsertVector(base_vectors.data(), base_size, ids.data());
    using VectorDataTmpl = std::remove_pointer_t<decltype(vector_data_cell.get())>;

    // init searcher and optimizer
    auto searcher = std::make_shared<BasicSearcher<GraphTmpl, VectorDataTmpl>>(
        graph_data_cell, vector_data_cell, common);
    auto optimizer =
        std::make_shared<Optimizer<BasicSearcher<GraphTmpl, VectorDataTmpl>>>(common, 1);
    optimizer->RegisterParameter(std::make_shared<IntRuntimeParameter>(PREFETCH_CACHE_LINE, 1, 10));
    optimizer->RegisterParameter(
        std::make_shared<IntRuntimeParameter>(PREFETCH_NEIGHBOR_CODE_NUM, 1, 10));
    optimizer->RegisterParameter(
        std::make_shared<IntRuntimeParameter>(PREFETCH_NEIGHBOR_VISIT_NUM, 1, 10));
    optimizer->Optimize(searcher);

    // search
    InnerSearchParam search_param;
    search_param.ep_ = fixed_entry_point_id;
    search_param.ef_ = ef_search;
    search_param.topk_ = k;
    search_param.is_id_allowed_ = nullptr;
    for (int i = 0; i < query_size; i++) {
        std::unordered_set<InnerIdType> valid_set, set;
        auto result = searcher->Search(base_vectors.data() + i * dim, search_param);
        auto valid_result = alg_hnsw->searchBaseLayerST<false, false>(
            fixed_entry_point_id, base_vectors.data() + i * dim, ef_search, nullptr);
        REQUIRE(result.size() == valid_result.size());

        for (int j = 0; j < k - 1; j++) {
            valid_set.insert(valid_result.top().second);
            set.insert(result.top().second);
            result.pop();
            valid_result.pop();
        }
        for (auto id : set) {
            REQUIRE(valid_set.find(id) != valid_set.end());
        }
        for (auto id : valid_set) {
            REQUIRE(set.find(id) != set.end());
        }
        REQUIRE(result.top().second == valid_result.top().second);
        REQUIRE(result.top().second == ids[i]);
    }
}