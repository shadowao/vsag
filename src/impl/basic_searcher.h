
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

#pragma once
#include <nlohmann/json.hpp>
#include <random>
#include <shared_mutex>

#include "../utils.h"
#include "ThreadPool.h"
#include "algorithm/hnswlib/algorithm_interface.h"
#include "algorithm/hnswlib/visited_list_pool.h"
#include "common.h"
#include "data_cell/adapter_graph_datacell.h"
#include "data_cell/flatten_datacell.h"
#include "data_cell/flatten_interface.h"
#include "index/index_common_param.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "runtime_parameter.h"

namespace vsag {

static const InnerIdType SAMPLE_SIZE = 10000;
static const uint32_t CENTROID_EF = 500;
static const uint32_t PREFETCH_DEGREE_DIVIDE = 3;
static const uint32_t PREFETCH_MAXIMAL_DEGREE = 1;
static const uint32_t PREFETCH_MAXIMAL_LINES = 1;

class InnerSearchParam {
public:
    int topk_{0};
    float radius_{0.0f};
    InnerIdType ep_{0};
    uint64_t ef_{10};
    BaseFilterFunctor* is_id_allowed_{nullptr};
};

struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, uint64_t> const& a,
               std::pair<float, uint64_t> const& b) const noexcept {
        return a.first < b.first;
    }
};

using MaxHeap = std::priority_queue<std::pair<float, InnerIdType>,
                                    Vector<std::pair<float, InnerIdType>>,
                                    CompareByFirst>;

template <typename GraphTmpl, typename VectorDataTmpl>
class BasicSearcher {
public:
    BasicSearcher(std::shared_ptr<GraphTmpl> graph,
                  std::shared_ptr<VectorDataTmpl> vector,
                  const IndexCommonParam& common_param);

    virtual MaxHeap
    Search(const float* query, InnerSearchParam& inner_search_param) const;

    virtual double
    MockRun() const;

    virtual void
    Resize(uint64_t new_size);

    virtual void
    SetRuntimeParameters(const UnorderedMap<std::string, ParamValue>& new_params);

private:
    uint32_t
    visit(hnswlib::VisitedListPtr vl,
          std::pair<float, uint64_t>& current_node_pair,
          Vector<InnerIdType>& to_be_visited_rid,
          Vector<InnerIdType>& to_be_visited_id) const;

private:
    Allocator* allocator_{nullptr};

    std::shared_ptr<GraphTmpl> graph_;

    std::shared_ptr<VectorDataTmpl> vector_data_cell_;

    std::shared_ptr<hnswlib::VisitedListPool> pool_{nullptr};

    int64_t dim_{0};

    uint32_t prefetch_neighbor_visit_num_{1};
};

}  // namespace vsag