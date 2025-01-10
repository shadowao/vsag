
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

namespace vsag {

template <typename GraphTmpl, typename VectorDataTmpl>
BasicSearcher<GraphTmpl, VectorDataTmpl>::BasicSearcher(std::shared_ptr<GraphTmpl> graph,
                                                        std::shared_ptr<VectorDataTmpl> vector,
                                                        const IndexCommonParam& common_param) {
    this->graph_ = graph;
    this->vector_data_cell_ = vector;
    this->allocator_ = common_param.allocator_.get();
    this->dim_ = common_param.dim_;
    pool_ = std::make_shared<hnswlib::VisitedListPool>(vector_data_cell_->TotalCount(), allocator_);
}

template <typename GraphTmpl, typename VectorDataTmpl>
void
BasicSearcher<GraphTmpl, VectorDataTmpl>::SetRuntimeParameters(
    const UnorderedMap<std::string, vsag::ParamValue>& new_params) {
    if (new_params.find(PREFETCH_NEIGHBOR_VISIT_NUM) != new_params.end()) {
        prefetch_neighbor_visit_num_ = std::get<int>(new_params.at(PREFETCH_NEIGHBOR_VISIT_NUM));
    }
    this->vector_data_cell_->SetRuntimeParameters(new_params);
}

template <typename GraphTmpl, typename VectorDataTmpl>
void
BasicSearcher<GraphTmpl, VectorDataTmpl>::Resize(uint64_t new_size) {
    pool_ = std::make_shared<hnswlib::VisitedListPool>(new_size, allocator_);
}

template <typename GraphTmpl, typename VectorDataTmpl>
uint32_t
BasicSearcher<GraphTmpl, VectorDataTmpl>::visit(hnswlib::VisitedListPtr vl,
                                                std::pair<float, uint64_t>& current_node_pair,
                                                Vector<InnerIdType>& to_be_visited_rid,
                                                Vector<InnerIdType>& to_be_visited_id) const {
    // to_be_visited_rid is used in redundant storage
    // to_be_visited_id  is used in flatten storage
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> neighbors(allocator_);

    graph_->GetNeighbors(current_node_pair.second, neighbors);

#ifdef USE_SSE
    for (uint32_t i = 0; i < prefetch_neighbor_visit_num_; i++) {
        _mm_prefetch(vl->mass + neighbors[i], _MM_HINT_T0);
    }
#endif

    for (uint32_t i = 0; i < neighbors.size(); i++) {
#ifdef USE_SSE
        if (i + prefetch_neighbor_visit_num_ < neighbors.size()) {
            _mm_prefetch(vl->mass + neighbors[i + prefetch_neighbor_visit_num_], _MM_HINT_T0);
        }
#endif
        if (vl->mass[neighbors[i]] != vl->curV) {
            to_be_visited_rid[count_no_visited] = i;
            to_be_visited_id[count_no_visited] = neighbors[i];
            count_no_visited++;
            vl->mass[neighbors[i]] = vl->curV;
        }
    }
    return count_no_visited;
}

template <typename GraphTmpl, typename VectorDataTmpl>
MaxHeap
BasicSearcher<GraphTmpl, VectorDataTmpl>::Search(const float* query,
                                                 InnerSearchParam& inner_search_param) const {
    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

    auto computer = vector_data_cell_->FactoryComputer(query);
    auto vl = pool_->getFreeVisitedList();

    float lower_bound;
    float dist;
    uint64_t candidate_id;
    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_rid(graph_->MaximumDegree(), allocator_);
    Vector<InnerIdType> to_be_visited_id(graph_->MaximumDegree(), allocator_);
    Vector<float> line_dists(graph_->MaximumDegree(), allocator_);

    InnerIdType ep_id = inner_search_param.ep_;
    vector_data_cell_->Query(&dist, computer, &ep_id, 1);
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);
    vl->mass[ep_id] = vl->curV;

    while (!candidate_set.empty()) {
        hops++;
        std::pair<float, uint64_t> current_node_pair = candidate_set.top();

        if ((-current_node_pair.first) > lower_bound &&
            (top_candidates.size() == inner_search_param.ef_)) {
            break;
        }
        candidate_set.pop();
        if (not candidate_set.empty()) {
            graph_->Prefetch(candidate_set.top().second, 0);
        }

        count_no_visited = visit(vl, current_node_pair, to_be_visited_rid, to_be_visited_id);

        dist_cmp += count_no_visited;

        // TODO(ZXY): implement mix storage query line
        vector_data_cell_->Query(
            line_dists.data(), computer, to_be_visited_id.data(), count_no_visited);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            candidate_id = to_be_visited_id[i];
            if (top_candidates.size() < inner_search_param.ef_ || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);

                top_candidates.emplace(dist, candidate_id);

                if (top_candidates.size() > inner_search_param.ef_)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lower_bound = top_candidates.top().first;
            }
        }
    }

    while (top_candidates.size() > inner_search_param.topk_) {
        top_candidates.pop();
    }

    pool_->releaseVisitedList(vl);
    return top_candidates;
}

template <typename GraphTmpl, typename VectorDataTmpl>
double
BasicSearcher<GraphTmpl, VectorDataTmpl>::MockRun() const {
    uint64_t sample_size = std::min(SAMPLE_SIZE, vector_data_cell_->TotalCount());

    InnerSearchParam search_param;
    search_param.ep_ = 0;
    search_param.ef_ = 80;  // experience value in benchmark
    search_param.is_id_allowed_ = nullptr;

    auto st = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < sample_size; ++i) {
        bool release = false;
        const auto* codes = vector_data_cell_->GetCodesById(i, release);
        Vector<float> raw_data(dim_, allocator_);
        vector_data_cell_->Decode(codes, raw_data.data());
        Search(raw_data.data(), search_param);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    double time_cost = std::chrono::duration<double>(ed - st).count();
    return time_cost;
}

template class BasicSearcher<
    AdaptGraphDataCell,
    FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>;

}  // namespace vsag