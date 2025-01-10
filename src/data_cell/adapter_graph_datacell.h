
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
#include "algorithm/hnswlib/hnswalg.h"
#include "algorithm/hnswlib/space_l2.h"

namespace vsag {
class AdaptGraphDataCell {
public:
    AdaptGraphDataCell(std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw) : alg_hnsw_(alg_hnsw){};

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        uint32_t size = alg_hnsw_->getListCount((hnswlib::linklistsizeint*)data);
        neighbor_ids.resize(size);
        for (uint32_t i = 0; i < size; i++) {
            neighbor_ids[i] = *(data + i + 1);
        }
    }

    uint32_t
    GetNeighborSize(InnerIdType id) {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        return alg_hnsw_->getListCount((hnswlib::linklistsizeint*)data);
    }

    void
    Prefetch(InnerIdType id, InnerIdType neighbor_i) {
        int* data = (int*)alg_hnsw_->get_linklist0(id);
        _mm_prefetch(data + neighbor_i + 1, _MM_HINT_T0);
    }

    uint32_t
    MaximumDegree() {
        return alg_hnsw_->getMaxDegree();
    }

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw_;
};
}  // namespace vsag