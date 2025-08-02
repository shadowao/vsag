
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

#include "rabitq_quantizer_parameter.h"

#include "inner_string_params.h"

namespace vsag {

RaBitQuantizerParameter::RaBitQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_RABITQ) {
}

void
RaBitQuantizerParameter::FromJson(const JsonType& json) {
    if (json.contains(PCA_DIM)) {
        this->pca_dim_ = json[PCA_DIM];
    }
    if (json.contains(RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY)) {
        this->num_bits_per_dim_query_ = json[RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY];
    }
    if (json.contains(RABITQ_QUANTIZATION_QUERY_ALIGNED_BITS)) {
        this->query_aligned_bits_ = json[RABITQ_QUANTIZATION_QUERY_ALIGNED_BITS];
    }
    if (json.contains(USE_FHT)) {
        this->use_fht_ = json[USE_FHT];
    }
}

JsonType
RaBitQuantizerParameter::ToJson() {
    JsonType json;
    json[QUANTIZATION_TYPE_KEY] = QUANTIZATION_TYPE_VALUE_RABITQ;
    json[PCA_DIM] = this->pca_dim_;
    json[RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY] = this->num_bits_per_dim_query_;
    json[RABITQ_QUANTIZATION_QUERY_ALIGNED_BITS] = this->query_aligned_bits_;
    json[USE_FHT] = this->use_fht_;
    return json;
}
}  // namespace vsag
