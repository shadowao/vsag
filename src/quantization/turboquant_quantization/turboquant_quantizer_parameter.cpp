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

#include "turboquant_quantizer_parameter.h"

#include "fmt/format.h"

#include "impl/logger/logger.h"
#include "inner_string_params.h"

namespace vsag {

TurboQuantQuantizerParameter::TurboQuantQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_TURBOQUANT) {
}

void
TurboQuantQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(TURBOQUANT_BITS_PER_DIM_KEY)) {
        this->bits_per_dim_ = static_cast<uint32_t>(json[TURBOQUANT_BITS_PER_DIM_KEY].GetInt());
    }
    if (json.Contains(TURBOQUANT_VARIANT_KEY)) {
        this->variant_ = json[TURBOQUANT_VARIANT_KEY].GetString();
    }
    if (json.Contains(TURBOQUANT_ROTATION_SEED_KEY)) {
        this->rotation_seed_ = static_cast<uint64_t>(json[TURBOQUANT_ROTATION_SEED_KEY].GetInt());
    }

    if (this->bits_per_dim_ < 1 || this->bits_per_dim_ > 8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("turboquant_bits_per_dim must be in [1, 8], got {}",
                                        this->bits_per_dim_));
    }
    if (this->variant_ != "mse") {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("only turboquant variant \"mse\" is supported, got {}",
                                        this->variant_));
    }
}

JsonType
TurboQuantQuantizerParameter::ToJson() const {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    json[TURBOQUANT_BITS_PER_DIM_KEY].SetInt(static_cast<int>(this->bits_per_dim_));
    json[TURBOQUANT_VARIANT_KEY].SetString(this->variant_);
    json[TURBOQUANT_ROTATION_SEED_KEY].SetInt(this->rotation_seed_);
    return json;
}

bool
TurboQuantQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto tq = std::dynamic_pointer_cast<TurboQuantQuantizerParameter>(other);
    if (not tq) {
        logger::error(
            "TurboQuantQuantizerParameter::CheckCompatibility: other is not "
            "TurboQuantQuantizerParameter");
        return false;
    }
    if (this->bits_per_dim_ != tq->bits_per_dim_) {
        logger::error(
            "TurboQuantQuantizerParameter::CheckCompatibility: bits_per_dim mismatch: {} vs {}",
            this->bits_per_dim_,
            tq->bits_per_dim_);
        return false;
    }
    if (this->variant_ != tq->variant_) {
        logger::error(
            "TurboQuantQuantizerParameter::CheckCompatibility: variant mismatch: {} vs {}",
            this->variant_,
            tq->variant_);
        return false;
    }
    if (this->rotation_seed_ != tq->rotation_seed_) {
        logger::error(
            "TurboQuantQuantizerParameter::CheckCompatibility: rotation_seed mismatch: {} vs {}",
            this->rotation_seed_,
            tq->rotation_seed_);
        return false;
    }
    return true;
}

}  // namespace vsag
