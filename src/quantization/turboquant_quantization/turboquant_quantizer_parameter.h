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

#include "quantization/quantizer_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(TurboQuantQuantizerParam, TurboQuantQuantizerParameter);

/**
 * Parameters for TurboQuant (arXiv:2504.19874) — MSE variant uses random rotation + scalar
 * quantization in the rotated domain; training is data-independent.
 */
class TurboQuantQuantizerParameter : public QuantizerParameter {
public:
    TurboQuantQuantizerParameter();

    ~TurboQuantQuantizerParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    /** Bits per rotated coordinate (K = 2^bits_per_dim_ levels). */
    uint32_t bits_per_dim_{4};
    /** "mse" is implemented; "prod" is reserved. */
    std::string variant_{"mse"};
    /** Seed to regenerate orthogonal Pi (and future extensions) deterministically. */
    uint64_t rotation_seed_{0xC0FFEEULL};
};

}  // namespace vsag
