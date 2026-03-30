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

#include <memory>

#include "impl/transform/fht_kac_rotate_transformer.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "turboquant_quantizer_parameter.h"

namespace vsag {

/**
 * TurboQuant MSE path: FHT–Kac rotation (seeded FhtKacRotator), per-dimension scalar quantization
 * in y = R x after L2 normalization (COSINE / IP / L2 with stored norm). Training is data-free.
 * Reference: arXiv:2504.19874.
 *
 * For IP/COSINE, ComputeDistImpl uses <R q, y> with y the per-dim quantized vector in rotated
 * space (same bits as encode), avoiding inverse FHT per neighbor — equivalent to <q, R^{-1} y>
 * when R preserves inner products (same idea as RaBitQ’s query–base fast path).
 *
 * For L2SQR, query buffer layout is (dim + 1 + dim) floats: unit direction, ||q||, then z = R q_unit;
 * the leading dim floats are reused as scratch for rotated-space y in ComputeDistImpl (not the same
 * layout as IP/COSINE’s 2×dim buffer).
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class TurboQuantQuantizer : public Quantizer<TurboQuantQuantizer<metric>> {
public:
    explicit TurboQuantQuantizer(const TurboQuantQuantizerParamPtr& param,
                                 const IndexCommonParam& common_param);

    explicit TurboQuantQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data) const;

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) const;

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const DataType* query,
                     Computer<TurboQuantQuantizer<metric>>& computer) const;

    void
    ComputeDistImpl(Computer<TurboQuantQuantizer<metric>>& computer,
                     const uint8_t* codes,
                     float* dists) const;

    void
    ScanBatchDistImpl(Computer<TurboQuantQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<TurboQuantQuantizer<metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_TURBOQUANT;
    }

private:
    void
    SetupSizesAndMetric(const IndexCommonParam& common_param);

    void
    InitializeState();

    bool
    StoresNorm() const;

    void
    ApplyPi(const float* x, float* y) const;

    void
    ApplyPiTransposed(const float* y, float* x) const;

    void
    DecodeUnitVector(const uint8_t* codes, float* out) const;

    /** Reconstruct per-dim quantized values in rotated space (no inverse FHT). */
    void
    DecodeToRotatedSpace(const uint8_t* codes, float* y_rotated) const;

    void
    UpdateQueryCodeSize();

    /** Keeps Allocator alive; base class only stores Allocator*. */
    std::shared_ptr<Allocator> allocator_holder_{};

    uint32_t bits_per_dim_{4};
    std::string variant_{"mse"};
    uint64_t rotation_seed_{0};
    uint64_t packed_bytes_{0};
    float sigma_{0};
    std::unique_ptr<FhtKacRotator> rotator_{};
    std::vector<float> centroids_;
};

}  // namespace vsag
