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

#include "turboquant_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "turboquant_codebook.h"

namespace vsag {

namespace {

constexpr float kNormEps = 1e-12F;

static inline uint64_t
PackedBitsBytes(uint64_t dim, uint32_t bits_per_dim) {
    return (dim * static_cast<uint64_t>(bits_per_dim) + 7ULL) / 8ULL;
}

static inline void
WriteBits(uint8_t* codes, uint32_t b, uint64_t dim_index, uint32_t value) {
    const uint64_t bit_pos = dim_index * static_cast<uint64_t>(b);
    const uint64_t byte_idx = bit_pos / 8ULL;
    const uint64_t offset = bit_pos % 8ULL;
    const uint32_t mask = (1U << b) - 1U;
    value &= mask;
    codes[byte_idx] |= static_cast<uint8_t>(value << offset);
    if (offset + static_cast<uint64_t>(b) > 8ULL) {
        codes[byte_idx + 1] |= static_cast<uint8_t>(value >> (8ULL - offset));
    }
}

static inline uint32_t
ReadBits(const uint8_t* codes, uint32_t b, uint64_t dim_index) {
    const uint64_t bit_pos = dim_index * static_cast<uint64_t>(b);
    const uint64_t byte_idx = bit_pos / 8ULL;
    const uint64_t offset = bit_pos % 8ULL;
    const uint32_t mask = (1U << b) - 1U;
    uint32_t v = static_cast<uint32_t>(codes[byte_idx] >> offset);
    if (offset + static_cast<uint64_t>(b) > 8ULL) {
        v |= static_cast<uint32_t>(codes[byte_idx + 1]) << (8U - static_cast<uint32_t>(offset));
    }
    return v & mask;
}

static inline uint32_t
NearestCentroidIndex(float y, const std::vector<float>& centroids) {
    const uint32_t k = static_cast<uint32_t>(centroids.size());
    uint32_t best = 0;
    float best_d = std::abs(y - centroids[0]);
    for (uint32_t j = 1; j < k; ++j) {
        float d = std::abs(y - centroids[j]);
        if (d < best_d) {
            best_d = d;
            best = j;
        }
    }
    return best;
}

}  // namespace

template <MetricType metric>
TurboQuantQuantizer<metric>::TurboQuantQuantizer(const TurboQuantQuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : Quantizer<TurboQuantQuantizer<metric>>(static_cast<int>(common_param.dim_),
                                             common_param.allocator_.get()),
      allocator_holder_(common_param.allocator_) {
    this->bits_per_dim_ = param->bits_per_dim_;
    this->variant_ = param->variant_;
    this->rotation_seed_ = param->rotation_seed_;
    SetupSizesAndMetric(common_param);
}

template <MetricType metric>
TurboQuantQuantizer<metric>::TurboQuantQuantizer(const QuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : TurboQuantQuantizer<metric>(std::dynamic_pointer_cast<TurboQuantQuantizerParameter>(param),
                                  common_param) {
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::UpdateQueryCodeSize() {
    if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                  metric == MetricType::METRIC_TYPE_COSINE) {
        // First dim floats: scratch for rotated-space code in ComputeDistImpl; second dim: z = Pi(q).
        this->query_code_size_ = 2ULL * static_cast<uint64_t>(this->dim_) * sizeof(float);
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        // q_unit [dim], ||q|| [1], z = Pi(q_unit) [dim]; first dim reused as y scratch in ComputeDist.
        this->query_code_size_ =
            (2ULL * static_cast<uint64_t>(this->dim_) + 1ULL) * sizeof(float);
    } else {
        const uint64_t query_floats =
            static_cast<uint64_t>(this->dim_) + (StoresNorm() ? 1ULL : 0ULL);
        this->query_code_size_ = query_floats * sizeof(float);
    }
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::SetupSizesAndMetric(const IndexCommonParam& common_param) {
    this->metric_ = common_param.metric_;
    this->packed_bytes_ = PackedBitsBytes(this->dim_, this->bits_per_dim_);
    this->code_size_ = this->packed_bytes_;
    if (StoresNorm()) {
        this->code_size_ += sizeof(float);
    }
    UpdateQueryCodeSize();
}

template <MetricType metric>
bool
TurboQuantQuantizer<metric>::StoresNorm() const {
    return metric == MetricType::METRIC_TYPE_L2SQR;
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::InitializeState() {
    this->sigma_ = 1.0F / std::sqrt(static_cast<float>(std::max<int64_t>(1LL, this->dim_)));
    ComputeGaussianScalarCodebook(this->bits_per_dim_, this->sigma_, this->centroids_);
    this->rotator_ = std::make_unique<FhtKacRotator>(this->allocator_, static_cast<int64_t>(this->dim_));
    this->rotator_->Train(this->rotation_seed_);
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::ApplyPi(const float* x, float* y) const {
    (void)this->rotator_->Transform(x, y);
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::ApplyPiTransposed(const float* y, float* x) const {
    this->rotator_->InverseTransform(y, x);
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::DecodeToRotatedSpace(const uint8_t* codes, float* y_rotated) const {
    const uint32_t k = static_cast<uint32_t>(this->centroids_.size());
    const uint32_t max_idx = k > 0 ? k - 1U : 0U;
    for (uint64_t j = 0; j < this->dim_; ++j) {
        uint32_t idx = ReadBits(codes, this->bits_per_dim_, j);
        idx = std::min(idx, max_idx);
        y_rotated[j] = this->centroids_[idx];
    }
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::DecodeUnitVector(const uint8_t* codes, float* out) const {
    Vector<float> y(this->allocator_);
    y.resize(this->dim_);
    DecodeToRotatedSpace(codes, y.data());
    ApplyPiTransposed(y.data(), out);
}

template <MetricType metric>
bool
TurboQuantQuantizer<metric>::TrainImpl(const DataType* /*data*/, uint64_t /*count*/) {
    InitializeState();
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
TurboQuantQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    Vector<float> work(this->allocator_);
    work.resize(this->dim_);
    Vector<float> y(this->allocator_);
    y.resize(this->dim_);

    float norm = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        norm = std::sqrt(FP32ComputeIP(data, data, this->dim_));
        if (norm > kNormEps) {
            for (uint64_t i = 0; i < this->dim_; ++i) {
                work[i] = data[i] / norm;
            }
        } else {
            std::fill(work.begin(), work.end(), 0.0F);
            norm = 0.0F;
        }
    } else {
        Normalize(data, work.data(), this->dim_);
    }

    ApplyPi(work.data(), y.data());

    std::memset(codes, 0, this->packed_bytes_);
    for (uint64_t j = 0; j < this->dim_; ++j) {
        uint32_t idx = NearestCentroidIndex(y[j], this->centroids_);
        WriteBits(codes, this->bits_per_dim_, j, idx);
    }
    if (StoresNorm()) {
        std::memcpy(codes + this->packed_bytes_, &norm, sizeof(float));
    }
    return true;
}

template <MetricType metric>
bool
TurboQuantQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
TurboQuantQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) const {
    DecodeUnitVector(codes, data);
    if (StoresNorm()) {
        float norm = 0.0F;
        std::memcpy(&norm, codes + this->packed_bytes_, sizeof(float));
        for (uint64_t i = 0; i < this->dim_; ++i) {
            data[i] *= norm;
        }
    }
    return true;
}

template <MetricType metric>
bool
TurboQuantQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes,
                                             DataType* data,
                                             uint64_t count) const {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
float
TurboQuantQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    Vector<float> a(this->allocator_);
    Vector<float> b(this->allocator_);
    a.resize(this->dim_);
    b.resize(this->dim_);
    this->DecodeOneImpl(codes1, a.data());
    this->DecodeOneImpl(codes2, b.data());

    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        return 1.0F - FP32ComputeIP(a.data(), b.data(), this->dim_);
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        return 1.0F - FP32ComputeIP(a.data(), b.data(), this->dim_);
    }
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(a.data(), b.data(), this->dim_);
    }
    return 0.0F;
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                              Computer<TurboQuantQuantizer<metric>>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
    } catch (const std::bad_alloc&) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
    float* buf = reinterpret_cast<float*>(computer.buf_);
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        float norm = std::sqrt(FP32ComputeIP(query, query, this->dim_));
        if (norm > kNormEps) {
            for (uint64_t i = 0; i < this->dim_; ++i) {
                buf[i] = query[i] / norm;
            }
            buf[this->dim_] = norm;
            ApplyPi(buf, buf + this->dim_ + 1);
        } else {
            std::fill(buf, buf + (2ULL * static_cast<uint64_t>(this->dim_) + 1ULL), 0.0F);
        }
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                        metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, buf, this->dim_);
        ApplyPi(buf, buf + this->dim_);
    }
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::ComputeDistImpl(Computer<TurboQuantQuantizer<metric>>& computer,
                                             const uint8_t* codes,
                                             float* dists) const {
    if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                  metric == MetricType::METRIC_TYPE_COSINE) {
        // <q, Pi^{-1} y> = <Pi q, y> when Pi preserves inner products (same y as scalar decode).
        float* y = reinterpret_cast<float*>(computer.buf_);
        const float* z = y + this->dim_;
        DecodeToRotatedSpace(codes, y);
        *dists = 1.0F - FP32ComputeIP(z, y, this->dim_);
        return;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        // || n_b Pi^{-1} y - q ||^2 = n_b^2 ||y||^2 + ||q||^2 - 2 n_b n_q <Pi u_q, y> (Pi orthogonal).
        float* buf = reinterpret_cast<float*>(computer.buf_);
        float* y = buf;
        const float nq = buf[this->dim_];
        const float* z = buf + this->dim_ + 1;
        DecodeToRotatedSpace(codes, y);
        float nb = 0.0F;
        std::memcpy(&nb, codes + this->packed_bytes_, sizeof(float));
        const float y_norm2 = FP32ComputeIP(y, y, this->dim_);
        const float dot_zy = FP32ComputeIP(z, y, this->dim_);
        *dists = nb * nb * y_norm2 + nq * nq - 2.0F * nb * nq * dot_zy;
        return;
    }

    *dists = 0.0F;
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::ScanBatchDistImpl(Computer<TurboQuantQuantizer<metric>>& computer,
                                               uint64_t count,
                                               const uint8_t* codes,
                                               float* dists) const {
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::ReleaseComputerImpl(
    Computer<TurboQuantQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
    computer.buf_ = nullptr;
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, this->bits_per_dim_);
    StreamWriter::WriteString(writer, this->variant_);
    StreamWriter::WriteObj(writer, this->rotation_seed_);
}

template <MetricType metric>
void
TurboQuantQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->bits_per_dim_);
    this->variant_ = StreamReader::ReadString(reader);
    StreamReader::ReadObj(reader, this->rotation_seed_);
    this->packed_bytes_ = PackedBitsBytes(this->dim_, this->bits_per_dim_);
    this->code_size_ = this->packed_bytes_;
    if (StoresNorm()) {
        this->code_size_ += sizeof(float);
    }
    UpdateQueryCodeSize();
    InitializeState();
}

TEMPLATE_QUANTIZER(TurboQuantQuantizer);

}  // namespace vsag
