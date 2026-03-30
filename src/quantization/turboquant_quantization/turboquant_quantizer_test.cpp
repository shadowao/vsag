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

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "inner_string_params.h"
#include "quantization/quantizer_parameter.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "storage/serialization_template_test.h"
#include "turboquant_codebook.h"

using namespace vsag;

namespace {

TurboQuantQuantizerParamPtr
MakeTurboParam(uint32_t bits_per_dim, uint64_t rotation_seed) {
    auto param = std::make_shared<TurboQuantQuantizerParameter>();
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    json[TURBOQUANT_BITS_PER_DIM_KEY].SetInt(bits_per_dim);
    json[TURBOQUANT_VARIANT_KEY].SetString("mse");
    json[TURBOQUANT_ROTATION_SEED_KEY].SetInt(rotation_seed);
    param->FromJson(json);
    return param;
}

IndexCommonParam
MakeCommon(int64_t dim, const std::shared_ptr<Allocator>& allocator, MetricType metric) {
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = metric;
    return common;
}

template <MetricType metric>
std::shared_ptr<TurboQuantQuantizer<metric>>
MakeQuantizer(int64_t dim, uint32_t bits, uint64_t seed) {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    auto param = MakeTurboParam(bits, seed);
    auto common = MakeCommon(dim, alloc, metric);
    auto q = std::make_shared<TurboQuantQuantizer<metric>>(param, common);
    std::vector<float> dummy(static_cast<size_t>(dim), 0.0F);
    REQUIRE(q->Train(dummy.data(), 1));
    return q;
}

template <MetricType metric>
void
TestSelfDistanceSmall(int64_t dim, uint32_t bits, uint64_t seed, int count) {
    auto quant = MakeQuantizer<metric>(dim, bits, seed);
    REQUIRE(quant->NameImpl() == QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    auto vecs = fixtures::generate_vectors(count, static_cast<uint32_t>(dim), false);
    for (int i = 0; i < count; ++i) {
        const float* vec = vecs.data() + i * dim;
        auto computer = quant->FactoryComputer();
        computer->SetQuery(vec);
        std::vector<uint8_t> code(quant->GetCodeSize());
        quant->EncodeOne(vec, code.data());
        float d = quant->ComputeDist(*computer, code.data());
        REQUIRE(std::abs(d) < 0.35F);
    }
}

}  // namespace

TEST_CASE("TurboQuant name and parameter factory", "[ut][TurboQuantQuantizer]") {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    json[TURBOQUANT_BITS_PER_DIM_KEY].SetInt(4);
    json[TURBOQUANT_VARIANT_KEY].SetString("mse");
    json[TURBOQUANT_ROTATION_SEED_KEY].SetInt(7);
    auto p = QuantizerParameter::GetQuantizerParameterByJson(json);
    REQUIRE(p->GetTypeName() == QUANTIZATION_TYPE_VALUE_TURBOQUANT);
}

TEST_CASE("TurboQuant invalid variant rejected", "[ut][TurboQuantQuantizer]") {
    JsonType json;
    json[TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    json[TURBOQUANT_BITS_PER_DIM_KEY].SetInt(4);
    json[TURBOQUANT_VARIANT_KEY].SetString("prod");
    REQUIRE_THROWS(QuantizerParameter::GetQuantizerParameterByJson(json));
}

TEST_CASE("TurboQuant self distance small after encode", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 48;
    const uint32_t bits = 4;
    const uint64_t seed = 424242ULL;
    const int count = 20;
    SECTION("L2SQR") {
        TestSelfDistanceSmall<MetricType::METRIC_TYPE_L2SQR>(dim, bits, seed, count);
    }
    SECTION("COSINE") {
        TestSelfDistanceSmall<MetricType::METRIC_TYPE_COSINE>(dim, bits, seed, count);
    }
    SECTION("IP") {
        TestSelfDistanceSmall<MetricType::METRIC_TYPE_IP>(dim, bits, seed, count);
    }
}

TEST_CASE("TurboQuant encode one matches encode batch", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 32;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_COSINE>(dim, 4, 99ULL);
    const int count = 11;
    auto vecs = fixtures::generate_vectors(count, static_cast<uint32_t>(dim), true);

    std::vector<uint8_t> codes1(quant->GetCodeSize() * count);
    std::vector<uint8_t> codes2(quant->GetCodeSize() * count);
    for (int i = 0; i < count; ++i) {
        quant->EncodeOne(vecs.data() + i * dim, codes1.data() + i * quant->GetCodeSize());
    }
    quant->EncodeBatch(vecs.data(), codes2.data(), count);
    REQUIRE(codes1 == codes2);
}

TEST_CASE("TurboQuant same seed same codes", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 40;
    const uint64_t seed = 10007ULL;
    auto q1 = MakeQuantizer<MetricType::METRIC_TYPE_L2SQR>(dim, 4, seed);
    auto q2 = MakeQuantizer<MetricType::METRIC_TYPE_L2SQR>(dim, 4, seed);
    auto vec = fixtures::generate_vectors(1, static_cast<uint32_t>(dim), false);
    std::vector<uint8_t> c1(q1->GetCodeSize());
    std::vector<uint8_t> c2(q2->GetCodeSize());
    q1->EncodeOne(vec.data(), c1.data());
    q2->EncodeOne(vec.data(), c2.data());
    REQUIRE(c1 == c2);
}

TEST_CASE("TurboQuant different seeds different codes", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 40;
    auto q1 = MakeQuantizer<MetricType::METRIC_TYPE_L2SQR>(dim, 4, 1ULL);
    auto q2 = MakeQuantizer<MetricType::METRIC_TYPE_L2SQR>(dim, 4, 2ULL);
    auto vec = fixtures::generate_vectors(1, static_cast<uint32_t>(dim), false);
    std::vector<uint8_t> c1(q1->GetCodeSize());
    std::vector<uint8_t> c2(q2->GetCodeSize());
    q1->EncodeOne(vec.data(), c1.data());
    q2->EncodeOne(vec.data(), c2.data());
    REQUIRE_FALSE(c1 == c2);
}

TEST_CASE("TurboQuant serialize deserialize preserves codes", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 36;
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    auto param = MakeTurboParam(4, 555ULL);
    auto common = MakeCommon(dim, alloc, MetricType::METRIC_TYPE_COSINE);

    TurboQuantQuantizer<MetricType::METRIC_TYPE_COSINE> q1(param, common);
    std::vector<float> dummy(static_cast<size_t>(dim), 0.0F);
    REQUIRE(q1.Train(dummy.data(), 1));

    TurboQuantQuantizer<MetricType::METRIC_TYPE_COSINE> q2(param, common);
    test_serializion(q1, q2);

    REQUIRE(q1.GetCodeSize() == q2.GetCodeSize());
    REQUIRE(q1.GetDim() == q2.GetDim());

    auto vec = fixtures::generate_vectors(1, static_cast<uint32_t>(dim), true);
    std::vector<uint8_t> a(q1.GetCodeSize());
    std::vector<uint8_t> b(q2.GetCodeSize());
    q1.EncodeOne(vec.data(), a.data());
    q2.EncodeOne(vec.data(), b.data());
    REQUIRE(a == b);
}

TEST_CASE("TurboQuant compute two codes vs decode distance cosine", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 24;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_COSINE>(dim, 4, 777ULL);
    auto vecs = fixtures::generate_vectors(2, static_cast<uint32_t>(dim), true);
    std::vector<uint8_t> c0(quant->GetCodeSize());
    std::vector<uint8_t> c1(quant->GetCodeSize());
    quant->EncodeOne(vecs.data(), c0.data());
    quant->EncodeOne(vecs.data() + dim, c1.data());

    float d_compute = quant->Compute(c0.data(), c1.data());

    std::vector<float> u0(static_cast<size_t>(dim));
    std::vector<float> u1(static_cast<size_t>(dim));
    quant->DecodeOne(c0.data(), u0.data());
    quant->DecodeOne(c1.data(), u1.data());
    float gt = 1.0F - FP32ComputeIP(u0.data(), u1.data(), static_cast<uint64_t>(dim));
    REQUIRE(std::abs(d_compute - gt) < 1e-4F);
}

TEST_CASE("TurboQuant cosine decode aligns with original direction", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 64;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_COSINE>(dim, 4, 12345ULL);
    auto vec = fixtures::generate_vectors(1, static_cast<uint32_t>(dim), false);
    std::vector<float> orig_norm(static_cast<size_t>(dim));
    Normalize(vec.data(), orig_norm.data(), static_cast<uint32_t>(dim));

    std::vector<uint8_t> code(quant->GetCodeSize());
    quant->EncodeOne(vec.data(), code.data());
    std::vector<float> dec(static_cast<size_t>(dim));
    quant->DecodeOne(code.data(), dec.data());

    const float dec_norm =
        std::sqrt(FP32ComputeIP(dec.data(), dec.data(), static_cast<uint64_t>(dim)));
    const float ip = FP32ComputeIP(orig_norm.data(), dec.data(), static_cast<uint64_t>(dim));
    const float cos_sim = ip / std::max(1e-12F, dec_norm);
    REQUIRE(cos_sim > 0.45F);
}

TEST_CASE("TurboQuant Gaussian codebook size and order", "[ut][TurboQuantQuantizer]") {
    std::vector<float> c2;
    ComputeGaussianScalarCodebook(2, 0.14F, c2);
    REQUIRE(c2.size() == 4);
    for (size_t i = 1; i < c2.size(); ++i) {
        REQUIRE(c2[i] >= c2[i - 1]);
    }
    std::vector<float> c8;
    ComputeGaussianScalarCodebook(8, 0.05F, c8);
    REQUIRE(c8.size() == 256);
}

TEST_CASE("TurboQuant bits 2 and 8 self distance", "[ut][TurboQuantQuantizer]") {
    SECTION("2 bits") {
        TestSelfDistanceSmall<MetricType::METRIC_TYPE_IP>(32, 2, 5000ULL, 8);
    }
    SECTION("8 bits") {
        TestSelfDistanceSmall<MetricType::METRIC_TYPE_IP>(24, 8, 6000ULL, 8);
    }
}

TEST_CASE("TurboQuant L2 zero vector self distance", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 32;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_L2SQR>(dim, 4, 303ULL);
    std::vector<float> zeros(static_cast<size_t>(dim), 0.0F);
    std::vector<uint8_t> code(quant->GetCodeSize());
    REQUIRE(quant->EncodeOne(zeros.data(), code.data()));
    std::vector<float> out(static_cast<size_t>(dim));
    REQUIRE(quant->DecodeOne(code.data(), out.data()));
    float sum_sq = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        sum_sq += out[i] * out[i];
    }
    REQUIRE(std::sqrt(sum_sq) < 1e-4F);

    auto computer = quant->FactoryComputer();
    computer->SetQuery(zeros.data());
    float d = quant->ComputeDist(*computer, code.data());
    REQUIRE(std::abs(d) < 1e-4F);
}

TEST_CASE("TurboQuant decode batch matches decode one", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 28;
    const int count = 13;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_L2SQR>(dim, 4, 909ULL);
    auto vecs = fixtures::generate_vectors(count, static_cast<uint32_t>(dim), false);
    std::vector<uint8_t> codes(quant->GetCodeSize() * count);
    quant->EncodeBatch(vecs.data(), codes.data(), count);

    std::vector<float> one(static_cast<size_t>(dim) * count);
    std::vector<float> batch(static_cast<size_t>(dim) * count);
    for (int i = 0; i < count; ++i) {
        quant->DecodeOne(codes.data() + i * quant->GetCodeSize(), one.data() + i * dim);
    }
    quant->DecodeBatch(codes.data(), batch.data(), count);
    for (size_t i = 0; i < static_cast<size_t>(dim) * count; ++i) {
        REQUIRE(std::abs(one[i] - batch[i]) < 1e-6F);
    }
}

TEST_CASE("TurboQuant ScanBatchDist matches ComputeDist", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 20;
    const uint64_t count = 7;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_IP>(dim, 4, 4242ULL);
    auto vecs = fixtures::generate_vectors(static_cast<int>(count) + 1,
                                           static_cast<uint32_t>(dim),
                                           true);
    const float* query = vecs.data();
    std::vector<uint8_t> codes(quant->GetCodeSize() * count);
    quant->EncodeBatch(vecs.data() + dim, codes.data(), count);

    auto computer = quant->FactoryComputer();
    computer->SetQuery(query);

    std::vector<float> scan_dists(count);
    std::vector<float> single_dists(count);
    quant->ScanBatchDists(*computer, count, codes.data(), scan_dists.data());
    for (uint64_t i = 0; i < count; ++i) {
        single_dists[i] =
            quant->ComputeDist(*computer, codes.data() + i * quant->GetCodeSize());
        REQUIRE(std::abs(scan_dists[i] - single_dists[i]) < 1e-6F);
    }
}

TEST_CASE("TurboQuant IP compute two codes vs cosine distance", "[ut][TurboQuantQuantizer]") {
    const int64_t dim = 24;
    auto quant = MakeQuantizer<MetricType::METRIC_TYPE_IP>(dim, 4, 1313ULL);
    auto vecs = fixtures::generate_vectors(2, static_cast<uint32_t>(dim), true);
    std::vector<uint8_t> c0(quant->GetCodeSize());
    std::vector<uint8_t> c1(quant->GetCodeSize());
    quant->EncodeOne(vecs.data(), c0.data());
    quant->EncodeOne(vecs.data() + dim, c1.data());

    float d_compute = quant->Compute(c0.data(), c1.data());

    std::vector<float> u0(static_cast<size_t>(dim));
    std::vector<float> u1(static_cast<size_t>(dim));
    quant->DecodeOne(c0.data(), u0.data());
    quant->DecodeOne(c1.data(), u1.data());
    float gt = 1.0F - FP32ComputeIP(u0.data(), u1.data(), static_cast<uint64_t>(dim));
    REQUIRE(std::abs(d_compute - gt) < 1e-4F);
}
