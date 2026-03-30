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

#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "inner_string_params.h"
#include "parameter_test.h"

using namespace vsag;

struct TurboQuantDefaultParam {
    int turboquant_bits_per_dim = 4;
    uint64_t turboquant_rotation_seed = 42;
    std::string turboquant_variant = "mse";
};

std::string
generate_turboquant_param(const TurboQuantDefaultParam& param) {
    static constexpr auto param_str = R"(
        {{
            "type": "turboquant",
            "turboquant_bits_per_dim": {},
            "turboquant_rotation_seed": {},
            "turboquant_variant": "{}"
        }}
    )";
    return fmt::format(param_str,
                       param.turboquant_bits_per_dim,
                       param.turboquant_rotation_seed,
                       param.turboquant_variant);
}

#define TURBOQUANT_TEST_COMPATIBILITY_CASE(section_name, param_member, val1, val2, expect_compatible) \
    SECTION(section_name) {                                                                         \
        TurboQuantDefaultParam p1;                                                                  \
        TurboQuantDefaultParam p2;                                                                  \
        p1.param_member = val1;                                                                     \
        p2.param_member = val2;                                                                     \
        auto s1 = generate_turboquant_param(p1);                                                      \
        auto s2 = generate_turboquant_param(p2);                                                    \
        auto tq1 = std::make_shared<TurboQuantQuantizerParameter>();                                \
        auto tq2 = std::make_shared<TurboQuantQuantizerParameter>();                                \
        tq1->FromString(s1);                                                                        \
        tq2->FromString(s2);                                                                        \
        if (expect_compatible) {                                                                    \
            REQUIRE(tq1->CheckCompatibility(tq2));                                                    \
        } else {                                                                                    \
            REQUIRE_FALSE(tq1->CheckCompatibility(tq2));                                            \
        }                                                                                           \
    }

TEST_CASE("TurboQuant Quantizer Parameter CheckCompatibility", "[ut][TurboQuantQuantizerParameter]") {
    SECTION("wrong parameter type") {
        TurboQuantDefaultParam def;
        auto param_str = generate_turboquant_param(def);
        auto param = std::make_shared<TurboQuantQuantizerParameter>();
        param->FromString(param_str);
        REQUIRE(param->CheckCompatibility(param));
        REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));
    }
    TURBOQUANT_TEST_COMPATIBILITY_CASE("different bits_per_dim", turboquant_bits_per_dim, 4, 8, false)
    TURBOQUANT_TEST_COMPATIBILITY_CASE("different rotation_seed", turboquant_rotation_seed, 1ULL, 2ULL, false)
    TURBOQUANT_TEST_COMPATIBILITY_CASE("same config", turboquant_bits_per_dim, 4, 4, true)
    SECTION("different variant") {
        TurboQuantDefaultParam def;
        auto s = generate_turboquant_param(def);
        auto tq1 = std::make_shared<TurboQuantQuantizerParameter>();
        auto tq2 = std::make_shared<TurboQuantQuantizerParameter>();
        tq1->FromString(s);
        tq2->FromString(s);
        tq2->variant_ = "not_mse";
        REQUIRE_FALSE(tq1->CheckCompatibility(tq2));
    }
}

TEST_CASE("TurboQuant Quantizer Parameter invalid bits_per_dim", "[ut][TurboQuantQuantizerParameter]") {
    TurboQuantDefaultParam def;
    auto tq = std::make_shared<TurboQuantQuantizerParameter>();
    def.turboquant_bits_per_dim = 0;
    REQUIRE_THROWS(tq->FromString(generate_turboquant_param(def)));
    def.turboquant_bits_per_dim = 9;
    REQUIRE_THROWS(tq->FromString(generate_turboquant_param(def)));
}

TEST_CASE("TurboQuant Quantizer Parameter ToJson round-trip", "[ut][TurboQuantQuantizerParameter]") {
    TurboQuantDefaultParam def;
    def.turboquant_bits_per_dim = 6;
    def.turboquant_rotation_seed = 999ULL;
    auto param_str = generate_turboquant_param(def);
    auto tq = std::make_shared<TurboQuantQuantizerParameter>();
    tq->FromString(param_str);
    auto out = tq->ToJson();
    REQUIRE(out[TYPE_KEY].GetString() == QUANTIZATION_TYPE_VALUE_TURBOQUANT);
    REQUIRE(static_cast<uint32_t>(out[TURBOQUANT_BITS_PER_DIM_KEY].GetInt()) == 6);
    REQUIRE(static_cast<uint64_t>(out[TURBOQUANT_ROTATION_SEED_KEY].GetInt()) == 999ULL);
}
