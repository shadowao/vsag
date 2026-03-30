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

#include "turboquant_codebook.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace vsag {

void
ComputeGaussianScalarCodebook(uint32_t bits_per_dim, float sigma, std::vector<float>& centroids) {
    const uint32_t k = 1U << bits_per_dim;
    centroids.assign(k, 0.0F);

    // Deterministic synthetic data for Lloyd–Max; depends on (bits_per_dim, sigma).
    uint32_t sigma_bits = 0;
    std::memcpy(&sigma_bits, &sigma, sizeof(float));
    const uint64_t seed = 0x9E3779B97F4A7C15ULL ^
                          (static_cast<uint64_t>(bits_per_dim) * 0xD6E8FEB8665FD96DULL) ^
                          static_cast<uint64_t>(sigma_bits);
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(0.0F, sigma);

    const int n_samples = 65536;
    std::vector<float> samples(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        samples[i] = dist(gen);
    }
    std::sort(samples.begin(), samples.end());

    // k-means++ style init: pick k points spread along sorted samples
    for (uint32_t i = 0; i < k; ++i) {
        size_t idx = static_cast<size_t>((static_cast<double>(i) + 0.5) * n_samples / k);
        idx = std::min(idx, samples.size() - 1);
        centroids[i] = samples[idx];
    }
    std::sort(centroids.begin(), centroids.end());

    std::vector<float> sum(k, 0.0F);
    std::vector<int> count(k, 0);

    for (int iter = 0; iter < 40; ++iter) {
        std::fill(sum.begin(), sum.end(), 0.0F);
        std::fill(count.begin(), count.end(), 0);

        for (float s : samples) {
            uint32_t best = 0;
            float best_d = std::abs(s - centroids[0]);
            for (uint32_t j = 1; j < k; ++j) {
                float d = std::abs(s - centroids[j]);
                if (d < best_d) {
                    best_d = d;
                    best = j;
                }
            }
            sum[best] += s;
            count[best] += 1;
        }

        for (uint32_t j = 0; j < k; ++j) {
            if (count[j] > 0) {
                centroids[j] = sum[j] / static_cast<float>(count[j]);
            }
        }
        std::sort(centroids.begin(), centroids.end());
    }
}

}  // namespace vsag
