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

#include <cstdint>
#include <vector>

namespace vsag {

/**
 * Build Lloyd–Max-style centroids for N(0, sigma^2) with K = 2^bits levels.
 * Uses a fixed deterministic sample set so the table depends only on (bits, sigma).
 * For large dim, sigma is typically 1/sqrt(dim) (spherical / Gaussian marginal approximation).
 */
void
ComputeGaussianScalarCodebook(uint32_t bits_per_dim, float sigma, std::vector<float>& centroids);

}  // namespace vsag
