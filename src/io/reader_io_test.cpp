
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

#include "reader_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "safe_allocator.h"
#include "test_reader.h"

using namespace vsag;

TEST_CASE("ReaderIO Read Test", "[ut][ReaderIO]") {
    vsag::Binary binary;
    int8_t data_size = 100;
    binary.data.reset(new int8_t[data_size]);
    binary.size = data_size;
    for (int8_t i = 0; i < data_size; i++) {
        binary.data[i] = i;
    }
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<ReaderIO>(std::make_shared<fixtures::TestReader>(binary), allocator.get());
    auto data = (uint8_t*)allocator->Allocate(data_size);
    io->Read(data_size, 0, data);
    for (int i = 0; i < data_size; i++) {
        REQUIRE(static_cast<int>(((int8_t*)data)[i]) == i);
    }
    allocator->Deallocate(data);
}