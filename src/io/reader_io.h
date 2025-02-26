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

#include "basic_io.h"
#include "vsag/readerset.h"

namespace vsag {

class ReaderIO : public BasicIO<ReaderIO> {
public:
    ReaderIO(std::shared_ptr<Reader> reader, Allocator* allocator) : BasicIO<ReaderIO>(allocator), reader_(reader) {}


    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        throw std::runtime_error("WriterIO is not supported");
    }

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        reader_->Read(offset, size, data);
        return true;
    }

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        auto data = (uint8_t*)allocator_->Allocate(size);
        reader_->Read(offset, size, data);
        need_release = true;
        return data;
    }

    inline void
    ReleaseImpl(const uint8_t* data) const {
        allocator_->Deallocate((void*)data);
    }

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        throw std::runtime_error("MultiReadImpl is not supported");
    }

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64){};

    static inline bool
    InMemoryImpl() {
        return false;
    }
private:
    std::shared_ptr<Reader> reader_;
};


}
