
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

#include "async_io.h"

#include <filesystem>

namespace vsag {
std::unique_ptr<IOContextPool> AsyncIO::io_context_pool =
    std::make_unique<IOContextPool>(10, nullptr);

AsyncIO::~AsyncIO() {
    close(this->wfd_);
    close(this->rfd_);
    // remove file
    std::filesystem::remove(this->filepath_);
}

}  // namespace vsag
