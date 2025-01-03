
#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pyvsag
import numpy as np
import json

dim = 64                     # dimension
num_elements = 10000         # database size
np.random.seed(47)           # make reproducible
data = np.float32(np.random.random((num_elements, dim)))
ids = range(num_elements)

index_parameter = {
    "dtype": "float32",
    "metric_type": "l2",
    "dim": dim,
    "hnsw": {"max_degree": 16, "ef_construction": 100}
}
index = pyvsag.Index(name="hnsw", parameters=json.dumps(index_parameter))
index.build(vectors=data, ids=ids, num_elements=num_elements, dim=dim)

query = np.float32(np.random.rand(dim))                 # query.shape: (128,)
search_parameter = {"hnsw": {"ef_search": 100}}         # search list size
_ids, dists = index.knn_search(vector=query,
                               k=5,
                               parameters=json.dumps(search_parameter))
print(_ids)
