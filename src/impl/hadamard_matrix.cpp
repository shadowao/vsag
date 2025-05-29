#include "hadamard_matrix.h"
#include "simd/rabitq_simd.h"
#include "simd/normalize.h"

namespace vsag
{

static size_t ceil_log2(size_t val)
{
    size_t res = 0;
    for (size_t i = 0; i < 31; ++i) {
        if ((1U << i) >= val) {
            res = i;
            break;
        }
    }
    return 1 << res;
}

static size_t floor_log2(size_t x) {
    size_t ret = 0;
    while (x > 1) {
        ret++;
        x >>= 1;
    }
    return 1 << ret;
}

static void vec_scale(float* data, int dim, float fac)
{
    for (int i = 0; i < dim; ++i) {
      data[i] = data[i] * fac;
    }
}

    HadamardMatrix::HadamardMatrix(uint64_t dim, Allocator* allocator) 
      : dim_(dim),  trunc_dim_(0), allocator_(allocator)

    {
      trunc_dim_ = floor_log2(dim_);
      fac_ = std::sqrt(static_cast<float>(trunc_dim_));
    }

    void HadamardMatrix::Transform(const float* original_vec, float* transformed_vec) const{
        std::copy(original_vec, original_vec + dim_, transformed_vec);

        int32_t start = dim_ - trunc_dim_;
        if (0 == start) {
          FHTRotate(transformed_vec, trunc_dim_);
          DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);
        } else {
          FHTRotate(transformed_vec, trunc_dim_);
          DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);

          FHTRotate(transformed_vec + start, trunc_dim_);
          DivScalar(transformed_vec + start, transformed_vec + start, trunc_dim_, fac_);

          FHTRotate(transformed_vec, trunc_dim_);
          DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);

          FHTRotate(transformed_vec + start, trunc_dim_);
          DivScalar(transformed_vec + start, transformed_vec + start, trunc_dim_, fac_);
        }
    }

    void HadamardMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const{
        Transform(transformed_vec, original_vec);
        for(int i = 0; i < dim_; i++){
            original_vec[i]/=dim_;
        }
        //利用Hadamard矩阵的特性做逆运算
    }

}//namespace vsag