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

static void vec_scale(float* data, int dim, float fac)
{
    for (int i = 0; i < dim; ++i) {
      data[i] = data[i] * fac;
    }
}

    HadamardMatrix::HadamardMatrix(uint64_t dim, Allocator* allocator) 
      : dim_(dim),  pad_dim_(0), allocator_(allocator)

    {
      pad_dim_ = ceil_log2(dim_);
      fac_ = std::sqrt(static_cast<float>(pad_dim_));
    }

    void HadamardMatrix::Transform(const float* original_vec, float* transformed_vec) const{
        std::vector<float> pad_vec(pad_dim_, 0);
        std::copy(original_vec, original_vec + dim_, pad_vec.data());
        std::fill(pad_vec.data() + dim_, pad_vec.data() + pad_dim_, 0.0F);

        FHTRotate(pad_vec.data(), pad_dim_);
        DivScalar(pad_vec.data(), pad_vec.data(), pad_dim_, fac_);
        std::copy(pad_vec.data(), pad_vec.data() + dim_, transformed_vec);
    }

    void HadamardMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const{
        Transform(transformed_vec, original_vec);
        for(int i = 0; i < dim_; i++){
            original_vec[i]/=dim_;
        }
        //利用Hadamard矩阵的特性做逆运算
    }

}//namespace vsag