#include "hadamard_matrix.h"
//#include <immintrin.h> // AVX-512 intrinsics

namespace vsag
{

static size_t ceil_log2(size_t val) {
    size_t res = 0;
    for (size_t i = 0; i < 31; ++i) {
        if ((1U << i) >= val) {
            res = i;
            break;
        }
    }
    return 1 << res;
}

    HadamardMatrix::HadamardMatrix(uint64_t dim, Allocator* allocator) 
      : dim_(dim),  pad_dim_(0), pad_vec_(allocator), allocator_(allocator)

    {
      pad_dim_ = ceil_log2(dim_);
      pad_vec_.resize(pad_dim_);
    }

    void HadamardMatrix::Transform(const float* original_vec, float* transformed_vec) const{
        int n = pad_dim_;
        int step = 1;
        std::copy(original_vec, original_vec + dim_, pad_vec_.data());
        std::fill(pad_vec_.data() + dim_, pad_vec_.data() + pad_dim_, 0.0F);
        // 逐步合并
        while (step < n) {
            for (int i = 0; i < n; i += step * 2) {
                // if(step>8 && step % 8 == 0){
                //     for (int j = 0; j < step; j+=16) {
                //         __m256 g1 = _mm256_loadu_ps(&pad_vec_[i + j]);
                //         __m256 g2 = _mm256_loadu_ps(&pad_vec_[i + j + step]);
                //         _mm256_storeu_ps(&pad_vec_[i + j],_mm256_add_ps(g1, g2));
                //         _mm256_storeu_ps(&pad_vec_[i + j + step],_mm256_sub_ps(g1, g2));
                //     }
                // } else
                {
                    for (int j = 0; j < step; j++) {
                        // 合并操作
                        float even = pad_vec_[i + j];
                        float odd = pad_vec_[i + j + step];
                        // 更新数组
                        pad_vec_[i + j] = even + odd;         // 相加
                        pad_vec_[i + j + step] = even - odd; // 相减
                    }
                }
            }
            step *= 2; // 增加步长
        }
        std::copy(pad_vec_.data(), pad_vec_.data() + dim_, transformed_vec);
    }

    void HadamardMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const{
        Transform(transformed_vec, original_vec);
        for(int i = 0; i < dim_; i++){
            original_vec[i]/=dim_;
        }
        //利用Hadamard矩阵的特性做逆运算
    }

}//namespace vsag