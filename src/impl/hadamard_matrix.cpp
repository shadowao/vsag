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

static void flip_sign(float* data, int dim, const uint8_t* flip)
{
    for (size_t i = 0; i < dim; i ++) {
        bool mask = flip[i/8] & (1 << (i%8));
        if (mask) {
          data[i] = -data[i];
        }
    }
}

static void kacs_walk(float* data, size_t len) {
    for (size_t i = 0; i < len / 2; i++) {
        float a = data[i];
        float b = data[i + (len / 2)];
        data[i] = a + b;
        data[i + (len / 2)] = a - b;
    }
}

HadamardMatrix::HadamardMatrix(uint64_t dim, Allocator* allocator) 
  : dim_(dim),  trunc_dim_(0), flip_(allocator), allocator_(allocator)

{
  trunc_dim_ = floor_log2(dim_);
  fac_ = std::sqrt(static_cast<float>(trunc_dim_));
  flip_.resize(4 * (dim_ + 7)/8);
  std::random_device rd;   // Seed
  std::mt19937 gen(rd());  // Mersenne Twister RNG
  // Uniform distribution in the range [0, 255]
  std::uniform_int_distribution<int> dist(0, 1);
  // Generate a single random uint8_t value
  for (auto& i : flip_) {
      i = static_cast<uint8_t>(dist(gen));
  }
}

void HadamardMatrix::Transform(const float* original_vec, float* transformed_vec) const{
    std::copy(original_vec, original_vec + dim_, transformed_vec);

    int32_t start = dim_ - trunc_dim_;
    if (0 == start) {
      flip_sign(transformed_vec, dim_, flip_.data());
      FHTRotate(transformed_vec, trunc_dim_);
      DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);

      flip_sign(transformed_vec, dim_, flip_.data() + (dim_ / 8));
      FHTRotate(transformed_vec + start, trunc_dim_);
      DivScalar(transformed_vec + start, transformed_vec + start, trunc_dim_, fac_);

      flip_sign(transformed_vec, dim_, flip_.data()+ (2 * dim_ / 8));
      FHTRotate(transformed_vec, trunc_dim_);
      DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);

      flip_sign(transformed_vec, dim_, flip_.data()+ (3 * dim_ / 8));
      FHTRotate(transformed_vec + start, trunc_dim_);
      DivScalar(transformed_vec + start, transformed_vec + start, trunc_dim_, fac_);

    } else {
      flip_sign(transformed_vec, dim_, flip_.data());
      FHTRotate(transformed_vec, trunc_dim_);
      DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);
      kacs_walk(transformed_vec, dim_);

      flip_sign(transformed_vec, dim_, flip_.data() + (dim_ / 8));
      FHTRotate(transformed_vec + start, trunc_dim_);
      DivScalar(transformed_vec + start, transformed_vec + start, trunc_dim_, fac_);
      kacs_walk(transformed_vec, dim_);

      flip_sign(transformed_vec, dim_, flip_.data()+ (2 * dim_ / 8));
      FHTRotate(transformed_vec, trunc_dim_);
      DivScalar(transformed_vec, transformed_vec, trunc_dim_, fac_);
      kacs_walk(transformed_vec, dim_);

      flip_sign(transformed_vec, dim_, flip_.data()+ (3 * dim_ / 8));
      FHTRotate(transformed_vec + start, trunc_dim_);
      DivScalar(transformed_vec + start, transformed_vec + start, trunc_dim_, fac_);
      kacs_walk(transformed_vec, dim_);
    }
}

void HadamardMatrix::InverseTransform(const float* transformed_vec, float* original_vec) const{
    Transform(transformed_vec, original_vec);
    for(int i = 0; i < dim_; i++){
        original_vec[i]/=dim_;
    }
}

void
HadamardMatrix::Serialize(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->flip_);
}

void
HadamardMatrix::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->flip_);
}

}//namespace vsag