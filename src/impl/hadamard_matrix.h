#include <cblas.h>
#include <lapacke.h>
#include "matrix_rotator.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include <random>
#include "../logger.h"
// #include "typing.h"
// #include "vsag/allocator.h"
namespace vsag{
    
class HadamardMatrix : public MatrixRotator {
    public:
    HadamardMatrix(uint64_t dim, Allocator* allocator);
    virtual ~HadamardMatrix() {}
    // void CopyOrthogonalMatrix() const;

    void Transform(const float* original_vec, float* transformed_vec) const override;

    void InverseTransform(const float* transformed_vec, float* original_vec) const override;

    bool GenerateRandomOrthogonalMatrix() override { return true; }

    void GenerateRandomOrthogonalMatrixWithRetry() override {}

    double ComputeDeterminant() const override { return 1.0; }

    void Serialize(StreamWriter& writer) override {}

    void Deserialize(StreamReader& reader) override {}


    private:
        const uint64_t dim_{0};
        uint32_t trunc_dim_{0};
        float fac_{0.0};
        Allocator* const allocator_{nullptr};


};

}