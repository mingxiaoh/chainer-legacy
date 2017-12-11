#include <vector>
#include <cstdlib>
#include "tensor.h"
#include "blas.h"

Tensor *Tensor::sum(vector<int> axis) {
    return blas_sum(this, axis);
}
