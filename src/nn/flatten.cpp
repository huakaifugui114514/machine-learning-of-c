#include "nn/flatten.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <numeric>
#include <vector>
#include <memory>

namespace dlt {
namespace nn {

Flatten::Flatten() {}

TensorPtr Flatten::forward(const TensorPtr& x) {
    const auto& shape = x->shape();
    if (shape.size() < 1) {
        throw std::invalid_argument("Flatten requires non-empty tensor");
    }
    
    // 确保张量连续
    TensorPtr contiguous_x = ops::contiguous(x);
    
    const int batch_size = shape[0];
    int total_size = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            throw std::invalid_argument("Invalid dimension size in Flatten");
        }
        total_size *= shape[i];
    }
    
    std::vector<int> new_shape = {batch_size, total_size};
    return ops::reshape(contiguous_x, new_shape);
}

std::vector<TensorPtr> Flatten::parameters() const {
    return {};
}

} // namespace nn
} // namespace dlt