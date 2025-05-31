#include "nn/flatten.hpp"
#include "ops.hpp"

namespace dlt {
namespace nn {

Flatten::Flatten() {}

TensorPtr Flatten::forward(const TensorPtr& x) {
    int batch_size = x->shape()[0];
    int total_size = 1;
    for (size_t i = 1; i < x->shape().size(); ++i) {
        total_size *= x->shape()[i];
    }
    std::vector<int> new_shape = {batch_size, total_size};
    return ops::reshape(x, new_shape);
}

std::vector<TensorPtr> Flatten::parameters() const {
    return {};
}

} // namespace nn
} // namespace dlt