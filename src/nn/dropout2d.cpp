#include "nn/dropout2d.hpp"
#include "ops.hpp"
#include <random>

namespace dlt {
namespace nn {

Dropout2d::Dropout2d(float p) : p_(p) {}

TensorPtr Dropout2d::forward(const TensorPtr& x) {
    if (is_training_) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dis(1 - p_);

        std::vector<float> mask_data(x->size());
        for (size_t i = 0; i < mask_data.size(); ++i) {
            mask_data[i] = dis(gen) ? 1.0f / (1 - p_) : 0.0f;
        }

        auto mask = tensor(mask_data, x->shape(), false);
        return ops::mul(x, mask);
    } else {
        return x;
    }
}

std::vector<TensorPtr> Dropout2d::parameters() const {
    return {};
}

} // namespace nn
} // namespace dlt