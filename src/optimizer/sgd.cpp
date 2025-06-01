#include "optimizer/sgd.hpp"
#include <stdexcept>

namespace dlt {
namespace optimizer {

SGD::SGD(float lr, float momentum) : lr_(lr), momentum_(momentum) {
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive.");
    }
    if (momentum < 0 || momentum > 1) {
        throw std::invalid_argument("Momentum must be in the range [0, 1].");
    }
}

void SGD::add_parameters(const std::vector<TensorPtr>& params) {
    for (const auto& param : params) {
        if (!param) {
            throw std::invalid_argument("Parameter tensor cannot be null.");
        }
    }
    std::lock_guard<std::mutex> lock(mtx_);
    parameters_.insert(parameters_.end(), params.begin(), params.end());
    if (momentum_ > 0) {
        for (const auto& param : params) {
            velocities_.emplace_back(param->data().size(), 0.0f);
        }
    }
}

void SGD::step() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        const auto& grad = param->grad();
        auto& data = param->data();

        if (momentum_ > 0) {
            auto& velocity = velocities_[i];
            for (size_t j = 0; j < data.size(); ++j) {
                velocity[j] = momentum_ * velocity[j] + lr_ * grad[j];
                data[j] -= velocity[j];
            }
        } else {
            for (size_t j = 0; j < data.size(); ++j) {
                data[j] -= lr_ * grad[j];
            }
        }
    }
}

void SGD::zero_grad() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& param : parameters_) {
        param->zero_grad();
    }
}

} // namespace optimizer
} // namespace dlt