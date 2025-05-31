#include "optimizer/sgd.hpp"

namespace dlt {
namespace optimizer {

SGD::SGD(float lr, float momentum) : lr_(lr), momentum_(momentum) {}

void SGD::add_parameters(const std::vector<TensorPtr>& params) {
    parameters_.insert(parameters_.end(), params.begin(), params.end());
    if (momentum_ > 0) {
        for (const auto& param : params) {
            velocities_.emplace_back(param->data().size(), 0.0f);
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        const auto& grad = param->grad();
        
        if (momentum_ > 0) {
            auto& velocity = velocities_[i];
            for (size_t j = 0; j < velocity.size(); ++j) {
                velocity[j] = momentum_ * velocity[j] + lr_ * grad[j];
                param->data()[j] -= velocity[j];
            }
        } else {
            for (size_t j = 0; j < param->data().size(); ++j) {
                param->data()[j] -= lr_ * grad[j];
            }
        }
    }
}

void SGD::zero_grad() {
    for (auto& param : parameters_) {
        param->zero_grad();
    }
}

} // namespace optimizer
} // namespace dlt