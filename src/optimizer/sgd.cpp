// sgd.cpp
#include "optimizer/sgd.hpp"
#include <stdexcept>

namespace dlt {
namespace optimizer {

SGD::SGD(float lr, float momentum, float clip_value) 
    : lr_(lr), momentum_(momentum), clip_value_(clip_value) {
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive.");
    }
    if (momentum < 0 || momentum > 1) {
        throw std::invalid_argument("Momentum must be in the range [0, 1].");
    }
    if (clip_value <= 0) {
        throw std::invalid_argument("Gradient clip value must be positive.");
    }
}

void SGD::add_parameters(const std::vector<std::shared_ptr<Tensor>>& params) {
    for (const auto& param : params) {
        if (!param) {
            throw std::invalid_argument("Parameter tensor cannot be null.");
        }
    }

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
        auto& data = param->data();

        // 梯度裁剪
        std::vector<float> clipped_grad(grad.size());
        for (size_t j = 0; j < grad.size(); ++j) {
            clipped_grad[j] = std::max(-clip_value_, std::min(grad[j], clip_value_));
        }

        // 更新参数
        if (momentum_ > 0) {
            auto& velocity = velocities_[i];
            for (size_t j = 0; j < data.size(); ++j) {
                velocity[j] = momentum_ * velocity[j] + lr_ * clipped_grad[j];
                data[j] -= velocity[j];
            }
        } else {
            for (size_t j = 0; j < data.size(); ++j) {
                data[j] -= lr_ * clipped_grad[j];
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