#include <iostream>
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
        std::cout << "sgd params:" << std::endl;
        param->print();
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

        // 添加梯度裁剪，防止爆炸
        std::vector<float> clipped_grad(grad.size());
        float max_grad = 1.0f;  // 梯度裁剪阈值
        for (size_t j = 0; j < grad.size(); ++j) {
            clipped_grad[j] = std::max(-max_grad, std::min(grad[j], max_grad));
        }

        // 使用裁剪后的梯度
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