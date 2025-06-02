// pj/src/optimizer/adam.cpp
#include "optimizer/adam.hpp"
#include <cmath>

namespace dlt {
namespace optimizer {

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {}

void Adam::add_parameters(const std::vector<TensorPtr>& params) {
    parameters_.insert(parameters_.end(), params.begin(), params.end());
    for (const auto& param : params) {
        m_.emplace_back(param->data().size(), 0.0f);
        v_.emplace_back(param->data().size(), 0.0f);
    }
}

void Adam::step() {
    t_++;
    const float bias_correction1 = 1.0f - std::pow(beta1_, t_);
    const float bias_correction2 = 1.0f - std::pow(beta2_, t_);

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        const auto& grad = param->grad();
        auto& m = m_[i];
        auto& v = v_[i];
        auto& data = param->data();
        
        for (size_t j = 0; j < data.size(); ++j) {
            const float g = grad[j];
            
            // 更新一阶矩
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * g;
            
            // 更新二阶矩
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * g * g;
            
            // 计算偏置校正后的矩估计
            const float m_hat = m[j] / bias_correction1;
            const float v_hat = v[j] / bias_correction2;
            
            // 按标准公式更新参数
            data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void Adam::zero_grad() {
    for (auto& param : parameters_) {
        param->zero_grad();
    }
}

const std::vector<TensorPtr>& Adam::get_parameters() const {
    return parameters_;
}

} // namespace optimizer
} // namespace dlt