// pj/src/optimizer/adam.cpp
#include "optimizer/adam.hpp"

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
        
        for (size_t j = 0; j < param->data().size(); ++j) {
            // 更新一阶矩估计
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * grad[j];
            
            // 更新二阶矩估计
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * grad[j] * grad[j];
            
            // 计算偏置校正后的估计
            const float m_hat = m[j] / bias_correction1;
            const float v_hat = v[j] / bias_correction2;
            param->data()[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void Adam::zero_grad() {
    for (auto& param : parameters_) {
        param->zero_grad();
    }
}

// 实现 get_parameters 函数
const std::vector<TensorPtr>& Adam::get_parameters() const {
    return parameters_;
}

} // namespace optimizer
} // namespace dlt