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
        size_t size = param->data().size();
        m_.insert(m_.end(), size, 0.0f);
        v_.insert(v_.end(), size, 0.0f);
    }
}

void Adam::step() {
    t_++;
    const float beta1_correction = 1.0f - std::pow(beta1_, t_);
    const float beta2_correction = 1.0f - std::pow(beta2_, t_);
    const float lr_corrected = lr_ * std::sqrt(beta2_correction) / beta1_correction;

    size_t param_index = 0;
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];
        const auto& grad = param->grad();
        auto& data = param->data();
        
        size_t size = data.size();
        for (size_t j = 0; j < size; ++j) {
            const size_t idx = param_index + j;
            const float g = grad[j];
            
            // 更新一阶矩
            m_[idx] = beta1_ * m_[idx] + (1.0f - beta1_) * g;
            
            // 更新二阶矩
            v_[idx] = beta2_ * v_[idx] + (1.0f - beta2_) * g * g;
            
            // 计算更新量（更精确的公式）
            data[j] -= lr_corrected * m_[idx] / (std::sqrt(v_[idx]) + eps_);
        }
        param_index += size;
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