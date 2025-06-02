#include "loss/mse_loss.hpp"
#include "ops.hpp"
#include <stdexcept>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::loss;

TensorPtr MSELoss::forward(const TensorPtr& input, const TensorPtr& target) {
    // 检查输入和目标张量是否为有效的指针
    if (!input || !target) {
        throw std::invalid_argument("Input and target tensors must not be null");
    }

    // 形状检查
    if (input->shape() != target->shape()) {
        throw std::invalid_argument("Input and target shapes must match");
    }

    // 直接计算差值和平方
    diff_ = sub(input, target);
    auto squared = mul(diff_, diff_);
    
    // 计算平均值
    float n = static_cast<float>(input->size());
    auto loss = sum(squared);
    auto loss_final = mul(loss, tensor({1.0f / n}, {1}));
    
    return loss_final;
}

std::vector<TensorPtr> MSELoss::backward() {
    // 检查是否已经调用了 forward 方法
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    float scale = 2.0f / input_->size();
    auto grad_input = mul(diff_, tensor({scale}, {1}));
    
    return {grad_input};
}

