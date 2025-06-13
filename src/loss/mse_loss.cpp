// mse_loss.cpp
#include "loss/mse_loss.hpp"
#include "ops.hpp"
#include <stdexcept>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::loss;

std::shared_ptr<Tensor> MSELoss::forward(
    const std::shared_ptr<Tensor>& input, 
    const std::shared_ptr<Tensor>& target) {
    
    // 检查输入和目标张量
    if (!input || !target) {
        throw std::invalid_argument("Input and target tensors must not be null");
    }

    // 形状检查
    if (input->shape() != target->shape()) {
        throw std::invalid_argument("Input and target shapes must match");
    }

    // 保存输入用于反向传播
    input_ = input;
    target_ = target;
    
    // 计算差值和平方
    diff_ = sub(input, target);
    auto squared = mul(diff_, diff_);
    
    // 计算平均值
    float n = static_cast<float>(input->size());
    auto loss = sum(squared);
    return mul(loss, 1.0f / n);
}

std::vector<std::shared_ptr<Tensor>> MSELoss::backward() {
    // 检查是否已经调用了 forward 方法
    if (!diff_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    float scale = 2.0f / input_->size();
    auto grad_input = mul(diff_, scale);
    
    return {grad_input};
}