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

    input_ = input;
    target_ = target;

    // 计算 input - target
    auto neg_target = mul(target, tensor(std::vector<float>(target->size(), -1.0f), target->shape()));
    diff_ = add(input, neg_target);

    // 计算平方误差
    auto squared = mul(diff_, diff_);

    // 计算误差总和
    auto loss = sum(squared);

    // 计算平均误差
    float n = static_cast<float>(input->size());
    auto loss_final = mul(loss, tensor(std::vector<float>(loss->size(), 1.0f / n), loss->shape()));

    return loss_final;
}

std::vector<TensorPtr> MSELoss::backward() {
    // 检查是否已经调用了 forward 方法
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    // 计算梯度: 2*(input - target)/n
    float n = static_cast<float>(input_->size());
    float scale = 2.0f / n;
    auto grad_input = mul(diff_, tensor(std::vector<float>(diff_->size(), scale), diff_->shape()));

    return {grad_input};
}

