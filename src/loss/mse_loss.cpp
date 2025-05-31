#include "loss/mse_loss.hpp"
#include "ops.hpp"
#include <stdexcept>

namespace dlt {
namespace loss {

TensorPtr MSELoss::forward(const TensorPtr& input, const TensorPtr& target) {
    // 形状检查
    if (input->shape() != target->shape()) {
        throw std::invalid_argument("Input and target shapes must match");
    }

    input_ = input;
    target_ = target;
    
    // 计算 input - target
    auto neg_ones = tensor(std::vector<float>(target->size(), -1.0f), target->shape());
    diff_ = ops::add(input, ops::mul(target, neg_ones));    
    float n = static_cast<float>(input->size());
    auto squared = ops::mul(diff_ , diff_);    
    auto loss = squared->sum();
    auto ones = tensor(std::vector<float>(loss->size(), 1.0f/n), loss->shape());
    auto loss_final = ops::mul(loss , ones);  
    return loss_final;
}

std::vector<TensorPtr> MSELoss::backward() {
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    float n = static_cast<float>(input_->size());
    float scale = 2.0f / n;  
    
    // 计算梯度: 2*(input - target)/n
    auto grad_input = ops::mul(diff_, tensor(std::vector<float>(diff_->size(), scale), diff_->shape()));
    return {grad_input};
}

} // namespace loss
} // namespace dlt