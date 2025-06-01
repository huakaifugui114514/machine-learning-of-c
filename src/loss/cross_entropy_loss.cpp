#include "loss/cross_entropy_loss.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <cmath>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::loss;

TensorPtr CrossEntropyLoss::forward(const TensorPtr& input, const TensorPtr& target) {
    // 形状检查
    if (input->shape() != target->shape()) {
        throw std::invalid_argument("Input and target shapes must match");
    }

    input_ = input;
    target_ = target;
    float epsilon = 1e-7f;

    // 计算 -target * log(input)
    std::vector<float> log_input_data(input_->size());
    for (size_t i = 0; i < input_->size(); ++i) {
        if (input_->data()[i] == 0) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        log_input_data[i] = std::log(input_->data()[i]);
    }

    auto log_input = tensor(log_input_data, input_->shape());
    auto neg_log_input = mul(target, log_input);  // target * log(input)
    auto loss = ops::mul(neg_log_input, tensor(std::vector<float>(target->size(), -1.0f), neg_log_input->shape())); // -target * log(input)

    return sum(loss);
}

std::vector<TensorPtr> CrossEntropyLoss::backward() {
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    // 计算 input 的倒数
    std::vector<float> inv_input_data(input_->size());
    for (size_t i = 0; i < input_->size(); ++i) {
        if (input_->data()[i] == 0) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        inv_input_data[i] = 1.0f / input_->data()[i];
    }
    TensorPtr inv_input = tensor(inv_input_data, input_->shape());

    auto grad_input = mul(target_, inv_input);
    auto grad_input_scaled = mul(grad_input, tensor(std::vector<float>(target_->size(), -1.0f), grad_input->shape()));

    return {grad_input_scaled};
}

