#include "loss/nll_loss.hpp"
#include "ops.hpp"
#include <stdexcept>

namespace dlt {
namespace loss {

TensorPtr NLLLoss::forward(const TensorPtr& input, const TensorPtr& target) {
    // 形状检查
    if (input->shape()[0] != target->shape()[0]) {
        throw std::invalid_argument("Input and target batch sizes must match");
    }

    input_ = input;
    target_ = target;

    std::vector<float> loss_data;
    for (int i = 0; i < input->shape()[0]; ++i) {
        int target_index = static_cast<int>(target->data()[i]);
        loss_data.push_back(-input->data()[i * input->shape()[1] + target_index]);
    }

    auto loss_tensor = tensor(loss_data, {input->shape()[0]}, false);
    return ops::mean(loss_tensor);
}

std::vector<TensorPtr> NLLLoss::backward() {
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    std::vector<float> grad_input_data(input_->size(), 0.0f);
    int batch_size = input_->shape()[0];
    int num_classes = input_->shape()[1];

    for (int i = 0; i < batch_size; ++i) {
        int target_index = static_cast<int>(target_->data()[i]);
        grad_input_data[i * num_classes + target_index] = -1.0f / batch_size;
    }

    auto grad_input = tensor(grad_input_data, input_->shape(), false);
    return {grad_input};
}

} // namespace loss
} // namespace dlt