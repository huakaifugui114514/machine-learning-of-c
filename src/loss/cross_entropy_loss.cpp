#include "loss/cross_entropy_loss.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::loss;

TensorPtr CrossEntropyLoss::forward(const TensorPtr& input, const TensorPtr& target) {
    // 检查形状 (batch_size, num_classes)
    if (input->shape().size() != 2 || target->shape().size() != 2) {
        throw std::invalid_argument("Both input and target must be 2D tensors");
    }
    
    const int batch_size = input->shape()[0];
    const int num_classes = input->shape()[1];
    
    input_ = input;
    target_ = target;
    
    const auto& input_data = input->data();
    const auto& target_data = target->data();
    
    float total_loss = 0.0f;
    softmax_output_ = std::make_shared<Tensor>(std::vector<float>(input->size()), input->shape(), false);
    
    for (int i = 0; i < batch_size; ++i) {
        // 找到最大值提高数值稳定性
        float max_val = *std::max_element(input_data.begin() + i * num_classes, 
                                         input_data.begin() + (i+1) * num_classes);
        
        float exp_sum = 0.0f;
        std::vector<float> logits(num_classes);
        
        // 计算softmax
        for (int j = 0; j < num_classes; ++j) {
            logits[j] = std::exp(input_data[i * num_classes + j] - max_val);
            exp_sum += logits[j];
        }
        
        // 计算交叉熵
        for (int j = 0; j < num_classes; ++j) {
            float prob = logits[j] / exp_sum;
            softmax_output_->data()[i * num_classes + j] = prob;
            if (target_data[i * num_classes + j] > 0) {
                total_loss -= target_data[i * num_classes + j] * 
                             std::log(prob + 1e-7f);
            }
        }
    }
    
    return tensor({total_loss / batch_size}, {1});
}

std::vector<TensorPtr> CrossEntropyLoss::backward() {
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    const int batch_size = input_->shape()[0];
    const int num_classes = input_->shape()[1];
    
    std::vector<float> grad_data(input_->size());
    const auto& target_data = target_->data();
    const auto& softmax_data = softmax_output_->data();
    
    // 计算梯度
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            const int idx = i * num_classes + j;
            grad_data[idx] = (softmax_data[idx] - target_data[idx]) / batch_size;
        }
    }
    
    return {tensor(grad_data, input_->shape())};
}