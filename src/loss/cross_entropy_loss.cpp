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
            if (target_data[i * num_classes + j] > 0) {
                total_loss -= target_data[i * num_classes + j] * 
                             std::log(logits[j] / (exp_sum + 1e-7f));
            }
        }
    }
    
    return tensor({total_loss / batch_size}, {1});
}

std::vector<TensorPtr> CrossEntropyLoss::backward() {
    if (!input_ || !target_) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    const float epsilon = 1e-7f;
    const int batch_size = input_->shape()[0];
    const int num_classes = input_->size() / batch_size;
    
    std::vector<float> grad_data(input_->size());
    const auto& input_data = input_->data();
    const auto& target_data = target_->data();
    
    // 计算梯度
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            const int idx = i * num_classes + j;
            float prob = std::clamp(input_data[idx], epsilon, 1.0f - epsilon);
            grad_data[idx] = (prob - target_data[idx]) / (prob * (1 - prob));
        }
    }
    
    // 归一化梯度
    for (float& val : grad_data) {
        val /= batch_size;
    }
    
    return {tensor(grad_data, input_->shape())};
}