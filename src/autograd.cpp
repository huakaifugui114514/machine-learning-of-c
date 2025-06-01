#include "autograd.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace dlt {

// AddFunction实现
TensorPtr AddFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    
    // 执行加法操作
    const auto& a_data = inputs[0]->data();
    const auto& b_data = inputs[1]->data();
    
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
    
    // 创建输出张量
    bool requires_grad = inputs[0]->requires_grad() || inputs[1]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    
    // 如果需要梯度，设置梯度函数
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}


std::vector<TensorPtr> AddFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    // dL/dx = dL/dy * dy/dx = dL/dy * 1
    if (inputs_[0]->requires_grad()) {
        grads[0] = grad_output;
    }
    
    // dL/dy = dL/dy * dy/dy = dL/dy * 1
    if (inputs_[1]->requires_grad()) {
        grads[1] = grad_output;
    }
    
    return grads;
}

// SubFunction实现
TensorPtr SubFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("SubtractFunction requires exactly two inputs");
    }
    inputs_ = inputs;
    
    // 获取输入张量数据
    const auto& a_data = inputs[0]->data();
    const auto& b_data = inputs[1]->data();
    
    // 检查形状是否匹配
    if (a_data.size() != b_data.size()) {
        throw std::invalid_argument("Tensor sizes must match for subtraction");
    }
    
    // 执行减法操作
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = a_data[i] - b_data[i];
    }
    
    // 创建输出张量
    bool requires_grad = inputs[0]->requires_grad() || inputs[1]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    
    // 设置梯度函数和子节点信息
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> SubFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    // 计算梯度: dL/dx = dL/dy * dy/dx = dL/dy * 1
    if (inputs_[0]->requires_grad()) {
        grads[0] = grad_output;  // 对应 a - b 中的 a，梯度为 +grad_output
    }
    
    // 计算梯度: dL/dy = dL/dy * dy/db = dL/dy * (-1)
    if (inputs_[1]->requires_grad()) {
        // 创建一个新的张量，元素为 -1.0 * grad_output
        std::vector<float> neg_grad_data(grad_output->data().size());
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_output_data.size(); ++i) {
            neg_grad_data[i] = -grad_output_data[i];
        }
        grads[1] = std::make_shared<Tensor>(neg_grad_data, grad_output->shape(), false);
    }
    
    return grads;
}

// MulFunction实现
TensorPtr MulFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MulFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    
    // 执行乘法操作
    const auto& a_data = inputs[0]->data();
    const auto& b_data = inputs[1]->data();
    
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
    
    // 创建输出张量
    bool requires_grad = inputs[0]->requires_grad() || inputs[1]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    
    // 如果需要梯度，设置梯度函数
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> MulFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    // dL/dx = dL/dy * dy/dx = dL/dy * y
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& b_data = inputs_[1]->data();
        const auto& grad_output_data = grad_output->data();
        
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] * b_data[i];
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    // dL/dy = dL/dy * dy/dy = dL/dy * x
    if (inputs_[1]->requires_grad()) {
        std::vector<float> grad_data(inputs_[1]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& b_data = inputs_[1]->data();
        const auto& grad_output_data = grad_output->data();
        
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] * a_data[i];
        }
        
        grads[1] = std::make_shared<Tensor>(grad_data, inputs_[1]->shape(), false);
    }
    
    return grads;
}

// MatMulFunction实现
TensorPtr MatMulFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatMulFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    
    // 执行矩阵乘法
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    
    const auto& a_data = a->data();
    const auto& b_data = b->data();
    
    // 检查矩阵维度是否兼容
    auto result_shape = ops::compute_matmul_result_shape(a->shape(), b->shape());
    if (result_shape.empty()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }
    
    int m = a->shape()[0];
    int n = a->shape()[1];
    int p = b->shape()[1];
    
    std::vector<float> result_data(m * p, 0.0f);
    
    // 执行矩阵乘法
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                result_data[i * p + j] += a_data[i * n + k] * b_data[k * p + j];
            }
        }
    }
    
    // 创建输出张量
    bool requires_grad = inputs[0]->requires_grad() || inputs[1]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, result_shape, requires_grad);
    
    // 如果需要梯度，设置梯度函数
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> MatMulFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    const auto& a = inputs_[0];
    const auto& b = inputs_[1];
    
    // dL/dA = dL/dY * B^T
    if (a->requires_grad()) {
        // 转置b
        int m = b->shape()[0];
        int n = b->shape()[1];
        std::vector<float> b_t_data(m * n);
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                b_t_data[j * m + i] = b->data()[i * n + j];
            }
        }
        
        auto b_t = std::make_shared<Tensor>(b_t_data, std::vector<int>{n, m}, false);
        
        // 计算grad_output和b_t的矩阵乘法
        grads[0] = ops::matmul(grad_output, b_t);
    }
    
    // dL/dB = A^T * dL/dY
    if (b->requires_grad()) {
        // 转置a
        int m = a->shape()[0];
        int n = a->shape()[1];
        std::vector<float> a_t_data(m * n);
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                a_t_data[j * m + i] = a->data()[i * n + j];
            }
        }
        
        auto a_t = std::make_shared<Tensor>(a_t_data, std::vector<int>{n, m}, false);
        
        // 计算a_t和grad_output的矩阵乘法
        grads[1] =ops::matmul(a_t, grad_output);
    }
    
    return grads;
}

// ReLUFunction实现
TensorPtr ReLUFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReLUFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    
    // 执行ReLU操作
    const auto& a_data = inputs[0]->data();
    
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::max(0.0f, a_data[i]);
    }
    
    // 创建输出张量
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    
    // 如果需要梯度，设置梯度函数
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> ReLUFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] * (a_data[i] > 0 ? 1.0f : 0.0f);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

// SumFunction实现
TensorPtr SumFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SumFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    
    // 执行求和操作
    const auto& a_data = inputs[0]->data();
    
    float sum = 0.0f;
    for (float val : a_data) {
        sum += val;
    }
    
    // 创建输出张量
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(std::vector<float>{sum}, std::vector<int>{1}, requires_grad);
    
    // 如果需要梯度，设置梯度函数
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> SumFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size(), grad_output->data()[0]);
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

// ExpFunction实现
TensorPtr ExpFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ExpFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::exp(a_data[i]);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> ExpFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] * std::exp(a_data[i]);
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// LogFunction实现
TensorPtr LogFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("LogFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::log(a_data[i]);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> LogFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] / a_data[i];
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// SinFunction实现
TensorPtr SinFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SinFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::sin(a_data[i]);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> SinFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] * std::cos(a_data[i]);
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// CosFunction实现
TensorPtr CosFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("CosFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::cos(a_data[i]);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> CosFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = -grad_output_data[i] * std::sin(a_data[i]);
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// TanFunction实现
TensorPtr TanFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("TanFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::tan(a_data[i]);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> TanFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] / (std::cos(a_data[i]) * std::cos(a_data[i]));
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// MaxFunction实现
TensorPtr MaxFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MaxFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    float max_val = *std::max_element(a_data.begin(), a_data.end());
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(std::vector<float>{max_val}, std::vector<int>{1}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> MaxFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        auto max_it = std::max_element(a_data.begin(), a_data.end());
        size_t max_index = std::distance(a_data.begin(), max_it);
        grad_data[max_index] = grad_output_data[0];
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// MinFunction实现
TensorPtr MinFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MinFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    float min_val = *std::min_element(a_data.begin(), a_data.end());
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(std::vector<float>{min_val}, std::vector<int>{1}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> MinFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        auto min_it = std::min_element(a_data.begin(), a_data.end());
        size_t min_index = std::distance(a_data.begin(), min_it);
        grad_data[min_index] = grad_output_data[0];
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// MaxDimFunction实现
TensorPtr MaxDimFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MaxDimFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a = inputs[0];
    if (a->shape().size() != 2) {
        throw std::invalid_argument("Only 2D tensors are supported for now");
    }
    std::vector<float> result;
    if (dim_ == 0) {
        for (int j = 0; j < a->shape()[1]; ++j) {
            float max_val = a->data()[j];
            for (int i = 1; i < a->shape()[0]; ++i) {
                max_val = std::max(max_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(max_val);
        }
    } else if (dim_ == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            float max_val = a->data()[i * a->shape()[1]];
            for (int j = 1; j < a->shape()[1]; ++j) {
                max_val = std::max(max_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(max_val);
        }
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result, std::vector<int>{static_cast<int>(result.size())}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> MaxDimFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& a = inputs_[0];
        std::vector<float> grad_data(a->size(), 0.0f);
        const auto& grad_output_data = grad_output->data();
        if (dim_ == 0) {
            for (int j = 0; j < a->shape()[1]; ++j) {
                float max_val = a->data()[j];
                int max_index = 0;
                for (int i = 1; i < a->shape()[0]; ++i) {
                    if (a->data()[i * a->shape()[1] + j] > max_val) {
                        max_val = a->data()[i * a->shape()[1] + j];
                        max_index = i;
                    }
                }
                grad_data[max_index * a->shape()[1] + j] = grad_output_data[j];
            }
        } else if (dim_ == 1) {
            for (int i = 0; i < a->shape()[0]; ++i) {
                float max_val = a->data()[i * a->shape()[1]];
                int max_index = 0;
                for (int j = 1; j < a->shape()[1]; ++j) {
                    if (a->data()[i * a->shape()[1] + j] > max_val) {
                        max_val = a->data()[i * a->shape()[1] + j];
                        max_index = j;
                    }
                }
                grad_data[i * a->shape()[1] + max_index] = grad_output_data[i];
            }
        }
        grads[0] = std::make_shared<Tensor>(grad_data, a->shape(), false);
    }
    return grads;
}

// MinDimFunction实现
TensorPtr MinDimFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MinDimFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a = inputs[0];
    if (a->shape().size() != 2) {
        throw std::invalid_argument("Only 2D tensors are supported for now");
    }
    std::vector<float> result;
    if (dim_ == 0) {
        for (int j = 0; j < a->shape()[1]; ++j) {
            float min_val = a->data()[j];
            for (int i = 1; i < a->shape()[0]; ++i) {
                min_val = std::min(min_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(min_val);
        }
    } else if (dim_ == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            float min_val = a->data()[i * a->shape()[1]];
            for (int j = 1; j < a->shape()[1]; ++j) {
                min_val = std::min(min_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(min_val);
        }
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result, std::vector<int>{static_cast<int>(result.size())}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> MinDimFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& a = inputs_[0];
        std::vector<float> grad_data(a->size(), 0.0f);
        const auto& grad_output_data = grad_output->data();
        if (dim_ == 0) {
            for (int j = 0; j < a->shape()[1]; ++j) {
                float min_val = a->data()[j];
                int min_index = 0;
                for (int i = 1; i < a->shape()[0]; ++i) {
                    if (a->data()[i * a->shape()[1] + j] < min_val) {
                        min_val = a->data()[i * a->shape()[1] + j];
                        min_index = i;
                    }
                }
                grad_data[min_index * a->shape()[1] + j] = grad_output_data[j];
            }
        } else if (dim_ == 1) {
            for (int i = 0; i < a->shape()[0]; ++i) {
                float min_val = a->data()[i * a->shape()[1]];
                int min_index = 0;
                for (int j = 1; j < a->shape()[1]; ++j) {
                    if (a->data()[i * a->shape()[1] + j] < min_val) {
                        min_val = a->data()[i * a->shape()[1] + j];
                        min_index = j;
                    }
                }
                grad_data[i * a->shape()[1] + min_index] = grad_output_data[i];
            }
        }
        grads[0] = std::make_shared<Tensor>(grad_data, a->shape(), false);
    }
    return grads;
}

// MeanFunction实现
TensorPtr MeanFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MeanFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    float sum = std::accumulate(a_data.begin(), a_data.end(), 0.0f);
    float mean_val = sum / a_data.size();
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(std::vector<float>{mean_val}, std::vector<int>{1}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> MeanFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& grad_output_data = grad_output->data();
        float grad_val = grad_output_data[0] / inputs_[0]->size();
        std::fill(grad_data.begin(), grad_data.end(), grad_val);
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// MeanDimFunction实现
TensorPtr MeanDimFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MeanDimFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a = inputs[0];
    if (a->shape().size() != 2) {
        throw std::invalid_argument("Only 2D tensors are supported for now");
    }
    std::vector<float> result;
    if (dim_ == 0) {
        for (int j = 0; j < a->shape()[1]; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < a->shape()[0]; ++i) {
                sum += a->data()[i * a->shape()[1] + j];
            }
            result.push_back(sum / a->shape()[0]);
        }
    } else if (dim_ == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < a->shape()[1]; ++j) {
                sum += a->data()[i * a->shape()[1] + j];
            }
            result.push_back(sum / a->shape()[1]);
        }
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result, std::vector<int>{static_cast<int>(result.size())}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> MeanDimFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& a = inputs_[0];
        std::vector<float> grad_data(a->size());
        const auto& grad_output_data = grad_output->data();
        if (dim_ == 0) {
            for (int j = 0; j < a->shape()[1]; ++j) {
                float grad_val = grad_output_data[j] / a->shape()[0];
                for (int i = 0; i < a->shape()[0]; ++i) {
                    grad_data[i * a->shape()[1] + j] = grad_val;
                }
            }
        } else if (dim_ == 1) {
            for (int i = 0; i < a->shape()[0]; ++i) {
                float grad_val = grad_output_data[i] / a->shape()[1];
                for (int j = 0; j < a->shape()[1]; ++j) {
                    grad_data[i * a->shape()[1] + j] = grad_val;
                }
            }
        }
        grads[0] = std::make_shared<Tensor>(grad_data, a->shape(), false);
    }
    return grads;
}

// ReshapeFunction实现
TensorPtr ReshapeFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReshapeFunction requires exactly one input");
    }
    inputs_ = inputs;
    int new_size = 1;
    for (int dim : new_shape_) {
        new_size *= dim;
    }
    if (new_size != inputs[0]->size()) {
        throw std::invalid_argument("New shape does not match the size of the tensor");
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(inputs[0]->data(), new_shape_, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> ReshapeFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        grads[0] = std::make_shared<Tensor>(grad_output->data(), inputs_[0]->shape(), false);
    }
    return grads;
}

// TransposeFunction实现
TensorPtr TransposeFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("TransposeFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a = inputs[0];
    if (a->shape().size() != 2 || dim0_ < 0 || dim0_ > 1 || dim1_ < 0 || dim1_ > 1) {
        throw std::invalid_argument("Only 2D tensors are supported for transpose");
    }
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    if (dim0_ == 0 && dim1_ == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            for (int j = 0; j < a->shape()[1]; ++j) {
                result_data[j * a->shape()[0] + i] = data[i * a->shape()[1] + j];
            }
        }
    }
    std::vector<int> new_shape = a->shape();
    std::swap(new_shape[dim0_], new_shape[dim1_]);
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, new_shape, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> TransposeFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& a = inputs_[0];
        std::vector<float> grad_data(a->size());
        const auto& grad_output_data = grad_output->data();
        if (dim0_ == 0 && dim1_ == 1) {
            for (int i = 0; i < a->shape()[0]; ++i) {
                for (int j = 0; j < a->shape()[1]; ++j) {
                    grad_data[i * a->shape()[1] + j] = grad_output_data[j * a->shape()[0] + i];
                }
            }
        }
        grads[0] = std::make_shared<Tensor>(grad_data, a->shape(), false);
    }
    return grads;
}

// ConcatFunction实现
TensorPtr ConcatFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("No tensors to concatenate");
    }
    inputs_ = inputs;
    std::vector<int> output_shape = inputs[0]->shape();
    int total_size = 0;
    for (const auto& tensor : inputs) {
        if (tensor->shape().size() != output_shape.size()) {
            throw std::invalid_argument("Tensors must have the same number of dimensions");
        }
        for (size_t i = 0; i < output_shape.size(); ++i) {
            if (i != static_cast<size_t>(dim_)) {
                if (tensor->shape()[i] != output_shape[i]) {
                    throw std::invalid_argument("Tensors must have the same shape except for the concatenation dimension");
                }
            }
        }
        total_size += tensor->shape()[dim_];
    }
    output_shape[dim_] = total_size;
    std::vector<float> result_data;
    for (const auto& tensor : inputs) {
        result_data.insert(result_data.end(), tensor->data().begin(), tensor->data().end());
    }
    bool requires_grad = false;
    for (const auto& tensor : inputs) {
        requires_grad = requires_grad || tensor->requires_grad();
    }
    output_ = std::make_shared<Tensor>(result_data, output_shape, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> ConcatFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(inputs_.size());
    size_t start_index = 0;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (inputs_[i]->requires_grad()) {
            std::vector<float> grad_data(inputs_[i]->size());
            size_t end_index = start_index + inputs_[i]->size();
            std::copy(grad_output->data().begin() + start_index, grad_output->data().begin() + end_index, grad_data.begin());
            grads[i] = std::make_shared<Tensor>(grad_data, inputs_[i]->shape(), false);
        }
        start_index += inputs_[i]->size();
    }
    return grads;
}

// SplitFunction实现
TensorPtr SplitFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SplitFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a = inputs[0];
    if (a->shape()[dim_] % sections_ != 0) {
        throw std::invalid_argument("The dimension to split must be divisible by the number of sections");
    }
    std::vector<int> section_shape = a->shape();
    section_shape[dim_] = a->shape()[dim_] / sections_;
    int section_size = 1;
    for (int dim_size : section_shape) {
        section_size *= dim_size;
    }
    std::vector<float> result_data;
    for (int i = 0; i < sections_; ++i) {
        int start_index = i * section_size;
        result_data.insert(result_data.end(), a->data().begin() + start_index, a->data().begin() + start_index + section_size);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, section_shape, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> SplitFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& grad_output_data = grad_output->data();
        int section_size = grad_output->size();
        for (int i = 0; i < sections_; ++i) {
            int start_index = i * section_size;
            std::copy(grad_output_data.begin(), grad_output_data.end(), grad_data.begin() + start_index);
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// DotFunction实现
TensorPtr DotFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("DotFunction requires exactly two inputs");
    }
    inputs_ = inputs;
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (a->size() != b->size()) {
        throw std::invalid_argument("Tensors must have the same size for dot product");
    }
    const auto& data_a = a->data();
    const auto& data_b = b->data();
    float dot_product = 0.0f;
    for (size_t i = 0; i < data_a.size(); ++i) {
        dot_product += data_a[i] * data_b[i];
    }
    bool requires_grad = inputs[0]->requires_grad() || inputs[1]->requires_grad();
    output_ = std::make_shared<Tensor>(std::vector<float>{dot_product}, std::vector<int>{1}, requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> DotFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    const auto& a = inputs_[0];
    const auto& b = inputs_[1];
    const auto& grad_output_data = grad_output->data();
    if (a->requires_grad()) {
        std::vector<float> grad_data_a(a->size());
        const auto& data_b = b->data();
        for (size_t i = 0; i < grad_data_a.size(); ++i) {
            grad_data_a[i] = grad_output_data[0] * data_b[i];
        }
        grads[0] = std::make_shared<Tensor>(grad_data_a, a->shape(), false);
    }
    if (b->requires_grad()) {
        std::vector<float> grad_data_b(b->size());
        const auto& data_a = a->data();
        for (size_t i = 0; i < grad_data_b.size(); ++i) {
            grad_data_b[i] = grad_output_data[0] * data_a[i];
        }
        grads[1] = std::make_shared<Tensor>(grad_data_b, b->shape(), false);
    }
    return grads;
}

// AbsFunction实现
TensorPtr AbsFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("AbsFunction requires exactly one input");
    }
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::abs(a_data[i]);
    }
    bool requires_grad = inputs[0]->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, inputs[0]->shape(), requires_grad);
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    return output_;
}

std::vector<TensorPtr> AbsFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            if (a_data[i] > 0) {
                grad_data[i] = grad_output_data[i];
            } else if (a_data[i] < 0) {
                grad_data[i] = -grad_output_data[i];
            } else {
                grad_data[i] = 0.0f; // 导数在0处不定义，设为0
            }
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}



} // namespace dlt    