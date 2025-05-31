#include "autograd.hpp"
#include "ops.hpp"
#include <stdexcept>

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
        grads[0] = grad_output->matmul(b_t);
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
        grads[1] = a_t->matmul(grad_output);
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


} // namespace dlt    