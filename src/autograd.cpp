#include "autograd.hpp"
#include "ops.hpp"
#include "nn/pooling.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dlt {

// 辅助工具
inline size_t tensor_index(const std::vector<int>& shape, int b, int c, int h, int w) {
    return ((b * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

// 通用索引计算优化
inline size_t compute_index(const std::vector<int>& shape, const std::vector<int>& indices) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return index;
}

// 优化后的张量索引计算
inline size_t optimized_tensor_index(const std::vector<int>& shape, 
                                    int b, int c, int h, int w) {
    // 假设形状为 [B, C, H, W]
    return (b * shape[1] * shape[2] * shape[3]) + 
           (c * shape[2] * shape[3]) + 
           (h * shape[3]) + 
           w;
}

// 宏定义修复：使用_Pragma代替#pragma，并添加括号保护
#define VECTORIZED_LOOP(operation) \
    const auto& in_data = inputs[0]->data(); \
    std::vector<float> result_data(in_data.size()); \
    const size_t size = in_data.size(); \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < size; ++i) { \
        result_data[i] = (operation); \
    } \
    bool requires_grad = inputs[0]->requires_grad(); \
    auto output = std::make_shared<Tensor>( \
        std::move(result_data), \
        std::vector<int>(inputs[0]->shape()), \
        requires_grad \
    ); \
    if (requires_grad) { \
        output->grad_fn_ = shared_from_this(); \
        output->children_ = inputs; \
        output->is_leaf_ = false; \
    } \
    return output;


// AddFunction实现
TensorPtr AddFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    const auto& b_data = inputs[1]->data();
    const size_t size = a_data.size();
    
    std::vector<float> result_data(size);
    _Pragma("omp parallel for")
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
    
    bool requires_grad = inputs[0]->requires_grad() || inputs[1]->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        std::vector<int>(inputs[0]->shape()), 
        requires_grad
    );
    
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
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    const int m = a_shape[0];
    const int k = a_shape[1];
    const int n = b_shape[1];
    
    std::vector<float> result_data(m * n, 0.0f);
    const float* a_data = a->data_ptr();
    const float* b_data = b->data_ptr();
    
    _Pragma("omp parallel for collapse(2)")
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                sum += a_data[i * k + kk] * b_data[kk * n + j];
            }
            result_data[i * n + j] = sum;
        }
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        std::vector<int>{m, n}, 
        requires_grad
    );
    
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
        grads[1] = ops::matmul(a_t, grad_output);
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

    const auto& input = inputs[0];
    const auto& input_shape = input->shape();

    if (dim_ < 0 || dim_ >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for splitting");
    }
    if (input_shape[dim_] % sections_ != 0) {
        throw std::invalid_argument("The size of the specified dimension must be divisible by sections");
    }

    int split_size = input_shape[dim_] / sections_;
    std::vector<int> split_shape = input_shape;
    split_shape[dim_] = split_size;

    std::vector<TensorPtr> split_tensors;
    const auto& input_data = input->data();
    int total_size = input->size();
    int slice_size = total_size / sections_;

    for (int i = 0; i < sections_; ++i) {
        std::vector<float> split_data(slice_size);
        for (int j = 0; j < slice_size; ++j) {
            split_data[j] = input_data[i * slice_size + j];
        }
        TensorPtr split_tensor = std::make_shared<Tensor>(split_data, split_shape, input->requires_grad());
        split_tensors.push_back(split_tensor);

        if (input->requires_grad()) {
            split_tensor->grad_fn_ = shared_from_this();
            split_tensor->children_ = inputs;
            split_tensor->is_leaf_ = false;
        }
    }

    output_ = split_tensors[0];
    return output_;
}

std::vector<TensorPtr> SplitFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);

    if (inputs_[0]->requires_grad()) {
        const auto& input = inputs_[0];
        const auto& input_shape = input->shape();
        int split_size = input_shape[dim_] / sections_;
        std::vector<int> split_shape = input_shape;
        split_shape[dim_] = split_size;

        std::vector<float> grad_data(input->size());
        const auto& grad_output_data = grad_output->data();
        int slice_size = grad_output->size();

        for (int i = 0; i < sections_; ++i) {
            for (int j = 0; j < slice_size; ++j) {
                grad_data[i * slice_size + j] = grad_output_data[j];
            }
        }

        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
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

// ContiguousFunction 实现
TensorPtr ContiguousFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ContiguousFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    
    // 创建连续内存副本（实际实现中可能需要处理跨步访问）
    std::vector<float> contiguous_data(input->data());
    
    // 创建输出张量
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(std::move(contiguous_data), input->shape(), requires_grad);
    
    // 设置梯度函数
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> ContiguousFunction::backward(const TensorPtr& grad_output) {
    // 梯度直接传递回输入张量
    return { grad_output };
}

// ExpandFunction实现
ExpandFunction::ExpandFunction(const std::vector<int>& new_shape) : new_shape_(new_shape) {}

TensorPtr ExpandFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ExpandFunction requires exactly one input");
    }

    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    const auto& input_data = input->data();

    if (input_shape.size() > new_shape_.size()) {
        throw std::invalid_argument("New shape must have at least as many dimensions as the input");
    }

    std::vector<int> expanded_shape = new_shape_;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != 1 && input_shape[i] != new_shape_[i + new_shape_.size() - input_shape.size()]) {
            throw std::invalid_argument("Non-singleton dimensions must match");
        }
    }

    int expanded_size = 1;
    for (int dim : expanded_shape) {
        expanded_size *= dim;
    }

    std::vector<float> expanded_data(expanded_size);

    // 扩展数据
    for (int idx = 0; idx < expanded_size; ++idx) {
        std::vector<int> multi_idx(expanded_shape.size());
        int temp_idx = idx;
        for (int i = expanded_shape.size() - 1; i >= 0; --i) {
            multi_idx[i] = temp_idx % expanded_shape[i];
            temp_idx /= expanded_shape[i];
        }

        int input_idx = 0;
        int stride = 1;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            input_idx += multi_idx[i + expanded_shape.size() - input_shape.size()] * stride;
            stride *= input_shape[i];
        }

        expanded_data[idx] = input_data[input_idx];
    }

    auto output = std::make_shared<Tensor>(expanded_data, expanded_shape, input->requires_grad());
    if (output->requires_grad()) {
        output->grad_fn_ = shared_from_this();
        output->children_ = inputs;
        output->is_leaf_ = false;
    }
    return output;
}

std::vector<TensorPtr> ExpandFunction::backward(const TensorPtr& grad_output) {
    const auto& grad_output_shape = grad_output->shape();
    const auto& grad_output_data = grad_output->data();

    const auto& input = inputs_[0];
    const auto& input_shape = input->shape();

    std::vector<float> grad_input_data(input->size(), 0.0f);

    for (int idx = 0; idx < grad_output->size(); ++idx) {
        std::vector<int> multi_idx(grad_output_shape.size());
        int temp_idx = idx;
        for (int i = grad_output_shape.size() - 1; i >= 0; --i) {
            multi_idx[i] = temp_idx % grad_output_shape[i];
            temp_idx /= grad_output_shape[i];
        }

        int input_idx = 0;
        int stride = 1;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            input_idx += multi_idx[i + grad_output_shape.size() - input_shape.size()] * stride;
            stride *= input_shape[i];
        }

        grad_input_data[input_idx] += grad_output_data[idx];
    }

    auto grad_input = std::make_shared<Tensor>(grad_input_data, input_shape, false);
    return {grad_input};
}

// ReLUFunction实现
TensorPtr ReLUFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReLUFunction requires exactly one input");
    }
    
    if (!inputs[0]) {
        throw std::invalid_argument("ReLU input is null");
    }
    
    inputs_ = inputs;
    
    const auto& in_data = inputs[0]->data();
    std::vector<float> result_data(in_data.size());
    const size_t size = in_data.size();
    
    _Pragma("omp parallel for simd")
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max(0.0f, in_data[i]);
    }
    
    bool requires_grad = inputs[0]->requires_grad();
    auto output = std::make_shared<Tensor>(
        std::move(result_data), 
        std::vector<int>(inputs[0]->shape()), 
        requires_grad
    );
    
    if (requires_grad) {
        output->grad_fn_ = shared_from_this();
        output->children_ = inputs;
        output->is_leaf_ = false;
    }
    
    return output;
}

std::vector<TensorPtr> ReLUFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    // 检查输入是否有效
    if (inputs_.empty() || !inputs_[0]) {
        std::cerr << "ReLU backward error: inputs_[0] is null!" << std::endl;
        return grads;
    }
    
    if (!grad_output) {
        std::cerr << "ReLU backward error: grad_output is null!" << std::endl;
        return grads;
    }
    
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& a_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        // 检查大小是否匹配
        if (grad_data.size() != grad_output_data.size()) {
            std::cerr << "ReLU backward size mismatch: " 
                      << grad_data.size() << " vs " << grad_output_data.size()
                      << std::endl;
            return grads;
        }
        
        _Pragma("omp parallel for simd")
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_output_data[i] * (a_data[i] > 0 ? 1.0f : 0.0f);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

// SigmoidFunction实现
TensorPtr SigmoidFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SigmoidFunction requires exactly one input");
    }
    VECTORIZED_LOOP(1.0f / (1.0f + std::exp(-in_data[i])));
}

std::vector<TensorPtr> SigmoidFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& output_data = output_->data(); // 前向传播的输出
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        _Pragma("omp parallel for simd")
        for (size_t i = 0; i < grad_data.size(); ++i) {
            float s = output_data[i];
            grad_data[i] = grad_output_data[i] * s * (1 - s); // σ'(x) = σ(x)(1-σ(x))
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// TanhFunction实现
TensorPtr TanhFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("TanhFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& a_data = inputs[0]->data();
    std::vector<float> result_data(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        result_data[i] = std::tanh(a_data[i]);
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

std::vector<TensorPtr> TanhFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& output_data = output_->data(); // 前向传播的输出
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        for (size_t i = 0; i < grad_data.size(); ++i) {
            float t = output_data[i];
            grad_data[i] = grad_output_data[i] * (1 - t * t); // tanh'(x) = 1 - tanh²(x)
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    return grads;
}

// SoftmaxFunction实现
TensorPtr SoftmaxFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SoftmaxFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& shape = input->shape();
    
    int actual_dim = dim_ < 0 ? shape.size() + dim_ : dim_;
    if (actual_dim < 0 || actual_dim >= static_cast<int>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for softmax");
    }
    
    // 计算沿指定维度的元素数量
    int outer_size = 1;
    for (int i = 0; i < actual_dim; ++i) {
        outer_size *= shape[i];
    }
    
    int inner_size = 1;
    for (int i = actual_dim + 1; i < static_cast<int>(shape.size()); ++i) {
        inner_size *= shape[i];
    }
    
    int dim_size = shape[actual_dim];
    int step_size = inner_size * dim_size;
    int total_size = input->size();
    
    std::vector<float> result_data(total_size);
    const float* input_ptr = input_data.data();
    
    _Pragma("omp parallel for")
    for (int i = 0; i < outer_size; ++i) {
        for (int j = 0; j < inner_size; ++j) {
            const int base_idx = i * step_size + j;
            
            // 找到当前切片的最大值
            float max_val = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < dim_size; ++k) {
                int idx = base_idx + k * inner_size;
                if (input_ptr[idx] > max_val) {
                    max_val = input_ptr[idx];
                }
            }
            
            // 计算指数和
            float exp_sum = 0.0f;
            for (int k = 0; k < dim_size; ++k) {
                int idx = base_idx + k * inner_size;
                float exp_val = std::exp(input_ptr[idx] - max_val);
                result_data[idx] = exp_val;
                exp_sum += exp_val;
            }
            
            // 归一化
            for (int k = 0; k < dim_size; ++k) {
                int idx = base_idx + k * inner_size;
                result_data[idx] /= exp_sum;
            }
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(std::move(result_data), shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

std::vector<TensorPtr> SoftmaxFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& output_data = output_->data();  // softmax输出
        const auto& grad_output_data = grad_output->data();
        const auto& shape = inputs_[0]->shape();
        
        // 确定实际维度
        int actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = shape.size() + actual_dim;
        }
        
        // 计算维度参数
        int outer_size = 1;
        for (int i = 0; i < actual_dim; ++i) {
            outer_size *= shape[i];
        }
        
        int inner_size = 1;
        for (int i = actual_dim + 1; i < static_cast<int>(shape.size()); ++i) {
            inner_size *= shape[i];
        }
        
        int dim_size = shape[actual_dim];
        int step_size = inner_size * dim_size;
        
        // 计算梯度
        std::vector<float> grad_data(output_data.size(), 0.0f);
        
        for (int i = 0; i < outer_size; ++i) {
            for (int j = 0; j < inner_size; ++j) {
                // 计算雅可比矩阵的点积
                for (int k = 0; k < dim_size; ++k) {
                    int idx_k = i * step_size + k * inner_size + j;
                    float sum = 0.0f;
                    
                    for (int l = 0; l < dim_size; ++l) {
                        int idx_l = i * step_size + l * inner_size + j;
                        float grad_val = grad_output_data[idx_l];
                        
                        // 雅可比矩阵项: ∂y_l/∂x_k
                        float jacobian = output_data[idx_l] * 
                                        ((k == l) ? (1.0f - output_data[idx_k]) 
                                                  : (-output_data[idx_k]));
                        
                        sum += grad_val * jacobian;
                    }
                    
                    grad_data[idx_k] = sum;
                }
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, shape, false);
    }
    return grads;
}

// nn操作
// AvgPool2dFunction实现
AvgPool2dFunction::AvgPool2dFunction(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Invalid pooling parameters");
    }
}
TensorPtr AvgPool2dFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("AvgPool2dFunction expects exactly one input");
    }
    const auto& x = inputs[0];
    const auto& shape = x->shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d expects 4D input");
    }
    
    input_shape_ = shape; // 保存输入形状
    
    const int batch_size = shape[0];
    const int in_channels = shape[1];
    const int in_height = shape[2];
    const int in_width = shape[3];

    const int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Output dimensions must be positive");
    }

    std::vector<float> output_data(batch_size * in_channels * out_height * out_width);
    const float* x_data = x->data_ptr();
    float* output_ptr = output_data.data();

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                const int h_start = h_out * stride_ - padding_;
                const int h_end = h_start + kernel_size_;
                const int h_low = std::max(0, h_start);
                const int h_high = std::min(in_height, h_end);

                for (int w_out = 0; w_out < out_width; ++w_out) {
                    const int w_start = w_out * stride_ - padding_;
                    const int w_end = w_start + kernel_size_;
                    const int w_low = std::max(0, w_start);
                    const int w_high = std::min(in_width, w_end);

                    float sum = 0.0f;
                    int count = 0;
                    
                    // 仅在有有效区域时计算
                    if (h_low < h_high && w_low < w_high) {
                        for (int h_in = h_low; h_in < h_high; ++h_in) {
                            for (int w_in = w_low; w_in < w_high; ++w_in) {
                                const size_t index = tensor_index(shape, b, c, h_in, w_in);
                                sum += x_data[index];
                                count++;
                            }
                        }
                    }
                    *output_ptr++ = (count > 0) ? (sum / count) : 0.0f;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    auto output = std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
    
    // 设置输入和输出张量
    inputs_ = inputs;
    output_ = output;
    
    return output;
}

std::vector<TensorPtr> AvgPool2dFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& output_shape = grad_output->shape();
        const int batch_size = output_shape[0];
        const int in_channels = output_shape[1];
        const int out_height = output_shape[2];
        const int out_width = output_shape[3];

        const int in_height = input_shape_[2];
        const int in_width = input_shape_[3];

        std::vector<float> grad_input_data(batch_size * in_channels * in_height * in_width, 0.0f);
        const float* grad_output_data = grad_output->data_ptr();

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h_out = 0; h_out < out_height; ++h_out) {
                    const int h_start = h_out * stride_ - padding_;
                    const int h_end = h_start + kernel_size_;
                    const int h_low = std::max(0, h_start);
                    const int h_high = std::min(in_height, h_end);

                    for (int w_out = 0; w_out < out_width; ++w_out) {
                        const int w_start = w_out * stride_ - padding_;
                        const int w_end = w_start + kernel_size_;
                        const int w_low = std::max(0, w_start);
                        const int w_high = std::min(in_width, w_end);

                        int count = (h_high - h_low) * (w_high - w_low);
                        float grad = grad_output_data[tensor_index(output_shape, b, c, h_out, w_out)] / count;

                        for (int h_in = h_low; h_in < h_high; ++h_in) {
                            for (int w_in = w_low; w_in < w_high; ++w_in) {
                                grad_input_data[tensor_index(input_shape_, b, c, h_in, w_in)] += grad;
                            }
                        }
                    }
                }
            }
        }

        grads[0] = std::make_shared<Tensor>(grad_input_data, input_shape_, grad_output->requires_grad());
    }
    return grads;
}

// MaxPool2dFunction
MaxPool2dFunction::MaxPool2dFunction(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Invalid pooling parameters");
    }
}
TensorPtr MaxPool2dFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MaxPool2dFunction expects exactly one input");
    }
    
    const auto& x = inputs[0];
    const auto& shape = x->shape();
    input_shape_ = shape;
    
    const int batch_size = shape[0];
    const int in_channels = shape[1];
    const int in_height = shape[2];
    const int in_width = shape[3];

    const int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    std::vector<float> output_data(batch_size * in_channels * out_height * out_width, 0.0f);
    argmax_indices_.resize(output_data.size(), -1);
    const float* x_data = x->data_ptr();
    
    // 预计算步长
    const int in_c_stride = in_height * in_width;
    const int in_b_stride = in_channels * in_c_stride;
    const int out_c_stride = out_height * out_width;
    const int out_b_stride = in_channels * out_c_stride;

    _Pragma("omp parallel for collapse(2)")
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                const int h_start = oh * stride_ - padding_;
                const int h_end = std::min(h_start + kernel_size_, in_height);
                const int h_low = std::max(0, h_start);
                
                for (int ow = 0; ow < out_width; ++ow) {
                    const int w_start = ow * stride_ - padding_;
                    const int w_end = std::min(w_start + kernel_size_, in_width);
                    const int w_low = std::max(0, w_start);
                    
                    const int out_idx = b * out_b_stride + c * out_c_stride + oh * out_width + ow;
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_index = -1;
                    const int in_base = b * in_b_stride + c * in_c_stride;
                    
                    for (int kh = h_low; kh < h_end; ++kh) {
                        for (int kw = w_low; kw < w_end; ++kw) {
                            const int in_idx = in_base + kh * in_width + kw;
                            if (x_data[in_idx] > max_val) {
                                max_val = x_data[in_idx];
                                max_index = in_idx;
                            }
                        }
                    }
                    
                    output_data[out_idx] = (max_val == -std::numeric_limits<float>::infinity()) ? 0.0f : max_val;
                    argmax_indices_[out_idx] = max_index;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    output_ = std::make_shared<Tensor>(std::move(output_data), output_shape, x->requires_grad());
    inputs_ = inputs;
    
    return output_;
}

std::vector<TensorPtr> MaxPool2dFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    if (inputs_[0]->requires_grad()) {
        const auto& output_shape = grad_output->shape();
        const int batch_size = output_shape[0];
        const int in_channels = output_shape[1];
        const int out_height = output_shape[2];
        const int out_width = output_shape[3];

        const int in_height = input_shape_[2];
        const int in_width = input_shape_[3];

        std::vector<float> grad_input_data(batch_size * in_channels * in_height * in_width, 0.0f);
        const float* grad_output_data = grad_output->data_ptr();

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < in_channels; ++c) {
                for (int h_out = 0; h_out < out_height; ++h_out) {
                    for (int w_out = 0; w_out < out_width; ++w_out) {
                        const size_t out_idx = ((b * in_channels + c) * out_height + h_out) * out_width + w_out;
                        const int max_index = argmax_indices_[out_idx];
                        if (max_index != -1) {
                            grad_input_data[max_index] += grad_output_data[out_idx];
                        }
                    }
                }
            }
        }

        grads[0] = std::make_shared<Tensor>(grad_input_data, input_shape_, grad_output->requires_grad());
    }
    return grads;
}

// Conv2dFunction实现
TensorPtr Conv2dFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Conv2dFunction requires exactly two inputs");
    }

    inputs_ = inputs;
    input_shape_ = inputs[0]->shape();

    const auto& x = inputs[0];
    const auto& weight = inputs[1];

    // 检查输入形状 (batch_size, channels, height, width)
    if (x->shape().size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input (batch, channels, height, width)");
    }

    const int batch_size = x->shape()[0];
    const int in_channels = x->shape()[1];
    const int in_height = x->shape()[2];
    const int in_width = x->shape()[3];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels (" + std::to_string(in_channels) + 
                                   ") don't match layer channels (" + std::to_string(in_channels_) + ")");
    }

    const int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Output dimensions must be positive. Calculated: " + 
                                   std::to_string(out_height) + "x" + std::to_string(out_width));
    }

    std::vector<float> output_data(batch_size * out_channels_ * out_height * out_width, 0.0f);
    const float* x_data = x->data_ptr();
    const float* weight_data = weight->data_ptr();

    // 预计算偏移量和步长
    const int in_c_stride = in_height * in_width;
    const int in_b_stride = in_channels_ * in_c_stride;
    const int out_c_stride = out_height * out_width;
    const int out_b_stride = out_channels_ * out_c_stride;
    const int weight_oc_stride = in_channels_ * kernel_size_ * kernel_size_;
    const int weight_ic_stride = kernel_size_ * kernel_size_;

    _Pragma("omp parallel for collapse(2)")
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                const int h_start = oh * stride_ - padding_;
                const int h_end = std::min(h_start + kernel_size_, in_height);
                const int h_low = std::max(0, h_start);

                for (int ow = 0; ow < out_width; ++ow) {
                    const int w_start = ow * stride_ - padding_;
                    const int w_end = std::min(w_start + kernel_size_, in_width);
                    const int w_low = std::max(0, w_start);

                    float sum = 0.0f;
                    const int out_idx = b * out_b_stride + oc * out_c_stride + oh * out_width + ow;

                    for (int ic = 0; ic < in_channels_; ++ic) {
                        const int in_base = b * in_b_stride + ic * in_c_stride;
                        const int weight_base = oc * weight_oc_stride + ic * weight_ic_stride;

                        for (int kh = h_low; kh < h_end; ++kh) {
                            const int in_h_idx = (kh - h_start) * in_width;
                            const int weight_h_idx = (kh - h_low) * kernel_size_;

                            for (int kw = w_low; kw < w_end; ++kw) {
                                const int in_idx = in_base + kh * in_width + kw;
                                const int weight_idx = weight_base + weight_h_idx + (kw - w_low);
                                sum += x_data[in_idx] * weight_data[weight_idx];
                            }
                        }
                    }
                    output_data[out_idx] = sum;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};
    output_ = std::make_shared<Tensor>(
        std::move(output_data), 
        output_shape,
        inputs[0]->requires_grad() || inputs[1]->requires_grad()
    );

    if (output_->requires_grad()) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }

    return output_;
}

std::vector<TensorPtr> Conv2dFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2, nullptr);
    const auto& x = inputs_[0];
    const auto& weight = inputs_[1];

    const int batch_size = input_shape_[0];
    const int in_height = input_shape_[2];
    const int in_width = input_shape_[3];
    const int out_height = grad_output->shape()[2];
    const int out_width = grad_output->shape()[3];

    // 预计算步长
    const int in_c_stride = in_height * in_width;
    const int in_b_stride = in_channels_ * in_c_stride;
    const int out_c_stride = out_height * out_width;
    const int out_b_stride = out_channels_ * out_c_stride;
    const int weight_oc_stride = in_channels_ * kernel_size_ * kernel_size_;
    const int weight_ic_stride = kernel_size_ * kernel_size_;

    // grad_x
    if (x->requires_grad()) {
        std::vector<float> grad_x_data(batch_size * in_channels_ * in_height * in_width, 0.0f);
        const float* grad_output_data = grad_output->data_ptr();
        const float* weight_data = weight->data_ptr();

        _Pragma("omp parallel for collapse(2)")
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels_; ++oc) {
                for (int oh = 0; oh < out_height; ++oh) {
                    const int h_start = oh * stride_ - padding_;
                    const int h_end = std::min(h_start + kernel_size_, in_height);
                    const int h_low = std::max(0, h_start);
                    
                    for (int ow = 0; ow < out_width; ++ow) {
                        const int w_start = ow * stride_ - padding_;
                        const int w_end = std::min(w_start + kernel_size_, in_width);
                        const int w_low = std::max(0, w_start);
                        
                        const float grad_val = grad_output_data[
                            b * out_b_stride + oc * out_c_stride + oh * out_width + ow];
                        
                        for (int ic = 0; ic < in_channels_; ++ic) {
                            const int weight_base = oc * weight_oc_stride + ic * weight_ic_stride;
                            
                            for (int kh = h_low; kh < h_end; ++kh) {
                                const int weight_h_idx = (kh - h_low) * kernel_size_;
                                const int in_h_idx = kh * in_width;
                                
                                for (int kw = w_low; kw < w_end; ++kw) {
                                    const int weight_idx = weight_base + weight_h_idx + (kw - w_low);
                                    const int in_idx = b * in_b_stride + ic * in_c_stride + in_h_idx + kw;
                                    
                                    #pragma omp atomic
                                    grad_x_data[in_idx] += grad_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        grads[0] = std::make_shared<Tensor>(std::move(grad_x_data), input_shape_, false);
    }

    // grad_weight
    if (weight->requires_grad()) {
        std::vector<float> grad_weight_data(weight->size(), 0.0f);
        const float* grad_output_data = grad_output->data_ptr();
        const float* x_data = x->data_ptr();

        _Pragma("omp parallel for collapse(2)")
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int ic = 0; ic < in_channels_; ++ic) {
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        for (int b = 0; b < batch_size; ++b) {
                            for (int oh = 0; oh < out_height; ++oh) {
                                const int ih = oh * stride_ + kh - padding_;
                                if (ih < 0 || ih >= in_height) continue;
                                
                                for (int ow = 0; ow < out_width; ++ow) {
                                    const int iw = ow * stride_ + kw - padding_;
                                    if (iw < 0 || iw >= in_width) continue;
                                    
                                    const float grad_val = grad_output_data[
                                        b * out_b_stride + oc * out_c_stride + oh * out_width + ow];
                                    
                                    const float x_val = x_data[
                                        b * in_b_stride + ic * in_c_stride + ih * in_width + iw];
                                    
                                    const int weight_idx = oc * weight_oc_stride + 
                                                          ic * weight_ic_stride + 
                                                          kh * kernel_size_ + kw;
                                    
                                    #pragma omp atomic
                                    grad_weight_data[weight_idx] += grad_val * x_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        grads[1] = std::make_shared<Tensor>(std::move(grad_weight_data), weight->shape(), false);
    }

    return grads;
}

} // namespace dlt    