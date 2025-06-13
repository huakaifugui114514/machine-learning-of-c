#include "autograd.hpp"
#include "ops.hpp"
#include "nn/pooling.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
# include <sstream>
#include <cfloat>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dlt {

// 内置函数

/**
 * 计算两个形状的广播结果形状（从最低维度开始比较）
 * 
 * @param shape_a 第一个张量的形状
 * @param shape_b 第二个张量的形状
 * @return 广播后的形状
 * @throws std::invalid_argument 如果形状不兼容
 */
// 替换原有的 broadcast_shapes 函数
inline std::vector<int> broadcast_shapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
    size_t max_dims = std::max(shape_a.size(), shape_b.size());
    std::vector<int> result(max_dims);
    
    int a_idx = shape_a.size() - 1;
    int b_idx = shape_b.size() - 1;
    
    for (int i = max_dims - 1; i >= 0; --i) {
        int dim_a = (a_idx >= 0) ? shape_a[a_idx--] : 1;
        int dim_b = (b_idx >= 0) ? shape_b[b_idx--] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            std::ostringstream oss;
            oss << "Broadcast dimension mismatch: dim[" << i << "] "
                << dim_a << " vs " << dim_b << "\n"
                << "  Shape A: [";
            for (int d : shape_a) oss << d << " ";
            oss << "]\n  Shape B: [";
            for (int d : shape_b) oss << d << " ";
            oss << "]";
            throw std::invalid_argument(oss.str());
        }
        
        result[i] = std::max(dim_a, dim_b);
    }
    return result;
}

// 广播计算函数
/**
 * 执行广播计算
 * 
 * @param a 第一个输入张量
 * @param b 第二个输入张量
 * @param result 输出结果数据
 * @param out_shape 广播后的形状
 * @param op 元素级操作函数
 */
inline void broadcast_compute(
    const TensorPtr& a, 
    const TensorPtr& b, 
    std::vector<float>& result,
    const std::vector<int>& out_shape,
    std::function<float(float, float)> op
) {
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    const auto& a_data = a->data();
    const auto& b_data = b->data();
    
    // 计算每个张量的步长(考虑广播)
    std::vector<size_t> a_strides(out_shape.size(), 0);
    std::vector<size_t> b_strides(out_shape.size(), 0);
    
    size_t a_stride = 1;
    size_t b_stride = 1;
    
    // 从最低维度开始计算
    for (int i = out_shape.size() - 1; i >= 0; --i) {
        int a_dim = (i >= out_shape.size() - a_shape.size()) ? 
                   a_shape[i - (out_shape.size() - a_shape.size())] : 1;
        int b_dim = (i >= out_shape.size() - b_shape.size()) ? 
                   b_shape[i - (out_shape.size() - b_shape.size())] : 1;
        
        a_strides[i] = (a_dim > 1) ? a_stride : 0;
        b_strides[i] = (b_dim > 1) ? b_stride : 0;
        
        a_stride *= (a_dim > 0) ? a_dim : 1;
        b_stride *= (b_dim > 0) ? b_dim : 1;
    }
    
    // 计算输出大小
    size_t total_size = 1;
    for (int dim : out_shape) {
        if (dim < 0) throw std::invalid_argument("Negative dimension in shape");
        total_size *= dim;
    }
    
    // 执行广播计算
    result.resize(total_size);
    
    #pragma omp parallel for
    for (size_t index = 0; index < total_size; ++index) {
        size_t a_index = 0;
        size_t b_index = 0;
        size_t remainder = index;
        
        // 从最高维度开始计算索引
        for (int i = 0; i < out_shape.size(); ++i) {
            int dim_size = out_shape[i];
            int coord = remainder % dim_size;
            remainder /= dim_size;
            
            // 计算输入索引
            if (a_strides[i] > 0) {
                a_index += (coord % a_shape[i]) * a_strides[i];
            }
            
            if (b_strides[i] > 0) {
                b_index += (coord % b_shape[i]) * b_strides[i];
            }
        }
        
        result[index] = op(a_data[a_index], b_data[b_index]);
    }
}

/**
 * 减少梯度到目标形状（用于广播操作的反向传播）
 * 
 * @param grad_output 上游梯度张量（广播后的形状）
 * @param target_shape 需要减少到的目标形状
 * @return 缩减后的梯度张量
 */
inline TensorPtr reduce_grad(const TensorPtr& grad_output, const std::vector<int>& target_shape) {
    // 确保张量是连续的
    auto cont_grad = dlt::ops::contiguous(grad_output);
    
    // 计算目标形状的总元素数
    int target_size = 1;
    for (int dim : target_shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
        target_size *= dim;
    }
    
    // 检查元素数量是否匹配
    if (cont_grad->size() != target_size) {
        // 尝试通过求和减少维度
        auto reduced_grad = cont_grad;
        int reduce_dim = -1;
        for (size_t i = 0; i < target_shape.size(); ++i) {
            if (reduced_grad->shape()[i] != target_shape[i]) {
                reduce_dim = i;
                break;
            }
        }
        
        if (reduce_dim != -1) {
            reduced_grad = dlt::ops::sum(reduced_grad, reduce_dim, true);
        }
        
        // 再次检查
        if (reduced_grad->size() != target_size) {
            std::ostringstream oss;
            oss << "Cannot reduce gradient from shape [";
            for (int dim : cont_grad->shape()) oss << dim << ", ";
            oss << "] to shape [";
            for (int dim : target_shape) oss << dim << ", ";
            oss << "]";
            throw std::runtime_error(oss.str());
        }
        cont_grad = reduced_grad;
    }
    
    // 重塑为目标形状
    return dlt::ops::reshape(cont_grad, target_shape);
}

// AddFunction实现
/**
 * 执行张量加法操作（支持广播）
 * 
 * @param inputs 输入张量列表（必须包含两个张量）
 * @return 加法结果张量
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr AddFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    auto& a = inputs[0];
    auto& b = inputs[1];
    
    // 计算广播形状
    auto out_shape = broadcast_shapes(a->shape(), b->shape());
    
    std::vector<float> result;
    broadcast_compute(a, b, result, out_shape, 
        [](float a_val, float b_val) { return a_val + b_val; }
    );
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    output_ = std::make_shared<Tensor>(result, out_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算加法操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 每个输入张量的梯度列表
 */
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
/**
 * 执行张量减法操作（支持广播）
 * 
 * @param inputs 输入张量列表（必须包含两个张量）
 * @return 减法结果张量
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr SubFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("SubtractFunction requires exactly two inputs");
    }
    inputs_ = inputs;
    auto& a = inputs[0];
    auto& b = inputs[1];
    
    // 计算广播形状
    auto out_shape = broadcast_shapes(a->shape(), b->shape());
    
    std::vector<float> result;
    broadcast_compute(a, b, result, out_shape, 
        [](float a_val, float b_val) { return a_val - b_val; }
    );
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    output_ = std::make_shared<Tensor>(result, out_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算减法操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 每个输入张量的梯度列表
 */
std::vector<TensorPtr> SubFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    // dL/dx = dL/dy * dy/dx = dL/dy * 1
    if (inputs_[0]->requires_grad()) {
        grads[0] = reduce_grad(grad_output, inputs_[0]->shape());
    }
    
    // dL/dy = dL/dy * dy/dy = dL/dy * (-1)
    if (inputs_[1]->requires_grad()) {
        // 创建一个负梯度张量
        std::vector<float> neg_grad_data(grad_output->size());
        const auto& grad_output_data = grad_output->data();
        for (size_t i = 0; i < neg_grad_data.size(); ++i) {
            neg_grad_data[i] = -grad_output_data[i];
        }
        auto neg_grad = std::make_shared<Tensor>(neg_grad_data, grad_output->shape(), false);
        grads[1] = reduce_grad(neg_grad, inputs_[1]->shape());
    }
    
    return grads;
}

// MulFunction实现
/**
 * 执行张量乘法操作（支持广播）
 * 
 * @param inputs 输入张量列表（必须包含两个张量）
 * @return 乘法结果张极
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr MulFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MulFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    auto& a = inputs[0];
    auto& b = inputs[1];
    
    // 计算广播形状
    auto out_shape = broadcast_shapes(a->shape(), b->shape());
    
    std::vector<float> result;
    broadcast_compute(a, b, result, out_shape, 
        [](float a_val, float b_val) { return a_val * b_val; }
    );
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    output_ = std::make_shared<Tensor>(result, out_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算乘法操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 每个输入张量的梯度列表
 */
std::vector<TensorPtr> MulFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    // dL/dx = dL/dy * y
    if (inputs_[0]->requires_grad()) {
        // 计算 grad_output * b
        auto tmp_grad = ops::mul(grad_output, inputs_[1]);
        grads[0] = reduce_grad(tmp_grad, inputs_[0]->shape());
    }
    
    // dL/dy = dL/dy * x
    if (inputs_[1]->requires_grad()) {
        // 计算 grad_output * a
        auto tmp_grad = ops::mul(grad_output, inputs_[0]);
        grads[1] = reduce_grad(tmp_grad, inputs_[1]->shape());
    }
    
    return grads;
}

// MatMulFunction实现（支持广播）
/**
 * 执行矩阵乘法操作（支持广播）
 * 
 * @param inputs 输入张量列表（必须包含两个张量）
 * @return 矩阵乘法结果张量
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr MatMulFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatMulFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    
    // 检查维度
    if (a_shape.size() < 2 || b_shape.size() < 2) {
        throw std::invalid_argument("Matrices must be at least 2-dimensional");
    }
    
    // 获取矩阵维度
    int m = a_shape[a_shape.size()-2];
    int k = a_shape[a_shape.size()-1];
    int n = b_shape[b_shape.size()-1];
    
    // 检查矩阵维度
    if (k != b_shape[b_shape.size()-2]) {
        std::ostringstream oss;
        oss << "Matrix dimensions incompatible: " 
            << "(" << m << "x" << k << ") vs (" 
            << b_shape[b_shape.size()-2] << "x" << n << ")";
        throw std::invalid_argument(oss.str());
    }
    
    // 计算广播形状
    std::vector<int> a_prefix_shape(a_shape.begin(), a_shape.end()-2);
    std::vector<int> b_prefix_shape(b_shape.begin(), b_shape.end()-2);
    std::vector<int> common_prefix = broadcast_shapes(a_prefix_shape, b_prefix_shape);
    
    // 计算输出形状
    std::vector<int> out_shape = common_prefix;
    out_shape.push_back(m);
    out_shape.push_back(n);
    
    // 计算元素总数
    size_t total_elements = 1;
    for (int dim : out_shape) {
        total_elements *= dim;
    }
    
    // 准备输出数据
    std::vector<float> result_data(total_elements, 0.0f);
    const float* a_data = a->data_ptr();
    const float* b_data = b->data_ptr();
    
    // 计算步长
    size_t a_matrix_size = m * k;
    size_t b_matrix_size = b_shape[b_shape.size()-2] * n;
    size_t out_matrix_size = m * n;
    
    // 计算广播维度
    std::vector<int> a_strides(common_prefix.size(), 0);
    std::vector<int> b_strides(common_prefix.size(), 0);
    
    // 计算每个输入在广播维度上的步长
    int a_stride = 1;
    int b_stride = 1;
    for (int i = common_prefix.size() - 1; i >= 0; --i) {
        a_strides[i] = (i >= (int)common_prefix.size() - (int)a_prefix_shape.size() && 
                       a_prefix_shape[i - (common_prefix.size() - a_prefix_shape.size())] > 1) ? a_stride : 0;
        b_strides[i] = (i >= (int)common_prefix.size() - (int)b_prefix_shape.size() && 
                       b_prefix_shape[i - (common_prefix.size() - b_prefix_shape.size())] > 1) ? b_stride : 0;
        
        a_stride *= (i >= (int)common_prefix.size() - (int)a_prefix_shape.size()) ? 
                   a_prefix_shape[i - (common_prefix.size() - a_prefix_shape.size())] : 1;
        b_stride *= (i >= (int)common_prefix.size() - (int)b_prefix_shape.size()) ? 
                   b_prefix_shape[i - (common_prefix.size() - b_prefix_shape.size())] : 1;
    }
    
    // 计算每个广播索引的矩阵乘法
    #pragma omp parallel for
    for (size_t prefix_idx = 0; prefix_idx < total_elements / out_matrix_size; ++prefix_idx) {
        // 计算广播索引
        std::vector<int> indices(common_prefix.size());
        size_t temp = prefix_idx;
        for (int i = common_prefix.size() - 1; i >= 0; --i) {
            indices[i] = temp % common_prefix[i];
            temp /= common_prefix[i];
        }
        
        // 计算输入矩阵的偏移量
        size_t a_offset = 0;
        size_t b_offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            a_offset += a_strides[i] * indices[i];
            b_offset += b_strides[i] * indices[i];
        }
        
        // 执行矩阵乘法
        const float* a_matrix = a_data + a_offset * a_matrix_size;
        const float* b_matrix = b_data + b_offset * b_matrix_size;
        float* out_matrix = result_data.data() + prefix_idx * out_matrix_size;
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; ++kk) {
                    sum += a_matrix[i * k + kk] * b_matrix[kk * n + j];
                }
                out_matrix[i * n + j] = sum;
            }
        }
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        out_shape,
        requires_grad
    );
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算矩阵乘法的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 每个输入张量的梯度列表
 */
std::vector<TensorPtr> MatMulFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    const auto& a = inputs_[0];
    const auto& b = inputs_[1];
    
    const auto& grad_output_shape = grad_output->shape();
    const auto& grad_output_data = grad_output->data_ptr();
    
    // 获取矩阵维度
    int m = a->shape()[a->shape().size()-2];
    int k = a->shape()[a->shape().size()-1];
    int n = b->shape()[b->shape().size()-1];
    
    // 计算广播形状
    std::vector<int> a_prefix_shape(a->shape().begin(), a->shape().end()-2);
    std::vector<int> b_prefix_shape(b->shape().begin(), b->shape().end()-2);
    std::vector<int> common_prefix = broadcast_shapes(a_prefix_shape, b_prefix_shape);
    
    // grad_a = grad_output * b^T
    if (a->requires_grad()) {
        // 计算输出形状
        std::vector<int> grad_a_shape = common_prefix;
        grad_a_shape.push_back(m);
        grad_a_shape.push_back(k);
        size_t grad_a_size = 1;
        for (int dim : grad_a_shape) grad_a_size *= dim;
        
        std::vector<float> grad_a_data(grad_a_size, 0.0f);
        
        // 计算步长
        size_t b_matrix_size = b->shape()[b->shape().size()-2] * b->shape()[b->shape().size()-1];
        size_t grad_matrix_size = m * n;
        size_t grad_a_matrix_size = m * k;
        
        // 计算广播维度步长
        std::vector<int> a_strides(common_prefix.size(), 0);
        std::vector<int> b_strides(common_prefix.size(), 0);
        std::vector<int> grad_strides(common_prefix.size(), 0);
        
        int a_stride = 1;
        int b_stride = 1;
        int grad_stride = 1;
        for (int i = common_prefix.size() - 1; i >= 0; --i) {
            a_strides[i] = (i >= (int)common_prefix.size() - (int)a_prefix_shape.size() && 
                          a_prefix_shape[i - (common_prefix.size() - a_prefix_shape.size())] > 1) ? a_stride : 0;
            b_strides[i] = (i >= (int)common_prefix.size() - (int)b_prefix_shape.size() && 
                          b_prefix_shape[i - (common_prefix.size() - b_prefix_shape.size())] > 1) ? b_stride : 0;
            grad_strides[i] = grad_stride;
            
            a_stride *= (i >= (int)common_prefix.size() - (int)a_prefix_shape.size()) ? 
                      a_prefix_shape[i - (common_prefix.size() - a_prefix_shape.size())] : 1;
            b_stride *= (i >= (int)common_prefix.size() - (int)b_prefix_shape.size()) ? 
                      b_prefix_shape[i - (common_prefix.size() - b_prefix_shape.size())] : 1;
            grad_stride *= common_prefix[i];
        }
        
        // 对每个广播索引
        #pragma omp parallel for
        for (size_t prefix_idx = 0; prefix_idx < grad_a_size / grad_a_matrix_size; ++prefix_idx) {
            // 计算广播索引
            std::vector<int> indices(common_prefix.size());
            size_t temp = prefix_idx;
            for (int i = common_prefix.size() - 1; i >= 0; --i) {
                indices[i] = temp % common_prefix[i];
                temp /= common_prefix[i];
            }
            
            // 计算偏移量
            size_t a_offset = 0;
            size_t b_offset = 0;
            size_t grad_offset = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                a_offset += a_strides[i] * indices[i];
                b_offset += b_strides[i] * indices[i];
                grad_offset += grad_strides[i] * indices[i];
            }
            
            // 获取矩阵指针
            const float* grad_matrix = grad_output_data + grad_offset * grad_matrix_size;
            const float* b_matrix = b->data_ptr() + b_offset * b_matrix_size;
            float* grad_a_matrix = grad_a_data.data() + prefix_idx * grad_a_matrix_size;
            
            // 计算 grad_a = grad_output * b^T
            for (int i = 0; i < m; ++i) {
                for (int kk = 0; kk < k; ++kk) {
                    float sum = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        sum += grad_matrix[i * n + j] * b_matrix[kk * n + j];
                    }
                    grad_a_matrix[i * k + kk] += sum;
                }
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_a_data, grad_a_shape, false);
    }
    
    // grad_b = a^T * grad_output
    if (b->requires_grad()) {
        // 计算输出形状
        std::vector<int> grad_b_shape = common_prefix;
        grad_b_shape.push_back(b->shape()[b->shape().size()-2]);
        grad_b_shape.push_back(n);
        size_t grad_b_size = 1;
        for (int dim : grad_b_shape) grad_b_size *= dim;
        
        std::vector<float> grad_b_data(grad_b_size, 0.0f);
        
        // 计算步长
        size_t a_matrix_size = a->shape()[a->shape().size()-2] * a->shape()[a->shape().size()-1];
        size_t grad_matrix_size = m * n;
        size_t grad_b_matrix_size = b->shape()[b->shape().size()-2] * n;
        
        // 计算广播维度步长
        std::vector<int> a_strides(common_prefix.size(), 0);
        std::vector<int> b_strides(common_prefix.size(), 0);
        std::vector<int> grad_strides(common_prefix.size(), 0);
        
        int a_stride = 1;
        int b_stride = 1;
        int grad_stride = 1;
        for (int i = common_prefix.size() - 1; i >= 0; --i) {
            a_strides[i] = (i >= (int)common_prefix.size() - (int)a_prefix_shape.size() && 
                          a_prefix_shape[i - (common_prefix.size() - a_prefix_shape.size())] > 1) ? a_stride : 0;
            b_strides[i] = (i >= (int)common_prefix.size() - (int)b_prefix_shape.size() && 
                          b_prefix_shape[i - (common_prefix.size() - b_prefix_shape.size())] > 1) ? b_stride : 0;
            grad_strides[i] = grad_stride;
            
            a_stride *= (i >= (int)common_prefix.size() - (int)a_prefix_shape.size()) ? 
                      a_prefix_shape[i - (common_prefix.size() - a_prefix_shape.size())] : 1;
            b_stride *= (i >= (int)common_prefix.size() - (int)b_prefix_shape.size()) ? 
                      b_prefix_shape[i - (common_prefix.size() - b_prefix_shape.size())] : 1;
            grad_stride *= common_prefix[i];
        }
        
        // 对每个广播索引
        #pragma omp parallel for
        for (size_t prefix_idx = 0; prefix_idx < grad_b_size / grad_b_matrix_size; ++prefix_idx) {
            // 计算广播索引
            std::vector<int> indices(common_prefix.size());
            size_t temp = prefix_idx;
            for (int i = common_prefix.size() - 1; i >= 0; --i) {
                indices[i] = temp % common_prefix[i];
                temp /= common_prefix[i];
            }
            
            // 计算偏移量
            size_t a_offset = 0;
            size_t b_offset = 0;
            size_t grad_offset = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                a_offset += a_strides[i] * indices[i];
                b_offset += b_strides[i] * indices[i];
                grad_offset += grad_strides[i] * indices[i];
            }
            
            // 获取矩阵指针
            const float* grad_matrix = grad_output_data + grad_offset * grad_matrix_size;
            const float* a_matrix = a->data_ptr() + a_offset * a_matrix_size;
            float* grad_b_matrix = grad_b_data.data() + prefix_idx * grad_b_matrix_size;
            
            // 计算 grad_b = a^T * grad_output
            for (int kk = 0; kk < k; ++kk) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int i = 0; i < m; ++i) {
                        sum += a_matrix[i * k + kk] * grad_matrix[i * n + j];
                    }
                    grad_b_matrix[kk * n + j] += sum;
                }
            }
        }
        
        grads[1] = std::make_shared<Tensor>(grad_b_data, grad_b_shape, false);
    }
    
    return grads;
}

/**
 * 执行张量求和操作（支持沿指定维度求和和广播）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 求和结果张量
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr SumFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SumFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    
    // 处理全局求和（默认行为）
    if (dim_ < 0) {
        float sum = 0.0f;
        const auto& data = input->data();
        for (float val : data) {
            sum += val;
        }
        
        bool requires_grad = input->requires_grad();
        output_ = std::make_shared<Tensor>(std::vector<float>{sum}, std::vector<int>{1}, requires_grad);
        
        if (requires_grad) {
            output_->grad_fn_ = shared_from_this();
            output_->children_ = inputs;
            output_->is_leaf_ = false;
        }
        
        return output_;
    }
    
    // 检查维度有效性
    int actual_dim = dim_;
    if (actual_dim < 0) {
        actual_dim = input_shape.size() + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for summation");
    }
    
    // 计算输出形状
    std::vector<int> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i != static_cast<size_t>(actual_dim) || keepdims_) {
            output_shape.push_back(i == static_cast<size_t>(actual_dim) ? 1 : input_shape[i]);
        }
    }
    
    // 计算沿维度求和
    size_t outer_size = 1;
    for (int i = 0; i < actual_dim; ++i) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }
    
    int dim_size = input_shape[actual_dim];
    size_t total_size = outer_size * inner_size;
    
    std::vector<float> result_data(total_size, 0.0f);
    const float* input_data = input->data_ptr();
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim_size; ++k) {
                size_t index = (i * dim_size + k) * inner_size + j;
                sum += input_data[index];
            }
            result_data[i * inner_size + j] = sum;
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, output_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算求和操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> SumFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_shape = inputs_[0]->shape();
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        
        // 处理全局求和
        if (dim_ < 0) {
            float grad_val = grad_output->data()[0];
            std::fill(grad_data.begin(), grad_data.end(), grad_val);
            grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
            return grads;
        }
        
        // 获取实际维度
        int actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input_shape.size() + actual_dim;
        }
        
        // 计算维度参数
        size_t outer_size = 1;
        for (int i = 0; i < actual_dim; ++i) {
            outer_size *= input_shape[i];
        }
        
        size_t inner_size = 1;
        for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
            inner_size *= input_shape[i];
        }
        
        int dim_size = input_shape[actual_dim];
        const float* grad_output_data = grad_output->data_ptr();
        
        // 创建梯度张量
        #pragma omp parallel for
        for (size_t i = 0; i < outer_size; ++i) {
            for (int k = 0; k < dim_size; ++k) {
                for (size_t j = 0; j < inner_size; ++j) {
                    size_t input_index = (i * dim_size + k) * inner_size + j;
                    size_t grad_index = i * inner_size + j;
                    grad_data[input_index] = grad_output_data[grad_index];
                }
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
    }
    
    return grads;
}

/**
 * 执行张量指数操作（e^x，逐元素）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 指数计算结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr ExpFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ExpFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算指数值
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::exp(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        input_shape,
        requires_grad
    );
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算指数操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> ExpFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_data = inputs_[0]->data();
        const auto& output_data = output_->data(); // 前向传播的输出
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            // 梯度计算：grad_input = grad_output * exp(input) = grad_output * output
            grad_data[i] = grad_output_data[i] * output_data[i];
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量自然对数操作（ln(x)，逐元素）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 对数计算结果张量
 * @throws std::invalid_argument 如果输入数量不正确或包含非正值
 */
TensorPtr LogFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("LogFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 检查输入值有效性
    for (float val : input_data) {
        if (val <= 0) {
            throw std::invalid_argument("Logarithm input must be positive");
        }
    }
    
    // 计算自然对数
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::log(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        input_shape,
        requires_grad
    );
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算对数操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> LogFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            // 梯度计算：grad_input = grad_output / x
            grad_data[i] = grad_output_data[i] / input_data[i];
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量正弦操作（sin(x)，逐元素）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 正弦计算结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr SinFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SinFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算正弦值
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::sin(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        input_shape,
        requires_grad
    );
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算正弦操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> SinFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            // 梯度计算：grad_input = grad_output * cos(x)
            grad_data[i] = grad_output_data[i] * std::cos(input_data[i]);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量余弦操作（cos(x)，逐元素）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 余弦计算结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr CosFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("CosFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算余弦值
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::cos(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        input_shape,
        requires_grad
    );
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算余弦操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> CosFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            // 梯度计算：grad_input = grad_output * (-sin(x))
            grad_data[i] = -grad_output_data[i] * std::sin(input_data[i]);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量正切操作（tan(x)，逐元素）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 正切计算结果张量
 * @throws std::invalid_argument 如果输入数量不正确或包含无效值
 */
TensorPtr TanFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("TanFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算正切值
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        // 检查输入值是否会导致无限大结果
        if (std::abs(std::cos(input_data[i])) < 1e-8) {
            throw std::invalid_argument("Tangent input causes undefined behavior");
        }
        result_data[i] = std::tan(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(
        std::move(result_data), 
        input_shape,
        requires_grad
    );
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算正切操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> TanFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& output_data = output_->data(); // 前向传播的输出（tan(x)）
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            // 梯度计算：grad_input = grad_output * (1 + tan²(x))
            //           = grad_output / cos²(x)
            float sec_sq = 1 + output_data[i] * output_data[i];
            grad_data[i] = grad_output_data[i] * sec_sq;
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

// MaxFunction实现
/**
 * 执行张量最大值操作（支持全局和沿指定维度）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 最大值结果张量
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr MaxFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MaxFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 处理全局最大值（默认行为）
    if (dim_ < 0) {
        float max_val = *std::max_element(input_data.begin(), input_data.end());
        bool requires_grad = input->requires_grad();
        output_ = std::make_shared<Tensor>(std::vector<float>{max_val}, std::vector<int>{1}, requires_grad);
        
        if (requires_grad) {
            // 找到最大值索引
            auto max_it = std::max_element(input_data.begin(), input_data.end());
            size_t max_index = std::distance(input_data.begin(), max_it);
            argmax_indices_ = {max_index};
            
            output_->grad_fn_ = shared_from_this();
            output_->children_ = inputs;
            output_->is_leaf_ = false;
        }
        
        return output_;
    }
    
    // 检查维度有效性
    int actual_dim = dim_;
    if (actual_dim < 0) {
        actual_dim = input_shape.size() + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for max operation");
    }
    
    // 计算输出形状
    std::vector<int> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i != static_cast<size_t>(actual_dim) || keepdims_) {
            output_shape.push_back(i == static_cast<size_t>(actual_dim) ? 1 : input_shape[i]);
        }
    }
    
    // 如果没有维度被保留，确保至少有一个维度
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }
    
    // 计算沿维度最大值
    size_t outer_size = 1;
    for (int i = 0; i < actual_dim; ++i) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }
    
    int dim_size = input_shape[actual_dim];
    size_t total_size = outer_size * inner_size;
    
    std::vector<float> result_data(total_size, std::numeric_limits<float>::lowest());
    argmax_indices_.resize(total_size);
    const float* input_ptr = input->data_ptr();
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = (i * dim_size) * inner_size + j;
            float max_val = input_ptr[base_index];
            size_t max_index = 0;
            
            for (int k = 1; k < dim_size; ++k) {
                size_t index = base_index + k * inner_size;
                if (input_ptr[index] > max_val) {
                    max_val = input_ptr[index];
                    max_index = k;
                }
            }
            
            result_data[i * inner_size + j] = max_val;
            argmax_indices_[i * inner_size + j] = (i * dim_size + max_index) * inner_size + j;
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, output_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算最大值操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> MaxFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_shape = inputs_[0]->shape();
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        const float* grad_output_ptr = grad_output->data_ptr();
        
        // 处理全局最大值
        if (dim_ < 0) {
            grad_data[argmax_indices_[0]] = grad_output_ptr[0];
            grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
            return grads;
        }
        
        // 获取实际维度
        int actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input_shape.size() + actual_dim;
        }
        
        // 计算维度参数
        size_t outer_size = 1;
        for (int i = 0; i < actual_dim; ++i) {
            outer_size *= input_shape[i];
        }
        
        size_t inner_size = 1;
        for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
            inner_size *= input_shape[i];
        }
        
        int dim_size = input_shape[actual_dim];
        size_t total_size = outer_size * inner_size;
        
        // 创建梯度张量
        for (size_t i = 0; i < total_size; ++i) {
            grad_data[argmax_indices_[i]] = grad_output_ptr[i];
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
    }
    
    return grads;
}

// MinFunction实现
/**
 * 执行张量最小值操作（支持全局和沿指定维度）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 最小值结果张量
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr MinFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MinFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 处理全局最小值（默认行为）
    if (dim_ < 0) {
        float min_val = *std::min_element(input_data.begin(), input_data.end());
        bool requires_grad = input->requires_grad();
        output_ = std::make_shared<Tensor>(std::vector<float>{min_val}, std::vector<int>{1}, requires_grad);
        
        if (requires_grad) {
            // 找到最小值索引
            auto min_it = std::min_element(input_data.begin(), input_data.end());
            size_t min_index = std::distance(input_data.begin(), min_it);
            argmin_indices_ = {min_index};
            
            output_->grad_fn_ = shared_from_this();
            output_->children_ = inputs;
            output_->is_leaf_ = false;
        }
        
        return output_;
    }
    
    // 检查维度有效性
    int actual_dim = dim_;
    if (actual_dim < 0) {
        actual_dim = input_shape.size() + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for min operation");
    }
    
    // 计算输出形状
    std::vector<int> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i != static_cast<size_t>(actual_dim) || keepdims_) {
            output_shape.push_back(i == static_cast<size_t>(actual_dim) ? 1 : input_shape[i]);
        }
    }
    
    // 如果没有维度被保留，确保至少有一个维度
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }
    
    // 计算沿维度最小值
    size_t outer_size = 1;
    for (int i = 0; i < actual_dim; ++i) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }
    
    int dim_size = input_shape[actual_dim];
    size_t total_size = outer_size * inner_size;
    
    std::vector<float> result_data(total_size, std::numeric_limits<float>::max());
    argmin_indices_.resize(total_size);
    const float* input_ptr = input->data_ptr();
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = (i * dim_size) * inner_size + j;
            float min_val = input_ptr[base_index];
            size_t min_index = 0;
            
            for (int k = 1; k < dim_size; ++k) {
                size_t index = base_index + k * inner_size;
                if (input_ptr[index] < min_val) {
                    min_val = input_ptr[index];
                    min_index = k;
                }
            }
            
            result_data[i * inner_size + j] = min_val;
            argmin_indices_[i * inner_size + j] = (i * dim_size + min_index) * inner_size + j;
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, output_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算最小值操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> MinFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_shape = inputs_[0]->shape();
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        const float* grad_output_ptr = grad_output->data_ptr();
        
        // 处理全局最小值
        if (dim_ < 0) {
            grad_data[argmin_indices_[0]] = grad_output_ptr[0];
            grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
            return grads;
        }
        
        // 获取实际维度
        int actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input_shape.size() + actual_dim;
        }
        
        // 计算维度参数
        size_t outer_size = 1;
        for (int i = 0; i < actual_dim; ++i) {
            outer_size *= input_shape[i];
        }
        
        size_t inner_size = 1;
        for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
            inner_size *= input_shape[i];
        }
        
        int dim_size = input_shape[actual_dim];
        size_t total_size = outer_size * inner_size;
        
        // 创建梯度张量
        for (size_t i = 0; i < total_size; ++i) {
            grad_data[argmin_indices_[i]] = grad_output_ptr[i];
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
    }
    
    return grads;
}

// MeanFunction实现
/**
 * 执行张量均值操作（支持全局和沿指定维度）
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 均值结果张量
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr MeanFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("MeanFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 处理全局均值（默认行为）
    if (dim_ < 0) {
        float sum = 0.0f;
        for (float val : input_data) {
            sum += val;
        }
        float mean_val = sum / input_data.size();
        bool requires_grad = input->requires_grad();
        output_ = std::make_shared<Tensor>(std::vector<float>{mean_val}, std::vector<int>{1}, requires_grad);
        
        if (requires_grad) {
            output_->grad_fn_ = shared_from_this();
            output_->children_ = inputs;
            output_->is_leaf_ = false;
        }
        return output_;
    }
    
    // 检查维度有效性
    int actual_dim = dim_;
    if (actual_dim < 0) {
        actual_dim = input_shape.size() + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for mean operation");
    }
    
    // 计算输出形状
    std::vector<int> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i != static_cast<size_t>(actual_dim) || keepdims_) {
            output_shape.push_back(i == static_cast<size_t>(actual_dim) ? 1 : input_shape[i]);
        }
    }
    
    // 如果没有维度被保留，确保至少有一个维度
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }
    
    // 计算沿维度均值
    size_t outer_size = 1;
    for (int i = 0; i < actual_dim; ++i) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }
    
    int dim_size = input_shape[actual_dim];
    size_t total_size = outer_size * inner_size;
    
    std::vector<float> result_data(total_size, 0.0f);
    const float* input_ptr = input->data_ptr();
    
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim_size; ++k) {
                size_t index = (i * dim_size + k) * inner_size + j;
                sum += input_ptr[index];
            }
            result_data[i * inner_size + j] = sum / dim_size;
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, output_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算均值操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> MeanFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_shape = inputs_[0]->shape();
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        
        // 处理全局均值
        if (dim_ < 0) {
            float grad_val = grad_output->data()[0] / inputs_[0]->size();
            std::fill(grad_data.begin(), grad_data.end(), grad_val);
            grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
            return grads;
        }
        
        // 获取实际维度
        int actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input_shape.size() + actual_dim;
        }
        
        // 计算维度参数
        size_t outer_size = 1;
        for (int i = 0; i < actual_dim; ++i) {
            outer_size *= input_shape[i];
        }
        
        size_t inner_size = 1;
        for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
            inner_size *= input_shape[i];
        }
        
        int dim_size = input_shape[actual_dim];
        const float* grad_output_data = grad_output->data_ptr();
        
        // 创建梯度张量
        #pragma omp parallel for
        for (size_t i = 0; i < outer_size; ++i) {
            for (int k = 0; k < dim_size; ++k) {
                for (size_t j = 0; j < inner_size; ++j) {
                    size_t input_index = (i * dim_size + k) * inner_size + j;
                    size_t grad_index = i * inner_size + j;
                    grad_data[input_index] = grad_output_data[grad_index] / dim_size;
                }
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
    }
    
    return grads;
}

// ReshapeFunction实现
/**
 * 执行张量形状变换操作
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 形状变换后的张量
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr ReshapeFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReshapeFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    
    // 检查新形状的元素总数是否匹配
    size_t new_size = 1;
    for (int dim : new_shape_) {
        new_size *= dim;
    }
    if (new_size != input->size()) {
        throw std::invalid_argument("Reshape size does not match the number of elements");
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(input->data(), new_shape_, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算形状变换操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> ReshapeFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        // 梯度直接变换回输入张量的形状
        const auto& input_shape = inputs_[0]->shape();
        auto grad_input = std::make_shared<Tensor>(grad_output->data(), input_shape, false);
        grads[0] = grad_input;
    }
    
    return grads;
}

/**
 * 执行张量转置操作
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 转置后的张量
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr TransposeFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("TransposeFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    const size_t ndim = input_shape.size();
    
    // 验证维度
    if (dim0_ < 0) dim0_ = ndim + dim0_;
    if (dim1_ < 0) dim1_ = ndim + dim1_;
    if (dim0_ < 0 || dim0_ >= static_cast<int>(ndim) || 
        dim1_ < 0 || dim1_ >= static_cast<int>(ndim)) {
        throw std::invalid_argument("Invalid dimensions for transpose");
    }
    
    // 计算新形状
    std::vector<int> new_shape = input_shape;
    std::swap(new_shape[dim0_], new_shape[dim1_]);
    
    // 计算转置后的数据
    const auto& input_data = input->data();
    std::vector<float> result_data(input_data.size());
    
    // 计算转置索引
    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * new_shape[i + 1];
    }
    
    std::vector<int> indices(ndim, 0);
    
    #pragma omp parallel for
    for (size_t i = 0; i < input_data.size(); ++i) {
        // 计算原始索引
        size_t remainder = i;
        for (int j = ndim - 1; j >= 0; --j) {
            indices[j] = remainder % input_shape[j];
            remainder /= input_shape[j];
        }
        
        // 交换维度
        std::swap(indices[dim0_], indices[dim1_]);
        
        // 计算新索引
        size_t new_index = 0;
        for (size_t j = 0; j < ndim; ++j) {
            new_index += indices[j] * strides[j];
        }
        
        result_data[new_index] = input_data[i];
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, new_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算转置操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> TransposeFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        // 反向转置操作
        auto grad_input = ops::transpose(grad_output, dim0_, dim1_);
        grads[0] = grad_input;
    }
    
    return grads;
}

/**
 * 执行张量连接操作
 * 
 * @param inputs 输入张量列表（至少包含一个张量）
 * @return 连接后的张量
 * @throws std::invalid_argument 如果输入为空或形状不兼容
 */
TensorPtr ConcatFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("ConcatFunction requires at least one input");
    }
    
    inputs_ = inputs;
    const auto& first_shape = inputs[0]->shape();
    const size_t ndim = first_shape.size();
    
    // 验证所有输入张量的维度数相同
    for (const auto& input : inputs) {
        if (input->shape().size() != ndim) {
            throw std::invalid_argument("All tensors must have the same number of dimensions");
        }
    }
    
    // 计算输出形状
    std::vector<int> output_shape = first_shape;
    int total_dim = 0;
    for (const auto& input : inputs) {
        total_dim += input->shape()[dim_];
    }
    output_shape[dim_] = total_dim;
    
    // 收集所有数据
    std::vector<float> result_data;
    for (const auto& input : inputs) {
        const auto& data = input->data();
        result_data.insert(result_data.end(), data.begin(), data.end());
    }
    
    // 检查是否需要梯度
    bool requires_grad = false;
    for (const auto& input : inputs) {
        requires_grad = requires_grad || input->requires_grad();
    }
    
    output_ = std::make_shared<Tensor>(result_data, output_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算连接操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 每个输入张量的梯度列表
 */
std::vector<TensorPtr> ConcatFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(inputs_.size());
    
    size_t start_index = 0;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (inputs_[i]->requires_grad()) {
            const auto& input_shape = inputs_[i]->shape();
            size_t num_elements = inputs_[i]->size();
            
            // 提取对应的梯度部分
            std::vector<float> grad_data(
                grad_output->data().begin() + start_index,
                grad_output->data().begin() + start_index + num_elements
            );
            
            grads[i] = std::make_shared<Tensor>(grad_data, input_shape, false);
        }
        start_index += inputs_[i]->size();
    }
    
    return grads;
}
/**
 * 执行张量切片操作
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 切片后的张量列表
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr SplitFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SplitFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    
    // 验证维度
    if (dim_ < 0) dim_ = input_shape.size() + dim_;
    if (dim_ < 0 || dim_ >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for splitting");
    }
    
    // 验证分割数
    if (input_shape[dim_] % sections_ != 0) {
        throw std::invalid_argument("Dimension size must be divisible by sections");
    }
    
    // 计算分割大小
    int split_size = input_shape[dim_] / sections_;
    
    // 计算输出形状
    std::vector<int> output_shape = input_shape;
    output_shape[dim_] = split_size;
    
    // 计算每个分割张量的元素数量
    size_t slice_num_elements = 1;
    for (int dim : output_shape) {
        slice_num_elements *= dim;
    }
    
    // 计算当前切片的起始位置
    slice_index_ = index_ * slice_num_elements;
    
    // 创建分割张量
    const float* input_data = input->data_ptr();
    std::vector<float> split_data(slice_num_elements);
    for (size_t i = 0; i < slice_num_elements; ++i) {
        split_data[i] = input_data[slice_index_ + i];
    }
    
    bool requires_grad = input->requires_grad();
    auto tensor = std::make_shared<Tensor>(split_data, output_shape, requires_grad);
    
    if (requires_grad) {
        tensor->grad_fn_ = shared_from_this();
        tensor->children_ = inputs;
        tensor->is_leaf_ = false;
    }
    
    return tensor; // 返回单个张量
}

std::vector<TensorPtr> SplitFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        // 创建一个与输入相同形状的梯度张量
        std::vector<float> grad_data(inputs_[0]->size(), 0.0f);
        const float* grad_output_data = grad_output->data_ptr();
        
        // 将当前分割张量的梯度放到对应的位置
        for (size_t i = 0; i < grad_output->size(); ++i) {
            grad_data[slice_index_ + i] = grad_output_data[i];
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量点积操作
 * 
 * @param inputs 输入张量列表（必须包含两个张量）
 * @return 点积结果张量（标量）
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr DotFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("DotFunction requires exactly two inputs");
    }
    
    inputs_ = inputs;
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    
    // 检查形状兼容性
    if (a->size() != b->size()) {
        throw std::invalid_argument("Tensors must have the same size for dot product");
    }
    
    // 计算点积
    float dot_product = 0.0f;
    const auto& a_data = a->data();
    const auto& b_data = b->data();
    for (size_t i = 0; i < a->size(); ++i) {
        dot_product += a_data[i] * b_data[i];
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    output_ = std::make_shared<Tensor>(std::vector<float>{dot_product}, std::vector<int>{1}, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算点积操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 每个输入张量的梯度列表
 */
std::vector<TensorPtr> DotFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(2);
    
    const float grad_val = grad_output->data()[0];
    
    if (inputs_[0]->requires_grad()) {
        std::vector<float> grad_data(inputs_[0]->size());
        const auto& b_data = inputs_[1]->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_val * b_data[i];
        }
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    if (inputs_[1]->requires_grad()) {
        std::vector<float> grad_data(inputs_[1]->size());
        const auto& a_data = inputs_[0]->data();
        for (size_t i = 0; i < grad_data.size(); ++i) {
            grad_data[i] = grad_val * a_data[i];
        }
        grads[1] = std::make_shared<Tensor>(grad_data, inputs_[1]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量绝对值操作
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 绝对值结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr AbsFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("AbsFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算绝对值
    std::vector<float> result_data(input_data.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        result_data[i] = std::abs(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, input_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算绝对值操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> AbsFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(input_data.size());
        for (size_t i = 0; i < grad_data.size(); ++i) {
            if (input_data[i] > 0) {
                grad_data[i] = grad_output_data[i];
            } else if (input_data[i] < 0) {
                grad_data[i] = -grad_output_data[i];
            } else {
                grad_data[i] = 0.0f; // 0处的导数未定义，设为0
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行张量连续内存操作
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 连续内存版本的张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr ContiguousFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ContiguousFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    
    // 创建连续内存副本
    std::vector<float> contiguous_data = input->data();
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(contiguous_data, input->shape(), requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算连续内存操作的梯度
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> ContiguousFunction::backward(const TensorPtr& grad_output) {
    // 梯度直接传递回输入张量
    return { grad_output };
}

/**
 * 执行张量扩展操作（支持广播）
 * 
 * 将输入张量扩展到指定的新形状，遵循广播规则：
 * 1. 从后向前比较维度
 * 2. 维度大小相同或输入维度大小为1时可广播
 * 3. 新形状的每个维度大小必须 ≥ 输入对应维度大小
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return 扩展后的张量
 * @throws std::invalid_argument 如果输入数量不正确或形状不兼容
 */
TensorPtr ExpandFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ExpandFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    const auto& input_data = input->data();
    
    // 直接使用广播形状计算
    try {
        // 计算广播形状
        auto out_shape = broadcast_shapes(input_shape, new_shape_);
        
        // 创建输出数据
        std::vector<float> output_data;
        const auto& dummy = tensor({0.0f}, {1}); // 用于广播计算的占位符
        
        // 使用广播计算函数
        broadcast_compute(
            input, dummy, output_data, out_shape,
            [](float a_val, float b_val) { return a_val; } // 只需返回输入值
        );
        
        bool requires_grad = input->requires_grad();
        output_ = std::make_shared<Tensor>(output_data, out_shape, requires_grad);
        
        if (requires_grad) {
            output_->grad_fn_ = shared_from_this();
            output_->children_ = inputs;
            output_->is_leaf_ = false;
        }
        
        return output_;
    } catch (const std::exception& e) {
        // 添加更具体的错误信息
        std::ostringstream oss;
        oss << "Expand failed: from shape [";
        for (int dim : input_shape) oss << dim << " ";
        oss << "] to [";
        for (int dim : new_shape_) oss << dim << " ";
        oss << "]: " << e.what();
        throw std::invalid_argument(oss.str());
    }
}

/**
 * 计算扩展操作的梯度
 * 
 * 将上游梯度缩减回输入张量的原始形状，遵循广播规则：
 * 1. 对广播维度上的梯度进行求和
 * 2. 非广播维度直接复制梯度
 * 
 * @param grad_output 上游梯度张量（扩展后的形状）
 * @return 输入张量的梯度（原始形状）
 */
std::vector<TensorPtr> ExpandFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input = inputs_[0];
        const auto& input_shape = input->shape();
        const auto& grad_output_data = grad_output->data();
        const auto& new_shape = grad_output->shape();
        
        // 对齐输入形状
        std::vector<int> aligned_input_shape = input_shape;
        while (aligned_input_shape.size() < new_shape.size()) {
            aligned_input_shape.insert(aligned_input_shape.begin(), 1);
        }
        
        // 确定哪些维度是广播的
        std::vector<bool> is_broadcast_dim(new_shape.size(), false);
        for (size_t i = 0; i < new_shape.size(); i++) {
            is_broadcast_dim[i] = (aligned_input_shape[i] == 1 && new_shape[i] > 1);
        }
        
        // 计算输入张量的步长
        std::vector<size_t> input_strides(aligned_input_shape.size(), 0);
        size_t input_stride = 1;
        for (int i = aligned_input_shape.size() - 1; i >= 0; --i) {
            input_strides[i] = (aligned_input_shape[i] > 1) ? input_stride : 0;
            input_stride *= aligned_input_shape[i];
        }
        
        // 初始化梯度
        std::vector<float> grad_data(input->size(), 0.0f);
        
        // 遍历输出梯度张量的每个元素
        for (size_t i = 0; i < grad_output_data.size(); ++i) {
            size_t input_index = 0;
            size_t remainder = i;
            bool valid = true;
            
            // 计算输入索引
            for (int j = new_shape.size() - 1; j >= 0; --j) {
                int dim_size = new_shape[j];
                int coord = remainder % dim_size;
                remainder /= dim_size;
                
                // 如果输入在该维度上大小不为1，检查坐标是否在输入范围内
                if (input_strides[j] != 0) {
                    if (coord >= aligned_input_shape[j]) {
                        valid = false;
                        break;
                    }
                    input_index += coord * input_strides[j];
                }
                // 否则（广播维度），我们不需要增加索引
            }
            
            if (valid) {
                #pragma omp atomic
                grad_data[input_index] += grad_output_data[i];
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
    }
    
    return grads;
}

/**
 * 执行ReLU激活函数操作（逐元素）
 * 
 * ReLU(x) = max(0, x)
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return ReLU激活结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr ReLUFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReLUFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算ReLU值
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::max(0.0f, input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, input_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算ReLU操作的梯度
 * 
 * grad_input = grad_output * (x > 0 ? 1 : 0)
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> ReLUFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_data = inputs_[0]->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(input_data.size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            grad_data[i] = grad_output_data[i] * (input_data[i] > 0 ? 1.0f : 0.0f);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行Sigmoid激活函数操作（逐元素）
 * 
 * Sigmoid(x) = 1 / (1 + exp(-x))
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return Sigmoid激活结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr SigmoidFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SigmoidFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算Sigmoid值（数值稳定版本）
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        // 数值稳定版本：避免大的负数导致exp溢出
        float x = input_data[i];
        if (x >= 0) {
            float exp_x = std::exp(-x);
            result_data[i] = 1.0f / (1.0f + exp_x);
        } else {
            float exp_x = std::exp(x);
            result_data[i] = exp_x / (exp_x + 1.0f);
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, input_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算Sigmoid操作的梯度
 * 
 * grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> SigmoidFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& output_data = output_->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            float s = output_data[i];
            grad_data[i] = grad_output_data[i] * s * (1 - s);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行Tanh激活函数操作（逐元素）
 * 
 * Tanh(x) = tanh(x)
 * 
 * @param inputs 输入极量列表（必须包含一个张量）
 * @return Tanh激活结果张量
 * @throws std::invalid_argument 如果输入数量不正确
 */
TensorPtr TanhFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("TanhFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_data = input->data();
    const auto& input_shape = input->shape();
    
    // 计算Tanh值
    std::vector<float> result_data(input_data.size());
    const size_t size = input_data.size();
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        result_data[i] = std::tanh(input_data[i]);
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, input_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算Tanh操作的梯度
 * 
 * grad_input = grad_output * (1 - tanh²(x))
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> TanhFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& output_data = output_->data();
        const auto& grad_output_data = grad_output->data();
        
        std::vector<float> grad_data(inputs_[0]->size());
        const size_t size = grad_data.size();
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < size; ++i) {
            float t = output_data[i];
            grad_data[i] = grad_output_data[i] * (1 - t * t);
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, inputs_[0]->shape(), false);
    }
    
    return grads;
}

/**
 * 执行Softmax激活函数操作（沿指定维度）
 * 
 * Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * @param inputs 输入张量列表（必须包含一个张量）
 * @return Softmax激活结果张量
 * @throws std::invalid_argument 如果输入数量不正确或维度无效
 */
TensorPtr SoftmaxFunction::apply(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("SoftmaxFunction requires exactly one input");
    }
    
    inputs_ = inputs;
    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    const auto& input_data = input->data();
    
    // 确定实际维度
    int actual_dim = dim_;
    if (actual_dim < 0) {
        actual_dim = input_shape.size() + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(input_shape.size())) {
        throw std::invalid_argument("Invalid dimension for softmax");
    }
    
    // 计算输出形状（与输入相同）
    std::vector<float> result_data(input_data.size());
    const int dim_size = input_shape[actual_dim];
    
    // 计算沿指定维度的元素数量
    size_t outer_size = 1;
    for (int i = 0; i < actual_dim; ++i) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }
    
    // 数值稳定的Softmax实现
    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            // 步骤1: 找到最大值（数值稳定性）
            float max_val = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < dim_size; ++k) {
                size_t idx = (i * dim_size + k) * inner_size + j;
                if (input_data[idx] > max_val) {
                    max_val = input_data[idx];
                }
            }
            
            // 步骤2: 计算指数和
            float exp_sum = 0.0f;
            for (int k = 0; k < dim_size; ++k) {
                size_t idx = (i * dim_size + k) * inner_size + j;
                float exp_val = std::exp(input_data[idx] - max_val);
                result_data[idx] = exp_val;
                exp_sum += exp_val;
            }
            
            // 步骤3: 归一化
            for (int k = 0; k < dim_size; ++k) {
                size_t idx = (i * dim_size + k) * inner_size + j;
                result_data[idx] /= exp_sum;
            }
        }
    }
    
    bool requires_grad = input->requires_grad();
    output_ = std::make_shared<Tensor>(result_data, input_shape, requires_grad);
    
    if (requires_grad) {
        output_->grad_fn_ = shared_from_this();
        output_->children_ = inputs;
        output_->is_leaf_ = false;
    }
    
    return output_;
}

/**
 * 计算Softmax操作的梯度
 * 
 * 对于Softmax函数 S = [s1, s2, ..., sn]，其雅可比矩阵为：
 * ∂s_i/∂x_j = s_i * (δ_ij - s_j)
 * 
 * grad_input = grad_output * (diag(S) - S·S^T)
 * 
 * @param grad_output 上游梯度张量
 * @return 输入张量的梯度
 */
std::vector<TensorPtr> SoftmaxFunction::backward(const TensorPtr& grad_output) {
    std::vector<TensorPtr> grads(1);
    
    if (inputs_[0]->requires_grad()) {
        const auto& input_shape = inputs_[0]->shape();
        const auto& grad_output_data = grad_output->data();
        const auto& output_data = output_->data();
        
        // 确定实际维度
        int actual_dim = dim_;
        if (actual_dim < 0) {
            actual_dim = input_shape.size() + actual_dim;
        }
        
        // 计算沿指定维度的元素数量
        size_t outer_size = 1;
        for (int i = 0; i < actual_dim; ++i) {
            outer_size *= input_shape[i];
        }
        
        size_t inner_size = 1;
        for (size_t i = actual_dim + 1; i < input_shape.size(); ++i) {
            inner_size *= input_shape[i];
        }
        
        const int dim_size = input_shape[actual_dim];
        std::vector<float> grad_data(input_shape.size());
        
        // 计算梯度
        #pragma omp parallel for
        for (size_t i = 0; i < outer_size; ++i) {
            for (size_t j = 0; j < inner_size; ++j) {
                // 计算点积: grad_output·S
                float dot_product = 0.0f;
                for (int k = 0; k < dim_size; ++k) {
                    size_t idx = (i * dim_size + k) * inner_size + j;
                    dot_product += grad_output_data[idx] * output_data[idx];
                }
                
                // 计算梯度
                for (int k = 0; k < dim_size; ++k) {
                    size_t idx = (i * dim_size + k) * inner_size + j;
                    grad_data[idx] = output_data[idx] * (grad_output_data[idx] - dot_product);
                }
            }
        }
        
        grads[0] = std::make_shared<Tensor>(grad_data, input_shape, false);
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
    
    input_shape_ = shape;
    
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

    // 预计算每个池化窗口的有效面积
    std::vector<int> window_areas(out_height * out_width);
    for (int h_out = 0; h_out < out_height; ++h_out) {
        const int h_start = h_out * stride_ - padding_;
        const int h_end = std::min(h_start + kernel_size_, in_height);
        const int h_low = std::max(0, h_start);
        const int h_size = h_end - h_low;
        
        for (int w_out = 0; w_out < out_width; ++w_out) {
            const int w_start = w_out * stride_ - padding_;
            const int w_end = std::min(w_start + kernel_size_, in_width);
            const int w_low = std::max(0, w_start);
            const int w_size = w_end - w_low;
            
            window_areas[h_out * out_width + w_out] = h_size * w_size;
        }
    }

    // 使用并行化和优化内存访问
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            const int channel_offset = b * in_channels * in_height * in_width + 
                                      c * in_height * in_width;
            
            for (int h_out = 0; h_out < out_height; ++h_out) {
                const int h_start = h_out * stride_ - padding_;
                const int h_end = std::min(h_start + kernel_size_, in_height);
                const int h_low = std::max(0, h_start);
                
                for (int w_out = 0; w_out < out_width; ++w_out) {
                    const int w_start = w_out * stride_ - padding_;
                    const int w_end = std::min(w_start + kernel_size_, in_width);
                    const int w_low = std::max(0, w_start);
                    
                    const int area = window_areas[h_out * out_width + w_out];
                    if (area == 0) {
                        *output_ptr++ = 0.0f;
                        continue;
                    }
                    
                    float sum = 0.0f;
                    const float* row_start = x_data + channel_offset + h_low * in_width;
                    
                    for (int h_in = h_low; h_in < h_end; ++h_in) {
                        const float* col_start = row_start + w_low;
                        const float* col_end = row_start + w_end;
                        
                        // 使用指针算术优化内存访问
                        for (const float* ptr = col_start; ptr < col_end; ++ptr) {
                            sum += *ptr;
                        }
                        row_start += in_width;
                    }
                    
                    *output_ptr++ = sum / area;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    auto output = std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
    
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

        // 预计算每个池化窗口的有效面积
        std::vector<int> window_areas(out_height * out_width);
        for (int h_out = 0; h_out < out_height; ++h_out) {
            const int h_start = h_out * stride_ - padding_;
            const int h_end = std::min(h_start + kernel_size_, in_height);
            const int h_low = std::max(0, h_start);
            const int h_size = h_end - h_low;
            
            for (int w_out = 0; w_out < out_width; ++w_out) {
                const int w_start = w_out * stride_ - padding_;
                const int w_end = std::min(w_start + kernel_size_, in_width);
                const int w_low = std::max(0, w_start);
                const int w_size = w_end - w_low;
                
                window_areas[h_out * out_width + w_out] = h_size * w_size;
            }
        }

        // 使用并行化和优化内存访问
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < in_channels; ++c) {
                const int channel_offset = b * in_channels * in_height * in_width + 
                                          c * in_height * in_width;
                float* grad_channel = grad_input_data.data() + channel_offset;
                
                for (int h_out = 0; h_out < out_height; ++h_out) {
                    const int h_start = h_out * stride_ - padding_;
                    const int h_end = std::min(h_start + kernel_size_, in_height);
                    const int h_low = std::max(0, h_start);
                    
                    for (int w_out = 0; w_out < out_width; ++w_out) {
                        const int w_start = w_out * stride_ - padding_;
                        const int w_end = std::min(w_start + kernel_size_, in_width);
                        const int w_low = std::max(0, w_start);
                        
                        const int area = window_areas[h_out * out_width + w_out];
                        if (area == 0) continue;
                        
                        const float grad_val = grad_output_data[
                            b * (in_channels * out_height * out_width) + 
                            c * (out_height * out_width) +
                            h_out * out_width + 
                            w_out] / area;
                        
                        float* row_start = grad_channel + h_low * in_width;
                        
                        for (int h_in = h_low; h_in < h_end; ++h_in) {
                            float* col_start = row_start + w_low;
                            float* col_end = row_start + w_end;
                            
                            // 使用指针算术优化内存访问
                            for (float* ptr = col_start; ptr < col_end; ++ptr) {
                                #pragma omp atomic
                                *ptr += grad_val;
                            }
                            row_start += in_width;
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

// MaxPool2dFunction 实现
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

    // 使用更高效的并行化和内存访问模式
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            const int in_base = b * in_b_stride + c * in_c_stride;
            const int out_base = b * out_b_stride + c * out_c_stride;
            
            for (int oh = 0; oh < out_height; ++oh) {
                const int h_start = oh * stride_ - padding_;
                const int h_end = std::min(h_start + kernel_size_, in_height);
                const int h_low = std::max(0, h_start);
                
                for (int ow = 0; ow < out_width; ++ow) {
                    const int w_start = ow * stride_ - padding_;
                    const int w_end = std::min(w_start + kernel_size_, in_width);
                    const int w_low = std::max(0, w_start);
                    
                    const int out_idx = out_base + oh * out_width + ow;
                    float max_val = -FLT_MAX; // 使用 FLT_MAX
                    int max_index = -1;
                    
                    // 使用指针算术优化内存访问
                    const float* window_start = x_data + in_base + h_low * in_width;
                    
                    for (int kh = h_low; kh < h_end; ++kh) {
                        const float* row_start = window_start + w_low;
                        const float* row_end = window_start + w_end;
                        
                        for (const float* ptr = row_start; ptr < row_end; ++ptr) {
                            if (*ptr > max_val) {
                                max_val = *ptr;
                                max_index = ptr - x_data;
                            }
                        }
                        window_start += in_width; // 移动到下一行
                    }
                    
                    output_data[out_idx] = (max_val > -FLT_MAX) ? max_val : 0.0f;
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
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    
    // 获取输入形状 [N, C_in, H_in, W_in]
    const auto& input_shape = input->shape();
    const int N = input_shape[0];
    const int C_in = input_shape[1];
    const int H_in = input_shape[2];
    const int W_in = input_shape[3];
    
    // 获取权重形状 [C_out, C_in, K, K]
    const auto& weight_shape = weight->shape();
    const int C_out = weight_shape[0];
    const int K = weight_shape[2];
    
    // 计算输出尺寸
    const int H_out = (H_in + 2 * padding_ - K) / stride_ + 1;
    const int W_out = (W_in + 2 * padding_ - K) / stride_ + 1;
    
    // 使用im2col方法优化卷积
    const int col_height = C_in * K * K;
    const int col_width = N * H_out * W_out;
    std::vector<float> col_data(col_height * col_width);
    
    // 执行im2col转换（高效实现）
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                const int h_start = h * stride_ - padding_;
                const int w_start = w * stride_ - padding_;
                
                for (int c = 0; c < C_in; ++c) {
                    for (int i = 0; i < K; ++i) {
                        const int h_idx = h_start + i;
                        if (h_idx < 0 || h_idx >= H_in) continue;
                        
                        for (int j = 0; j < K; ++j) {
                            const int w_idx = w_start + j;
                            if (w_idx < 0 || w_idx >= W_in) continue;
                            
                            const int col_index = ((n * H_out + h) * W_out + w) * col_height + 
                                                 (c * K * K + i * K + j);
                            const int in_index = ((n * C_in + c) * H_in + h_idx) * W_in + w_idx;
                            col_data[col_index] = input->data_ptr()[in_index];
                        }
                    }
                }
            }
        }
    }
    
    // 将权重重塑为2D矩阵 [C_out, C_in*K*K]
    const float* weight_ptr = weight->data_ptr();
    
    // 执行矩阵乘法: output = weight * col_matrix
    std::vector<float> output_data(N * C_out * H_out * W_out, 0.0f);
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int hw = 0; hw < H_out * W_out; ++hw) {
                float sum = 0.0f;
                for (int k = 0; k < col_height; ++k) {
                    const int weight_idx = c_out * col_height + k;
                    const int col_idx = (n * H_out * W_out + hw) * col_height + k;
                    sum += weight_ptr[weight_idx] * col_data[col_idx];
                }
                output_data[(n * C_out + c_out) * H_out * W_out + hw] = sum;
            }
        }
    }
    
    std::vector<int> output_shape = {N, C_out, H_out, W_out};
    bool requires_grad = input->requires_grad() || weight->requires_grad();
    output_ = std::make_shared<Tensor>(output_data, output_shape, requires_grad);
    
    if (requires_grad) {
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
    const float* grad_output_data = grad_output->data_ptr();
    const float* x_data = x->data_ptr();
    const float* weight_data = weight->data_ptr();

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
        
        // 优化1: 循环重排序，提高内存局部性
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int ic = 0; ic < in_channels_; ++ic) {
                const int in_base = b * in_b_stride + ic * in_c_stride;
                
                for (int oc = 0; oc < out_channels_; ++oc) {
                    const int weight_base = oc * weight_oc_stride + ic * weight_ic_stride;
                    
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
                            
                            // 优化2: 预计算权重偏移
                            const float* weight_ptr = weight_data + weight_base;
                            
                            // 优化3: 指针算术代替索引计算
                            float* grad_x_ptr = grad_x_data.data() + in_base + h_low * in_width;
                            
                            for (int kh = h_low; kh < h_end; ++kh) {
                                const int weight_h_offset = (kh - h_low) * kernel_size_;
                                const float* weight_row = weight_ptr + weight_h_offset;
                                
                                float* grad_x_row = grad_x_ptr + w_low;
                                const float* grad_x_row_end = grad_x_ptr + w_end;
                                
                                // 优化4: 内层循环向量化
                                #pragma omp simd
                                for (int kw = w_low; kw < w_end; ++kw) {
                                    const int weight_idx = weight_h_offset + (kw - w_low);
                                    *grad_x_row++ += grad_val * weight_row[kw - w_low];
                                }
                                
                                grad_x_ptr += in_width; // 下一行
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
        
        // 优化5: 使用局部累加器减少原子操作
        #pragma omp parallel for collapse(2)
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int ic = 0; ic < in_channels_; ++ic) {
                // 为每个线程创建局部累加器
                std::vector<float> local_grad(kernel_size_ * kernel_size_, 0.0f);
                
                for (int b = 0; b < batch_size; ++b) {
                    const int in_base = b * in_b_stride + ic * in_c_stride;
                    
                    for (int oh = 0; oh < out_height; ++oh) {
                        const int ih = oh * stride_ - padding_;
                        if (ih < 0 || ih + kernel_size_ > in_height) continue;
                        
                        for (int ow = 0; ow < out_width; ++ow) {
                            const int iw = ow * stride_ - padding_;
                            if (iw < 0 || iw + kernel_size_ > in_width) continue;
                            
                            const float grad_val = grad_output_data[
                                b * out_b_stride + oc * out_c_stride + oh * out_width + ow];
                            
                            // 优化6: 直接内存访问
                            const float* x_ptr = x_data + in_base + ih * in_width + iw;
                            
                            // 优化7: 内层循环向量化
                            #pragma omp simd
                            for (int kh = 0; kh < kernel_size_; ++kh) {
                                const int row_offset = kh * in_width;
                                for (int kw = 0; kw < kernel_size_; ++kw) {
                                    local_grad[kh * kernel_size_ + kw] += grad_val * x_ptr[row_offset + kw];
                                }
                            }
                        }
                    }
                }
                
                // 合并局部累加器到全局梯度
                const int weight_base = oc * weight_oc_stride + ic * weight_ic_stride;
                #pragma omp critical
                {
                    for (int i = 0; i < kernel_size_ * kernel_size_; ++i) {
                        grad_weight_data[weight_base + i] += local_grad[i];
                    }
                }
            }
        }
        grads[1] = std::make_shared<Tensor>(std::move(grad_weight_data), weight->shape(), false);
    }

    return grads;
}

} // namespace dlt    