#include "ops.hpp"
#include "autograd.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace dlt {
namespace ops {

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    // 检查形状是否匹配
    if (!shapes_match(a->shape(), b->shape())) {
        throw std::invalid_argument("Tensor shapes do not match for addition");
    }
    
    // 创建加法函数
    auto func = std::make_shared<AddFunction>();
    return func->apply({a, b});
}


TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
    // 检查形状是否匹配
    if (!shapes_match(a->shape(), b->shape())) {
        throw std::invalid_argument("Tensor shapes do not match for multiplication");
    }
    
    // 创建乘法函数
    auto func = std::make_shared<MulFunction>();
    return func->apply({a, b});
}


TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
    // 检查矩阵维度是否兼容
    auto result_shape = compute_matmul_result_shape(a->shape(), b->shape());
    if (result_shape.empty()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }
    
    // 创建矩阵乘法函数
    auto func = std::make_shared<MatMulFunction>();
    return func->apply({a, b});
}

TensorPtr sum(const TensorPtr& a) {
    // 创建求和函数
    auto func = std::make_shared<SumFunction>();
    return func->apply({a});
}

TensorPtr exp(const TensorPtr& a) {
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = std::exp(data[i]);
    }
    return std::make_shared<Tensor>(result_data, a->shape(), a->requires_grad());
}

TensorPtr log(const TensorPtr& a) {
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = std::log(data[i]);
    }
    return std::make_shared<Tensor>(result_data, a->shape(), a->requires_grad());
}

TensorPtr sin(const TensorPtr& a) {
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = std::sin(data[i]);
    }
    return std::make_shared<Tensor>(result_data, a->shape(), a->requires_grad());
}

TensorPtr cos(const TensorPtr& a) {
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = std::cos(data[i]);
    }
    return std::make_shared<Tensor>(result_data, a->shape(), a->requires_grad());
}

TensorPtr tan(const TensorPtr& a) {
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = std::tan(data[i]);
    }
    return std::make_shared<Tensor>(result_data, a->shape(), a->requires_grad());
}

TensorPtr max(const TensorPtr& a) {
    const auto& data = a->data();
    float max_val = *std::max_element(data.begin(), data.end());
    return std::make_shared<Tensor>(std::vector<float>{max_val}, std::vector<int>{1}, a->requires_grad());
}

TensorPtr min(const TensorPtr& a) {
    const auto& data = a->data();
    float min_val = *std::min_element(data.begin(), data.end());
    return std::make_shared<Tensor>(std::vector<float>{min_val}, std::vector<int>{1}, a->requires_grad());
}

// 简单实现按维度最大值，暂不考虑复杂形状
TensorPtr max(const TensorPtr& a, int dim) {
    // 这里简单处理二维情况
    if (a->shape().size() != 2) {
        throw std::invalid_argument("Only 2D tensors are supported for now");
    }
    std::vector<float> result;
    if (dim == 0) {
        for (int j = 0; j < a->shape()[1]; ++j) {
            float max_val = a->data()[j];
            for (int i = 1; i < a->shape()[0]; ++i) {
                max_val = std::max(max_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(max_val);
        }
    } else if (dim == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            float max_val = a->data()[i * a->shape()[1]];
            for (int j = 1; j < a->shape()[1]; ++j) {
                max_val = std::max(max_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(max_val);
        }
    }
    return std::make_shared<Tensor>(result, std::vector<int>{static_cast<int>(result.size())}, a->requires_grad());
}

TensorPtr min(const TensorPtr& a, int dim) {
    // 这里简单处理二维情况
    if (a->shape().size() != 2) {
        throw std::invalid_argument("Only 2D tensors are supported for now");
    }
    std::vector<float> result;
    if (dim == 0) {
        for (int j = 0; j < a->shape()[1]; ++j) {
            float min_val = a->data()[j];
            for (int i = 1; i < a->shape()[0]; ++i) {
                min_val = std::min(min_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(min_val);
        }
    } else if (dim == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            float min_val = a->data()[i * a->shape()[1]];
            for (int j = 1; j < a->shape()[1]; ++j) {
                min_val = std::min(min_val, a->data()[i * a->shape()[1] + j]);
            }
            result.push_back(min_val);
        }
    }
    return std::make_shared<Tensor>(result, std::vector<int>{static_cast<int>(result.size())}, a->requires_grad());
}

TensorPtr mean(const TensorPtr& a) {
    const auto& data = a->data();
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean_val = sum / data.size();
    return std::make_shared<Tensor>(std::vector<float>{mean_val}, std::vector<int>{1}, a->requires_grad());
}

// 简单实现按维度平均值，暂不考虑复杂形状
TensorPtr mean(const TensorPtr& a, int dim) {
    // 这里简单处理二维情况
    if (a->shape().size() != 2) {
        throw std::invalid_argument("Only 2D tensors are supported for now");
    }
    std::vector<float> result;
    if (dim == 0) {
        for (int j = 0; j < a->shape()[1]; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < a->shape()[0]; ++i) {
                sum += a->data()[i * a->shape()[1] + j];
            }
            result.push_back(sum / a->shape()[0]);
        }
    } else if (dim == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < a->shape()[1]; ++j) {
                sum += a->data()[i * a->shape()[1] + j];
            }
            result.push_back(sum / a->shape()[1]);
        }
    }
    return std::make_shared<Tensor>(result, std::vector<int>{static_cast<int>(result.size())}, a->requires_grad());
}

TensorPtr reshape(const TensorPtr& a, const std::vector<int>& new_shape) {
    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != a->size()) {
        throw std::invalid_argument("New shape does not match the size of the tensor");
    }
    return std::make_shared<Tensor>(a->data(), new_shape, a->requires_grad());
}

TensorPtr transpose(const TensorPtr& a, int dim0, int dim1) {
    if (a->shape().size() != 2 || dim0 < 0 || dim0 > 1 || dim1 < 0 || dim1 > 1) {
        throw std::invalid_argument("Only 2D tensors are supported for transpose");
    }
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    if (dim0 == 0 && dim1 == 1) {
        for (int i = 0; i < a->shape()[0]; ++i) {
            for (int j = 0; j < a->shape()[1]; ++j) {
                result_data[j * a->shape()[0] + i] = data[i * a->shape()[1] + j];
            }
        }
    }
    std::vector<int> new_shape = a->shape();
    std::swap(new_shape[dim0], new_shape[dim1]);
    return std::make_shared<Tensor>(result_data, new_shape, a->requires_grad());
}

TensorPtr concat(const std::vector<TensorPtr>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("No tensors to concatenate");
    }
    std::vector<int> output_shape = tensors[0]->shape();
    int total_size = 0;
    for (const auto& tensor : tensors) {
        if (tensor->shape().size() != output_shape.size()) {
            throw std::invalid_argument("Tensors must have the same number of dimensions");
        }
        for (size_t i = 0; i < output_shape.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                if (tensor->shape()[i] != output_shape[i]) {
                    throw std::invalid_argument("Tensors must have the same shape except for the concatenation dimension");
                }
            }
        }
        total_size += tensor->shape()[dim];
    }
    output_shape[dim] = total_size;
    std::vector<float> result_data;
    for (const auto& tensor : tensors) {
        result_data.insert(result_data.end(), tensor->data().begin(), tensor->data().end());
    }
    return std::make_shared<Tensor>(result_data, output_shape, tensors[0]->requires_grad());
}

std::vector<TensorPtr> split(const TensorPtr& a, int dim, int sections) {
    if (a->shape()[dim] % sections != 0) {
        throw std::invalid_argument("The dimension to split must be divisible by the number of sections");
    }
    std::vector<TensorPtr> result;
    std::vector<int> section_shape = a->shape();
    section_shape[dim] = a->shape()[dim] / sections;
    int section_size = 1;
    for (int dim_size : section_shape) {
        section_size *= dim_size;
    }
    for (int i = 0; i < sections; ++i) {
        std::vector<float> section_data(section_size);
        int start_index = i * section_size;
        std::copy(a->data().begin() + start_index, a->data().begin() + start_index + section_size, section_data.begin());
        result.push_back(std::make_shared<Tensor>(section_data, section_shape, a->requires_grad()));
    }
    return result;
}

TensorPtr dot(const TensorPtr& a, const TensorPtr& b) {
    if (a->size() != b->size()) {
        throw std::invalid_argument("Tensors must have the same size for dot product");
    }
    const auto& data_a = a->data();
    const auto& data_b = b->data();
    float dot_product = 0.0f;
    for (size_t i = 0; i < data_a.size(); ++i) {
        dot_product += data_a[i] * data_b[i];
    }
    return std::make_shared<Tensor>(std::vector<float>{dot_product}, std::vector<int>{1}, a->requires_grad() || b->requires_grad());
}

TensorPtr abs(const TensorPtr& a) {
    std::vector<float> result_data(a->size());
    const auto& data = a->data();
    for (size_t i = 0; i < data.size(); ++i) {
        result_data[i] = std::abs(data[i]);
    }
    return std::make_shared<Tensor>(result_data, a->shape(), a->requires_grad());
}


// 辅助函数实现
std::vector<int> compute_matmul_result_shape(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        return {}; // 不是二维矩阵
    }
    
    if (shape_a[1] != shape_b[0]) {
        return {}; // 维度不兼容
    }
    
    return {shape_a[0], shape_b[1]};
}

bool shapes_match(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
    if (shape_a.size() != shape_b.size()) {
        return false;
    }
    
    for (size_t i = 0; i < shape_a.size(); ++i) {
        if (shape_a[i] != shape_b[i]) {
            return false;
        }
    }
    
    return true;
}

} // namespace ops
} // namespace dlt    