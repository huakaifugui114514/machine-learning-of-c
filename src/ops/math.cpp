#include <iostream>
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

TensorPtr sub(const TensorPtr& a, const TensorPtr& b) {
    // 检查形状是否匹配
    if (!shapes_match(a->shape(), b->shape())) {
        throw std::invalid_argument("Tensor shapes do not match for subtraction");
    }
    
    // 创建减法函数
    auto func = std::make_shared<SubFunction>();
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
    auto func = std::make_shared<ExpFunction>();
    return func->apply({a});
}

TensorPtr log(const TensorPtr& a) {
    auto func = std::make_shared<LogFunction>();
    return func->apply({a});
}

TensorPtr sin(const TensorPtr& a) {
    auto func = std::make_shared<SinFunction>();
    return func->apply({a});
}

TensorPtr cos(const TensorPtr& a) {
    auto func = std::make_shared<CosFunction>();
    return func->apply({a});
}

TensorPtr tan(const TensorPtr& a) {
    auto func = std::make_shared<TanFunction>();
    return func->apply({a});
}

TensorPtr max(const TensorPtr& a) {
    auto func = std::make_shared<MaxFunction>();
    return func->apply({a});
}

TensorPtr min(const TensorPtr& a) {
    auto func = std::make_shared<MinFunction>();
    return func->apply({a});
}

TensorPtr max(const TensorPtr& a, int dim) {
    auto func = std::make_shared<MaxDimFunction>(dim);
    return func->apply({a});
}

TensorPtr min(const TensorPtr& a, int dim) {
    auto func = std::make_shared<MinDimFunction>(dim);
    return func->apply({a});
}

TensorPtr mean(const TensorPtr& a) {
    auto func = std::make_shared<MeanFunction>();
    return func->apply({a});
}

TensorPtr mean(const TensorPtr& a, int dim) {
    auto func = std::make_shared<MeanDimFunction>(dim);
    return func->apply({a});
}

TensorPtr reshape(const TensorPtr& a, const std::vector<int>& new_shape) {
    auto func = std::make_shared<ReshapeFunction>(new_shape);
    return func->apply({a});
}

std::vector<TensorPtr> split(const TensorPtr& a, int dim, int sections) {
    auto func = std::make_shared<SplitFunction>(dim, sections);
    std::vector<TensorPtr> inputs = {a};
    func->apply(inputs);
    std::vector<TensorPtr> result;
    const auto& input = inputs[0];
    const auto& input_shape = input->shape();
    int split_size = input_shape[dim] / sections;
    std::vector<int> split_shape = input_shape;
    split_shape[dim] = split_size;

    const auto& input_data = input->data();
    int total_size = input->size();
    int slice_size = total_size / sections;

    for (int i = 0; i < sections; ++i) {
        std::vector<float> split_data(slice_size);
        for (int j = 0; j < slice_size; ++j) {
            split_data[j] = input_data[i * slice_size + j];
        }
        TensorPtr split_tensor = std::make_shared<Tensor>(split_data, split_shape, input->requires_grad());
        result.push_back(split_tensor);

        if (input->requires_grad()) {
            split_tensor->set_grad_fn(func);
            for (const auto& child : inputs) {
                split_tensor->add_child(child);
            }
            split_tensor->set_is_leaf(false);
        }
    }
    return result;
}

TensorPtr transpose(const TensorPtr& a, int dim0, int dim1) {
    auto func = std::make_shared<TransposeFunction>(dim0, dim1);
    return func->apply({a});
}

TensorPtr concat(const std::vector<TensorPtr>& tensors, int dim) {
    auto func = std::make_shared<ConcatFunction>(dim);
    return func->apply(tensors);
}

TensorPtr dot(const TensorPtr& a, const TensorPtr& b) {
    auto func = std::make_shared<DotFunction>();
    return func->apply({a, b});
}

TensorPtr abs(const TensorPtr& a) {
    auto func = std::make_shared<AbsFunction>();
    return func->apply({a});
}

TensorPtr contiguous(const TensorPtr& a) {
    // 创建连续化函数
    auto func = std::make_shared<ContiguousFunction>();
    return func->apply({a});
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
    return shape_a == shape_b; 
}

} // namespace ops
} // namespace dlt    