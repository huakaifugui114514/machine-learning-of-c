#include <iostream>
#include "ops.hpp"
#include "autograd.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace dlt {
namespace ops {

// ================== 数学运算 ==================
TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    // 添加广播支持
    if (a->shape() != b->shape()) {
        // 扩展张量以匹配形状
        auto expanded_a = expand(a, b->shape());
        auto expanded_b = expand(b, a->shape());
        auto func = std::make_shared<AddFunction>();
        return func->apply({expanded_a, expanded_b});
    }
    auto func = std::make_shared<AddFunction>();
    return func->apply({a, b});
}

TensorPtr sub(const TensorPtr& a, const TensorPtr& b) {
    auto func = std::make_shared<SubFunction>();
    return func->apply({a, b});
}

TensorPtr mul(const TensorPtr& a, const TensorPtr& b) {
    auto func = std::make_shared<MulFunction>();
    return func->apply({a, b});
}

std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, float scalar) {
    // 创建标量张量
    auto scalar_tensor = std::make_shared<Tensor>(std::vector<float>{scalar}, std::vector<int>{1});
    return mul(a, scalar_tensor);
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b) {
    // 确保内存连续性
    auto a_contiguous = contiguous(a);
    auto b_contiguous = contiguous(b);
    auto func = std::make_shared<MatMulFunction>();
    return func->apply({a_contiguous, b_contiguous});
}

TensorPtr sum(const TensorPtr& a, int dim, bool keepdims) {
    auto func = std::make_shared<SumFunction>(dim, keepdims);
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

TensorPtr max(const TensorPtr& a, int dim, bool keepdims) {
    auto func = std::make_shared<MaxFunction>(dim, keepdims);
    return func->apply({a});
}

TensorPtr min(const TensorPtr& a, int dim, bool keepdims) {
    auto func = std::make_shared<MinFunction>(dim, keepdims);
    return func->apply({a});
}

TensorPtr mean(const TensorPtr& a, int dim, bool keepdims) {
    auto func = std::make_shared<MeanFunction>(dim, keepdims);
    return func->apply({a});
}

TensorPtr reshape(const TensorPtr& a, const std::vector<int>& new_shape) {
    auto func = std::make_shared<ReshapeFunction>(new_shape);
    return func->apply({a});
}

TensorPtr transpose(const TensorPtr& a, int dim0, int dim1) {
    auto func = std::make_shared<TransposeFunction>(dim0, dim1);
    return func->apply({a});
}

TensorPtr concat(const std::vector<TensorPtr>& tensors, int dim) {
    auto func = std::make_shared<ConcatFunction>(dim);
    return func->apply(tensors);
}

std::vector<TensorPtr> split(const TensorPtr& a, int dim, int sections) {
    std::vector<TensorPtr> outputs;
    
    for (int i = 0; i < sections; ++i) {
        auto func = std::make_shared<SplitFunction>(dim, sections, i);
        outputs.push_back(func->apply({a}));
    }
    
    return outputs;
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
    auto func = std::make_shared<ContiguousFunction>();
    return func->apply({a});
}

TensorPtr expand(const TensorPtr& a, const std::vector<int>& new_shape) {
    auto func = std::make_shared<ExpandFunction>(new_shape);
    return func->apply({a});
}

// ================== 激活函数 ==================
TensorPtr relu(const TensorPtr& a) {
    auto func = std::make_shared<ReLUFunction>();
    return func->apply({a});
}

TensorPtr sigmoid(const TensorPtr& a) {
    auto func = std::make_shared<SigmoidFunction>();
    return func->apply({a});
}

TensorPtr tanh(const TensorPtr& a) {
    auto func = std::make_shared<TanhFunction>();
    return func->apply({a});
}

TensorPtr softmax(const TensorPtr& a, int dim) {
    auto func = std::make_shared<SoftmaxFunction>(dim);
    return func->apply({a});
}

// ================== 操作符重载 ==================
TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) {
    return add(a, b);
}

TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) {
    return sub(a, b);
}

TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) {
    return mul(a, b);
}

} // namespace ops
} // namespace dlt