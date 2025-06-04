#include "ops.hpp"
#include "autograd.hpp"

namespace dlt {
namespace ops {

TensorPtr relu(const TensorPtr& a) {
    // 创建ReLU函数
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

} // namespace ops
} // namespace dlt    