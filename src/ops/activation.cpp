#include "ops.hpp"
#include "autograd.hpp"

namespace dlt {
namespace ops {

TensorPtr relu(const TensorPtr& a) {
    // 创建ReLU函数
    auto func = std::make_shared<ReLUFunction>();
    return func->apply({a});
}

} // namespace ops
} // namespace dlt    