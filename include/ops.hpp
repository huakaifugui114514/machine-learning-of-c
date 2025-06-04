#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

namespace dlt {
namespace ops {

// 数学运算

TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr sub(const TensorPtr& a, const TensorPtr& b);
TensorPtr mul(const TensorPtr& a, const TensorPtr& b);
// 操作符重载
inline TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) {
     return add(a, b);
}
inline TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) {
     return sub(a, b);
}
inline TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) {
     return mul(a, b);
}

TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
TensorPtr sum(const TensorPtr& a);
TensorPtr exp(const TensorPtr& a);
TensorPtr log(const TensorPtr& a);
TensorPtr sin(const TensorPtr& a);
TensorPtr cos(const TensorPtr& a);
TensorPtr tan(const TensorPtr& a);
TensorPtr max(const TensorPtr& a);
TensorPtr min(const TensorPtr& a);
TensorPtr max(const TensorPtr& a, int dim);
TensorPtr min(const TensorPtr& a, int dim);
TensorPtr mean(const TensorPtr& a);
TensorPtr mean(const TensorPtr& a, int dim);
TensorPtr reshape(const TensorPtr& a, const std::vector<int>& new_shape);
TensorPtr transpose(const TensorPtr& a, int dim0, int dim1);
TensorPtr concat(const std::vector<TensorPtr>& tensors, int dim);
std::vector<TensorPtr> split(const TensorPtr& a, int dim, int sections);
TensorPtr dot(const TensorPtr& a, const TensorPtr& b);
TensorPtr abs(const TensorPtr& a);

// 激活函数
TensorPtr relu(const TensorPtr& a);
TensorPtr sigmoid(const TensorPtr& a);
TensorPtr tanh(const TensorPtr& a);
TensorPtr softmax(const TensorPtr& a, int dim = -1);

TensorPtr contiguous(const TensorPtr& a);

// 辅助函数
std::vector<int> compute_matmul_result_shape(const std::vector<int>& shape_a, const std::vector<int>& shape_b);
bool shapes_match(const std::vector<int>& shape_a, const std::vector<int>& shape_b);

} // namespace ops
} // namespace dlt

#endif // OPS_HPP    