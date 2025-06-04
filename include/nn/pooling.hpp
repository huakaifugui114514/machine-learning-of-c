#ifndef POOLING_HPP
#define POOLING_HPP

#include "tensor.hpp"
#include "autograd.hpp"
#include <vector>

namespace dlt {
namespace nn {

class MaxPool2d {
public:
    MaxPool2d(int kernel_size, int stride, int padding);
    TensorPtr forward(const TensorPtr& x);
private:
    int kernel_size_;
    int stride_;
    int padding_;
    std::shared_ptr<MaxPool2dFunction> pool_fn_;
};

class AvgPool2d {
public:
    AvgPool2d(int kernel_size, int stride, int padding);
    TensorPtr forward(const TensorPtr& x);
private:
    int kernel_size_;
    int stride_;
    int padding_;
    std::vector<int> input_shape_;
    std::shared_ptr<AvgPool2dFunction> pool_fn_;
};

} // namespace nn
} // namespace dlt

#endif // POOLING_HPP