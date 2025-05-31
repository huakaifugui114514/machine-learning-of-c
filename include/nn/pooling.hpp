#ifndef POOLING_HPP
#define POOLING_HPP

#include "tensor.hpp"
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
};

class AvgPool2d {
public:
    AvgPool2d(int kernel_size, int stride, int padding);
    TensorPtr forward(const TensorPtr& x);
private:
    int kernel_size_;
    int stride_;
    int padding_;
};

} // namespace nn
} // namespace dlt

#endif // POOLING_HPP