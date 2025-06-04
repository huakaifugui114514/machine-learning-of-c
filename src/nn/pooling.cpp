#include "nn/pooling.hpp"
#include "autograd.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace dlt {
namespace nn {

MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding), pool_fn_(std::make_shared<MaxPool2dFunction>(kernel_size, stride, padding)) {
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Invalid pooling parameters");
    }
}

TensorPtr MaxPool2d::forward(const TensorPtr& x) {
    return pool_fn_->apply({x});
}

AvgPool2d::AvgPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding), pool_fn_(std::make_shared<AvgPool2dFunction>(kernel_size, stride, padding)) {
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Invalid pooling parameters");
    }
}

TensorPtr AvgPool2d::forward(const TensorPtr& x) {
    return pool_fn_->apply({x});
}


} // namespace nn
} // namespace dlt