#ifndef CONV_TRANSPOSE_HPP
#define CONV_TRANSPOSE_HPP

#include "tensor.hpp"
#include <vector>

namespace dlt {
namespace nn {

class ConvTranspose2d {
public:
    ConvTranspose2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_bias);
    TensorPtr forward(const TensorPtr& x);
    std::vector<TensorPtr> parameters() const;
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool use_bias_; // 修改为 use_bias_ 避免混淆
    TensorPtr weight_;
    TensorPtr bias_;
};

} // namespace nn
} // namespace dlt

#endif // CONV_TRANSPOSE_HPP