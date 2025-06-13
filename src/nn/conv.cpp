#include "nn/conv.hpp"
#include "ops.hpp"
#include "autograd.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <memory>

namespace dlt {
namespace nn {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels), 
      kernel_size_(kernel_size), stride_(stride), padding_(padding), has_bias_(bias) {
    
    // 参数验证
    if (in_channels <= 0 || out_channels <= 0) {
        throw std::invalid_argument("Number of channels must be positive");
    }
    if (kernel_size <= 0) {
        throw std::invalid_argument("Kernel size must be positive");
    }
    if (stride <= 0) {
        throw std::invalid_argument("Stride must be positive");
    }
    if (padding < 0) {
        throw std::invalid_argument("Padding must be non-negative");
    }

    // 权重初始化
    const int weight_size = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
    const float stdv = 1.0f / std::sqrt(in_channels_ * kernel_size_ * kernel_size_);
    
    std::vector<float> weight_data(weight_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, stdv);
    
    for (int i = 0; i < weight_size; ++i) {
        weight_data[i] = dis(gen);
    }
    
    weight_ = tensor(weight_data, {out_channels_, in_channels_, kernel_size_, kernel_size_}, true);
    
    // 偏置初始化
    if (has_bias_) {
        std::vector<float> bias_data(out_channels_, 0.0f);
        bias_ = tensor(bias_data, {out_channels_}, true);
    }
}

TensorPtr Conv2d::forward(const TensorPtr& x) {
    auto conv_fn = std::make_shared<dlt::Conv2dFunction>(in_channels_, out_channels_, kernel_size_, stride_, padding_);
    auto output = conv_fn->apply({x, weight_});

    if (has_bias_) {
        // 正确的目标形状：[1, out_channels_, 1, 1]
        std::vector<int> bias_shape(4, 1);
        bias_shape[1] = out_channels_;
        
        // 扩展偏置并添加到输出
        auto expanded_bias = ops::expand(bias_, bias_shape);
        output = ops::add(output, expanded_bias);
    }
    return output;
}

std::vector<TensorPtr> Conv2d::parameters() const {
    if (has_bias_) {
        return {weight_, bias_};
    } else {
        return {weight_};
    }
}

} // namespace nn
} // namespace dlt