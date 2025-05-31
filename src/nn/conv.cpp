#include "nn/conv.hpp"
#include "ops.hpp"
#include <cmath>
#include <random>
#include <stdexcept>  

namespace dlt {
namespace nn {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),kernel_size_(kernel_size), stride_(stride),padding_(padding), has_bias_(bias) {
    // 初始化权重 (out_channels, in_channels, kernel_size, kernel_size)
    std::vector<int> weight_shape = {out_channels_, in_channels_, kernel_size_, kernel_size_};
    int weight_size = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
    
    // 初始化weight
    float stdv = 1.0f / std::sqrt(in_channels_ * kernel_size_ * kernel_size_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, stdv);
    std::vector<float> weight_data(weight_size);
    for (int i = 0; i < weight_size; ++i) {
        weight_data[i] = dis(gen);
    }
    weight_ = tensor(weight_data, weight_shape, true);
    
    // 初始化偏置bias
    if (has_bias_) {
        std::vector<float> bias_data(out_channels_, 0.0f);
        bias_ = tensor(bias_data, {out_channels_}, true);
    }
}


TensorPtr Conv2d::forward(const TensorPtr& x) {
    // 检查输入形状 (batch_size, channels, height, width)
    if (x->shape().size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input (batch, channels, height, width)");
    }
    
    int batch_size = x->shape()[0];
    int in_channels = x->shape()[1];
    int in_height = x->shape()[2];
    int in_width = x->shape()[3];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels don't match layer channels");
    }
    
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    std::vector<float> output_data(batch_size * out_channels_ * out_height * out_width, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = b * in_channels_ * in_height * in_width + 
                                                   ic * in_height * in_width + ih * in_width + iw;
                                    int weight_idx = oc * in_channels_ * kernel_size_ * kernel_size_ + 
                                                    ic * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;
                                    
                                    sum += x->data()[input_idx] * weight_->data()[weight_idx];
                                }
                            }
                        }
                    }
                    
                    if (has_bias_) {
                        sum += bias_->data()[oc];
                    }
                    int output_idx = b * out_channels_ * out_height * out_width + 
                                    oc * out_height * out_width + oh * out_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }
    
    std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};
    return tensor(output_data, output_shape);
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