#include "nn/conv.hpp"
#include "ops.hpp"
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
    // 检查输入形状 (batch_size, channels, height, width)
    if (x->shape().size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input (batch, channels, height, width)");
    }
    
    const int batch_size = x->shape()[0];
    const int in_channels = x->shape()[1];
    const int in_height = x->shape()[2];
    const int in_width = x->shape()[3];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels (" + std::to_string(in_channels) + 
                                   ") don't match layer channels (" + std::to_string(in_channels_) + ")");
    }
    
    const int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Output dimensions must be positive. Calculated: " + 
                                   std::to_string(out_height) + "x" + std::to_string(out_width));
    }
    
    std::vector<float> output_data(batch_size * out_channels_ * out_height * out_width, 0.0f);
    
    // 获取输入数据和权重数据的指针
    const std::vector<float>& x_data_vec = x->data();
    const float* x_data = x_data_vec.data();
    
    const std::vector<float>& weight_data_vec = weight_->data();
    const float* weight_data = weight_data_vec.data();
    
    // 卷积计算
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int ih = oh * stride_ + kh - padding_;
                                const int iw = ow * stride_ + kw - padding_;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    // 输入索引计算
                                    const size_t input_idx = 
                                        b * in_channels_ * in_height * in_width + 
                                        ic * in_height * in_width + 
                                        ih * in_width + iw;
                                    
                                    // 权重索引计算
                                    const size_t weight_idx = 
                                        oc * in_channels_ * kernel_size_ * kernel_size_ + 
                                        ic * kernel_size_ * kernel_size_ + 
                                        kh * kernel_size_ + kw;
                                    
                                    sum += x_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                    
                    // 添加偏置
                    if (has_bias_) {
                        sum += bias_->data()[oc]; // 注意：这里直接访问向量元素
                    }
                    
                    // 输出索引计算
                    const size_t output_idx = 
                        b * out_channels_ * out_height * out_width + 
                        oc * out_height * out_width + 
                        oh * out_width + ow;
                    
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