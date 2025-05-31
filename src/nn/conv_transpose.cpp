#include "nn/conv_transpose.hpp"
#include <iostream>
#include <stdexcept> // 添加头文件以支持异常抛出

namespace dlt {
namespace nn {

ConvTranspose2d::ConvTranspose2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias) {
    // 修正：确保 tensor 函数调用参数正确
    weight_ = tensor(
        std::vector<float>(in_channels * out_channels * kernel_size * kernel_size), 
        {in_channels, out_channels, kernel_size, kernel_size}, 
        true  // 确保 requires_grad 参数被正确传递
    );
    
    if (use_bias_) {
        bias_ = tensor(
            std::vector<float>(out_channels), 
            {out_channels}, 
            true  // 确保 requires_grad 参数被正确传递
        );
    }
}

TensorPtr ConvTranspose2d::forward(const TensorPtr& x) {
    // 检查输入通道数
    int input_channels = x->shape()[1];
    if (input_channels != in_channels_) {
        throw std::invalid_argument("Input channels don't match layer channels");
    }

    int batch_size = x->shape()[0];
    int in_height = x->shape()[2];
    int in_width = x->shape()[3];

    int out_height = (in_height - 1) * stride_ - 2 * padding_ + kernel_size_;
    int out_width = (in_width - 1) * stride_ - 2 * padding_ + kernel_size_;

    std::vector<float> output_data(batch_size * out_channels_ * out_height * out_width);

    for (int b = 0; b < batch_size; ++b) {
        for (int c_out = 0; c_out < out_channels_; ++c_out) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                for (int w_out = 0; w_out < out_width; ++w_out) {
                    float sum = 0.0f;
                    for (int c_in = 0; c_in < in_channels_; ++c_in) {
                        for (int h_kernel = 0; h_kernel < kernel_size_; ++h_kernel) {
                            for (int w_kernel = 0; w_kernel < kernel_size_; ++w_kernel) {
                                int h_in = (h_out + padding_ - h_kernel) / stride_;
                                int w_in = (w_out + padding_ - w_kernel) / stride_;
                                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width && (h_out + padding_ - h_kernel) % stride_ == 0 && (w_out + padding_ - w_kernel) % stride_ == 0) {
                                    int input_index = b * in_channels_ * in_height * in_width + c_in * in_height * in_width + h_in * in_width + w_in;
                                    int weight_index = c_in * out_channels_ * kernel_size_ * kernel_size_ + c_out * kernel_size_ * kernel_size_ + h_kernel * kernel_size_ + w_kernel;
                                    sum += x->data()[input_index] * weight_->data()[weight_index];
                                }
                            }
                        }
                    }
                    if (use_bias_ && bias_) {
                        sum += bias_->data()[c_out];
                    }
                    int output_index = b * out_channels_ * out_height * out_width + c_out * out_height * out_width + h_out * out_width + w_out;
                    output_data[output_index] = sum;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};
    return std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
}

std::vector<TensorPtr> ConvTranspose2d::parameters() const {
    if (use_bias_ && bias_) {
        return {weight_, bias_};
    } else {
        return {weight_};
    }
}

} // namespace nn
} // namespace dlt