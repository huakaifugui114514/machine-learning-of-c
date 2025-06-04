#include "nn/pooling.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace dlt {
namespace nn {

// 索引计算工具函数
inline size_t tensor_index(const std::vector<int>& shape, int b, int c, int h, int w) {
    return ((b * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Invalid pooling parameters");
    }
}

TensorPtr MaxPool2d::forward(const TensorPtr& x) {
    const auto& shape = x->shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d expects 4D input");
    }
    
    const int batch_size = shape[0];
    const int in_channels = shape[1];
    const int in_height = shape[2];
    const int in_width = shape[3];

    const int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};

    // 预计算输出尺寸
    const int out_hw = out_height * out_width;
    const int in_chw = in_channels * in_height * in_width;
    const int out_chw = in_channels * out_hw;
    
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Output dimensions must be positive");
    }

    std::vector<float> output_data(batch_size * out_chw);
    const float* x_data = x->data_ptr();

    constexpr float FLT_MIN = -std::numeric_limits<float>::infinity();
    
    // 并行计算，提高性能
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                const int h_start = oh * stride_ - padding_;
                const int h_end = std::min(h_start + kernel_size_, in_height);
                const int h_low = std::max(0, h_start);
                
                for (int ow = 0; ow < out_width; ++ow) {
                    const int w_start = ow * stride_ - padding_;
                    const int w_end = std::min(w_start + kernel_size_, in_width);
                    const int w_low = std::max(0, w_start);
                    
                    float max_val = FLT_MIN;
                    for (int kh = h_low; kh < h_end; ++kh) {
                        for (int kw = w_low; kw < w_end; ++kw) {
                            const size_t idx = tensor_index(shape, b, c, kh, kw);
                            if (x_data[idx] > max_val) max_val = x_data[idx];
                        }
                    }
                    
                    const size_t out_idx = ((b * in_channels + c) * out_height + oh) * out_width + ow;
                    output_data[out_idx] = (max_val == FLT_MIN) ? 0.0f : max_val;
                }
            }
        }
    }
    
    return std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
}

AvgPool2d::AvgPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Invalid pooling parameters");
    }
}

TensorPtr AvgPool2d::forward(const TensorPtr& x) {
    const auto& shape = x->shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d expects 4D input");
    }
    
    const int batch_size = shape[0];
    const int in_channels = shape[1];
    const int in_height = shape[2];
    const int in_width = shape[3];

    const int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Output dimensions must be positive");
    }

    std::vector<float> output_data(batch_size * in_channels * out_height * out_width);
    const float* x_data = x->data_ptr();
    float* output_ptr = output_data.data();

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                const int h_start = h_out * stride_ - padding_;
                const int h_end = h_start + kernel_size_;
                const int h_low = std::max(0, h_start);
                const int h_high = std::min(in_height, h_end);

                for (int w_out = 0; w_out < out_width; ++w_out) {
                    const int w_start = w_out * stride_ - padding_;
                    const int w_end = w_start + kernel_size_;
                    const int w_low = std::max(0, w_start);
                    const int w_high = std::min(in_width, w_end);

                    float sum = 0.0f;
                    int count = 0;
                    
                    // 仅在有有效区域时计算
                    if (h_low < h_high && w_low < w_high) {
                        for (int h_in = h_low; h_in < h_high; ++h_in) {
                            for (int w_in = w_low; w_in < w_high; ++w_in) {
                                const size_t index = tensor_index(shape, b, c, h_in, w_in);
                                sum += x_data[index];
                                count++;
                            }
                        }
                    }
                    *output_ptr++ = (count > 0) ? (sum / count) : 0.0f;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    return std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
}

} // namespace nn
} // namespace dlt