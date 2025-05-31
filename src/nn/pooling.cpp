#include "nn/pooling.hpp"
#include <algorithm>
#include <limits>

namespace dlt {
namespace nn {

MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

TensorPtr MaxPool2d::forward(const TensorPtr& x) {
    int batch_size = x->shape()[0];
    int in_channels = x->shape()[1];
    int in_height = x->shape()[2];
    int in_width = x->shape()[3];

    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    std::vector<float> output_data(batch_size * in_channels * out_height * out_width);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                for (int w_out = 0; w_out < out_width; ++w_out) {
                    int h_start = h_out * stride_ - padding_;
                    int w_start = w_out * stride_ - padding_;
                    float max_val = -std::numeric_limits<float>::max();
                    for (int h_kernel = 0; h_kernel < kernel_size_; ++h_kernel) {
                        for (int w_kernel = 0; w_kernel < kernel_size_; ++w_kernel) {
                            int h_in = h_start + h_kernel;
                            int w_in = w_start + w_kernel;
                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int index = b * in_channels * in_height * in_width + c * in_height * in_width + h_in * in_width + w_in;
                                max_val = std::max(max_val, x->data()[index]);
                            }
                        }
                    }
                    int output_index = b * in_channels * out_height * out_width + c * out_height * out_width + h_out * out_width + w_out;
                    output_data[output_index] = max_val;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    return std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
}

AvgPool2d::AvgPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

TensorPtr AvgPool2d::forward(const TensorPtr& x) {
    int batch_size = x->shape()[0];
    int in_channels = x->shape()[1];
    int in_height = x->shape()[2];
    int in_width = x->shape()[3];

    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    std::vector<float> output_data(batch_size * in_channels * out_height * out_width);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                for (int w_out = 0; w_out < out_width; ++w_out) {
                    int h_start = h_out * stride_ - padding_;
                    int w_start = w_out * stride_ - padding_;
                    float sum = 0.0f;
                    int count = 0;
                    for (int h_kernel = 0; h_kernel < kernel_size_; ++h_kernel) {
                        for (int w_kernel = 0; w_kernel < kernel_size_; ++w_kernel) {
                            int h_in = h_start + h_kernel;
                            int w_in = w_start + w_kernel;
                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int index = b * in_channels * in_height * in_width + c * in_height * in_width + h_in * in_width + w_in;
                                sum += x->data()[index];
                                ++count;
                            }
                        }
                    }
                    int output_index = b * in_channels * out_height * out_width + c * out_height * out_width + h_out * out_width + w_out;
                    output_data[output_index] = sum / count;
                }
            }
        }
    }

    std::vector<int> output_shape = {batch_size, in_channels, out_height, out_width};
    return std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
}

} // namespace nn
} // namespace dlt