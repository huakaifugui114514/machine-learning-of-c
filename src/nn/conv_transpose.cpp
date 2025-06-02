#include "nn/conv_transpose.hpp"
#include "ops.hpp"
#include <stdexcept>
#include <cmath>
#include <random>
#include <vector>
#include <memory>

namespace dlt {
namespace nn {

// 索引计算工具函数
inline size_t tensor_index(const std::vector<int>& shape, int b, int c, int h, int w) {
    size_t index = b;
    for (size_t i = 1; i < shape.size(); ++i) {
        index *= shape[i];
    }
    if (shape.size() > 1) index += c * shape[2] * shape[3];
    if (shape.size() > 2) index += h * shape[3];
    if (shape.size() > 3) index += w;
    return index;
}

ConvTranspose2d::ConvTranspose2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels), 
      kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias) {
    
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
    const int weight_size = in_channels * out_channels * kernel_size * kernel_size;
    const float stdv = 1.0f / std::sqrt(in_channels * kernel_size * kernel_size);
    
    std::vector<float> weight_data(weight_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, stdv);
    
    for (int i = 0; i < weight_size; ++i) {
        weight_data[i] = dis(gen);
    }
    
    weight_ = tensor(weight_data, {in_channels, out_channels, kernel_size, kernel_size}, true);
    
    // 偏置初始化
    if (use_bias_) {
        std::vector<float> bias_data(out_channels, 0.0f);
        bias_ = tensor(bias_data, {out_channels}, true);
    }
}

TensorPtr ConvTranspose2d::forward(const TensorPtr& x) {
    const auto& shape = x->shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("ConvTranspose2d expects 4D input (batch, channels, height, width)");
    }
    
    const int batch_size = shape[0];
    const int in_channels = shape[1];
    const int in_height = shape[2];
    const int in_width = shape[3];

    if (in_channels != in_channels_) {
        throw std::invalid_argument("Input channels (" + std::to_string(in_channels) + 
                                   ") don't match layer channels (" + std::to_string(in_channels_) + ")");
    }

    // 更精确的输出尺寸计算
    const int out_height = (in_height - 1) * stride_ + kernel_size_ - 2 * padding_;
    const int out_width = (in_width - 1) * stride_ + kernel_size_ - 2 * padding_;
    
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Output dimensions must be positive. Calculated: " + 
                                   std::to_string(out_height) + "x" + std::to_string(out_width));
    }

    // 初始化输出张量
    std::vector<float> output_data(batch_size * out_channels_ * out_height * out_width, 0.0f);
    
    // 获取输入数据和权重数据的指针
    const std::vector<float>& x_data_vec = x->data();
    const float* x_data = x_data_vec.data();
    
    const std::vector<float>& weight_data_vec = weight_->data();
    const float* weight_data = weight_data_vec.data();
    
    const std::vector<int> weight_shape = {in_channels_, out_channels_, kernel_size_, kernel_size_};
    const std::vector<int> output_shape = {batch_size, out_channels_, out_height, out_width};

    // 转置卷积计算
    for (int b = 0; b < batch_size; ++b) {
        for (int c_in = 0; c_in < in_channels_; ++c_in) {
            for (int h_in = 0; h_in < in_height; ++h_in) {
                for (int w_in = 0; w_in < in_width; ++w_in) {
                    const size_t input_index = tensor_index(shape, b, c_in, h_in, w_in);
                    const float input_val = x_data[input_index];
                    
                    for (int c_out = 0; c_out < out_channels_; ++c_out) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            const int h_out = h_in * stride_ + kh - padding_;
                            
                            // 跳过无效行
                            if (h_out < 0 || h_out >= out_height) continue;
                            
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                const int w_out = w_in * stride_ + kw - padding_;
                                
                                // 跳过无效列
                                if (w_out < 0 || w_out >= out_width) continue;
                                
                                // 计算权重索引
                                const size_t weight_idx = tensor_index(
                                    weight_shape, 
                                    c_in, c_out, kh, kw
                                );
                                
                                // 计算输出索引
                                const size_t output_idx = tensor_index(
                                    output_shape, 
                                    b, c_out, h_out, w_out
                                );
                                
                                // 累加贡献
                                output_data[output_idx] += input_val * weight_data[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // 添加偏置
    if (use_bias_ && bias_) {
        // 获取偏置数据的指针
        const std::vector<float>& bias_data_vec = bias_->data();
        const float* bias_data = bias_data_vec.data();
        
        const int spatial_size = out_height * out_width;
        
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < out_channels_; ++c) {
                const float bias_val = bias_data[c];
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        const size_t index = tensor_index(output_shape, b, c, h, w);
                        output_data[index] += bias_val;
                    }
                }
            }
        }
    }

    return std::make_shared<Tensor>(output_data, output_shape, x->requires_grad());
}

std::vector<TensorPtr> ConvTranspose2d::parameters() const {
    if (use_bias_ && bias_) {
        return {weight_, bias_};
    }
    return {weight_};
}

} // namespace nn
} // namespace dlt