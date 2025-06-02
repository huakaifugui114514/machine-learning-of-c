#include "nn/dropout2d.hpp"
#include "tensor.hpp"
#include <random>
#include <vector>
#include <memory>
#include <stdexcept>

namespace dlt {

TensorPtr dropout2d(const TensorPtr& input, float p, bool training) {
    // 如果不需要训练，直接返回输入
    if (!training || p == 0.0f) {
        return input;
    }

    // 确保输入是4维（batch, channels, height, width）
    if (input->dim() != 4) {  
        throw std::invalid_argument("dropout2d: input must be 4D tensor");
    }

    // 获取输入形状
    const auto& shape = input->shape();
    int batch_size = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];

    // 创建mask张量，形状为[batch_size, channels, 1, 1]（每个通道一个掩码）
    std::vector<int> mask_shape = {batch_size, channels, 1, 1};
    // 使用工厂函数创建mask，并设置requires_grad为false
    auto mask = dlt::ones(mask_shape, false);  // 正确使用工厂函数

    // 生成随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0f - p);  // 以1-p的概率生成true（保留）

    // 遍历mask的数据，按通道应用伯努利分布
    auto& mask_data = mask->data();
    for (int i = 0; i < batch_size * channels; ++i) {
        mask_data[i] = dist(gen) ? 1.0f : 0.0f;
        // 为了保持期望值不变，对保留的神经元进行缩放
        mask_data[i] /= (1.0f - p);
    }

    // 将mask扩展到整个特征图（利用广播机制）
    // 注意：我们不需要显式扩展，因为广播会自动进行（mask的形状是[batch, channels, 1, 1]）

    // 创建输出张量，与输入形状相同
    auto output = dlt::zeros(shape, input->requires_grad());

    // 执行乘法：output = input * mask (广播)
    const auto& input_data = input->data();
    auto& output_data = output->data();
    int feature_size = height * width;
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            float m = mask_data[b * channels + c];  // 对应通道的mask值
            for (int i = 0; i < feature_size; ++i) {
                output_data[b * channels * feature_size + c * feature_size + i] = 
                    input_data[b * channels * feature_size + c * feature_size + i] * m;
            }
        }
    }

    if (input->requires_grad()) {
        
    }

    return output;
}

} // namespace dlt