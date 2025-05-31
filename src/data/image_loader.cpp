// pj/src/data/image_loader.cpp
#include "data/image_loader.hpp"
#include <stdexcept>
#include <algorithm>


// 声明stb_image的函数
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace dlt {
namespace data {

TensorPtr ImageLoader::load_image(const std::string& path, int target_size) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3); // 强制加载为RGB
    
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path + " (" + stbi_failure_reason() + ")");
    }

    // 图像数据现在是RGB格式：[width*height*3]，按行优先存储
    std::vector<float> rgb_data(target_size * target_size * 3, 0.0f);

    // 简单的缩放（最近邻插值）
    for (int y = 0; y < target_size; ++y) {
        for (int x = 0; x < target_size; ++x) {
            // 计算原始图像中的对应位置
            int src_x = std::min(static_cast<int>(x * static_cast<float>(width) / target_size), width - 1);
            int src_y = std::min(static_cast<int>(y * static_cast<float>(height) / target_size), height - 1);
            
            // 计算在源数据和目标数据中的索引
            int src_idx = (src_y * width + src_x) * 3;
            int dst_idx = (y * target_size + x) * 3;
            
            // 归一化并复制像素值
            rgb_data[dst_idx]     = static_cast<float>(data[src_idx]) / 255.0f;     // R
            rgb_data[dst_idx + 1] = static_cast<float>(data[src_idx + 1]) / 255.0f; // G
            rgb_data[dst_idx + 2] = static_cast<float>(data[src_idx + 2]) / 255.0f; // B
        }
    }

    // 释放图像数据
    stbi_image_free(data);

    // 创建张量 [1, 3, target_size, target_size]
    std::vector<int> shape = {1, 3, target_size, target_size};
    return tensor(rgb_data, shape, false);
}

} // namespace data
} // namespace dlt