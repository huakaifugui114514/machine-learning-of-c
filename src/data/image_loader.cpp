#include "data/image_loader.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"

namespace dlt {
namespace data {

TensorPtr ImageLoader::load_image(const std::string& path, int target_size) {
    int width, height, orig_channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &orig_channels, 3);
    
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path + " (" + stbi_failure_reason() + ")");
    }

    // 使用双线性插值调整大小
    std::vector<unsigned char> resized_data(target_size * target_size * 3);
    stbir_resize_uint8(data, width, height, 0,
                       resized_data.data(), target_size, target_size, 0, 3);
    
    stbi_image_free(data);

    // 直接创建float张量避免额外复制
    std::vector<float> float_data;
    float_data.reserve(resized_data.size());
    
    for (size_t i = 0; i < resized_data.size(); i++) {
        float_data.push_back(static_cast<float>(resized_data[i]) / 255.0f);
    }

    // 创建张量 [1, 3, target_size, target_size]
    return tensor(std::move(float_data), {1, 3, target_size, target_size}, false);
}

} // namespace data
} // namespace dlt