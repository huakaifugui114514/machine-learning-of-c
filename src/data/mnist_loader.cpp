// mnist_loader.cpp
#include "mnist_loader.hpp"
#include <stdexcept>
#include <cstdint>

std::vector<std::vector<float>> MNISTLoader::load_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path);

    // 读取文件头
    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // 转换大端序到主机序
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic != 2051) throw std::runtime_error("Invalid image file format");

    // 读取图像数据
    std::vector<std::vector<float>> images;
    const size_t image_size = rows * cols;
    std::vector<uint8_t> buffer(image_size);
    
    for (size_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        std::vector<float> image;
        image.reserve(image_size);
        for (auto pixel : buffer) {
            image.push_back(static_cast<float>(pixel) / 255.0f); // 归一化
        }
        images.push_back(std::move(image));
    }
    return images;
}

std::vector<int> MNISTLoader::load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open labels file");

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    if (magic != 2049) throw std::runtime_error("Invalid label file format");

    std::vector<uint8_t> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);
    return std::vector<int>(buffer.begin(), buffer.end());
}