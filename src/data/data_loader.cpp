// pj/src/data/data_loader.cpp
#include "data/data_loader.hpp"
#include "data/image_loader.hpp"
#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

namespace dlt {
namespace data {

DataLoader::DataLoader(const std::string& data_dir, int batch_size, bool shuffle)
    : data_dir_(data_dir), batch_size_(batch_size), shuffle_(shuffle), current_index_(0), num_classes_(0) {
    load_image_files();
    if (shuffle_) {
        shuffle_indices();
    }
}

void DataLoader::load_data() {
    // 数据已在构造函数中加载
}

void DataLoader::load_image_files() {
    if (!fs::exists(data_dir_) || !fs::is_directory(data_dir_)) {
        std::cerr << "Error: Data directory " << data_dir_ << " does not exist or is not a directory." << std::endl;
        return;
    }

    int label = 0;
    for (const auto& category : fs::directory_iterator(data_dir_)) {
        if (category.is_directory()) {
            for (const auto& entry : fs::directory_iterator(category.path())) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        data_files_.push_back({entry.path().string(), label});
                    }
                }
            }
            ++label;
        }
    }
    num_classes_ = label;
    indices_.resize(data_files_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
}

void DataLoader::shuffle_indices() {
    static std::random_device rd;
    static std::mt19937 g(rd());
    std::shuffle(indices_.begin(), indices_.end(), g);
}

std::pair<std::vector<TensorPtr>, std::vector<int>> DataLoader::get_next_batch() {
    std::vector<TensorPtr> batch_images;
    std::vector<int> batch_labels;

    for (int i = 0; i < batch_size_ && current_index_ < indices_.size(); ++i) {
        int index = indices_[current_index_++];
        auto [file_path, label] = data_files_[index];
        TensorPtr image = load_image(file_path);
        if (image) {
            batch_images.push_back(image);
            batch_labels.push_back(label);
        }
    }

    return {batch_images, batch_labels};
}

bool DataLoader::has_next_batch() const {
    return current_index_ < indices_.size();
}

void DataLoader::reset() {
    current_index_ = 0;
    if (shuffle_) {
        shuffle_indices();
    }
}

TensorPtr DataLoader::load_image(const std::string& file_path) {
    return ImageLoader::load_image(file_path, 32); // 统一调整为32x32
}

size_t DataLoader::get_data_size() const {
    return data_files_.size();
}

} // namespace data
} // namespace dlt