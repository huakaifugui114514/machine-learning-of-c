#include "data/data_loader.hpp"
#include "data/image_loader.hpp"
#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>
#include <numeric>
#include <mutex>

namespace fs = std::filesystem;

namespace dlt {
namespace data {

DataLoader::DataLoader(const std::string& data_dir, int batch_size, bool shuffle)
    : data_dir_(data_dir), batch_size_(batch_size), shuffle_(shuffle), 
      current_index_(0), num_classes_(0) {
    load_image_files();
    if (shuffle_) {
        shuffle_indices();
    }
}

void DataLoader::load_image_files() {
    if (!fs::exists(data_dir_) || !fs::is_directory(data_dir_)) {
        std::cerr << "Error: Data directory " << data_dir_ << " does not exist or is not a directory." << std::endl;
        return;
    }

    int label = 0;
    for (const auto& category : fs::directory_iterator(data_dir_)) {
        if (category.is_directory()) {
            bool has_image = false;
            for (const auto& entry : fs::directory_iterator(category.path())) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        data_files_.push_back({entry.path().string(), label});
                        has_image = true;
                    }
                }
            }
            
            if (has_image) {
                label++;  // 只在有图片的目录增加标签
            } else {
                std::cerr << "Warning: No images found in directory: " 
                          << category.path() << ". Skipping." << std::endl;
            }
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

TensorPtr DataLoader::get_from_cache(const std::string& path) {
    auto it = image_cache_.find(path);
    if (it != image_cache_.end()) {
        return it->second;
    }
    return nullptr;
}

void DataLoader::add_to_cache(const std::string& path, TensorPtr tensor) {
    if (image_cache_.size() >= MAX_CACHE_SIZE) {
        clean_cache();
    }
    image_cache_[path] = tensor;
}

void DataLoader::clean_cache() {
    // 简单策略：随机删除一半缓存
    size_t target_size = MAX_CACHE_SIZE / 2;
    while (image_cache_.size() > target_size) {
        auto it = image_cache_.begin();
        std::advance(it, rand() % image_cache_.size());
        image_cache_.erase(it);
    }
}

std::pair<std::vector<TensorPtr>, std::vector<int>> DataLoader::get_next_batch() {
    std::vector<TensorPtr> batch_images;
    std::vector<int> batch_labels;
    batch_images.reserve(batch_size_);
    batch_labels.reserve(batch_size_);

    size_t loaded_count = 0;
    while (loaded_count < batch_size_ && current_index_ < indices_.size()) {
        size_t index = indices_[current_index_];
        current_index_++;
        
        auto [file_path, label] = data_files_[index];
        TensorPtr image = get_from_cache(file_path);
        
        if (!image) {
            try {
                image = load_image(file_path);
                add_to_cache(file_path, image);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load image: " << file_path << " - " << e.what() << std::endl;
                continue;
            }
        }
        
        if (image) {
            batch_images.emplace_back(std::move(image));
            batch_labels.push_back(label);
            loaded_count++;
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
    return ImageLoader::load_image(file_path, 32);
}

size_t DataLoader::get_data_size() const {
    return data_files_.size();
}

int DataLoader::get_num_classes() const {
    return num_classes_;
}

} // namespace data
} // namespace dlt