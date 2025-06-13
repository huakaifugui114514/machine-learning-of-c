#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace dlt {
namespace data {

class DataLoader {
public:
    DataLoader(const std::string& data_dir, int batch_size = 32, bool shuffle = true);
    void load_data();
    std::pair<std::vector<TensorPtr>, std::vector<int>> get_next_batch();
    bool has_next_batch() const;
    void reset();
    size_t get_data_size() const;
    int get_num_classes() const;  // 添加获取类别数的方法

private:
    std::string data_dir_;
    int batch_size_;
    bool shuffle_;
    std::vector<std::pair<std::string, int>> data_files_;
    std::vector<size_t> indices_;
    size_t current_index_;
    int num_classes_;
    
    // 添加图像缓存
    std::unordered_map<std::string, TensorPtr> image_cache_;
    static constexpr size_t MAX_CACHE_SIZE = 1000;

    void load_image_files();
    void shuffle_indices();
    TensorPtr load_image(const std::string& file_path);
    
    // 缓存管理
    void add_to_cache(const std::string& path, TensorPtr tensor);
    TensorPtr get_from_cache(const std::string& path);
    void clean_cache();
};

} // namespace data
} // namespace dlt

#endif // DATA_LOADER_HPP