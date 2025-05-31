// pj/include/data/data_loader.hpp
#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include "tensor.hpp"
#include <vector>
#include <string>

namespace dlt {
namespace data {

class DataLoader {
public:
    DataLoader(const std::string& data_dir, int batch_size = 32, bool shuffle = true);
    void load_data();
    std::pair<std::vector<TensorPtr>, std::vector<int>> get_next_batch();
    bool has_next_batch() const;
    void reset();
    // 添加新方法
    size_t get_data_size() const; 

private:
    std::string data_dir_;
    int batch_size_;
    bool shuffle_;
    std::vector<std::pair<std::string, int>> data_files_;
    std::vector<int> indices_;
    size_t current_index_;
    int num_classes_;

    void load_image_files();
    void shuffle_indices();
    TensorPtr load_image(const std::string& file_path);
};

} // namespace data
} // namespace dlt

#endif // DATA_LOADER_HPP