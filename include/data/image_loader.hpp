// pj/include/data/image_loader.hpp
#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include "tensor.hpp"
#include <string>
#include <vector>


namespace dlt {
namespace data {

class ImageLoader {
public:
    static TensorPtr load_image(const std::string& path, int target_size = 32);
};

} // namespace data
} // namespace dlt

#endif // IMAGE_LOADER_HPP