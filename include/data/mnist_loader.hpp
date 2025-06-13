#include <vector>
#include <string>
#include <fstream>

namespace dlt {
namespace data {

class MNISTLoader {
public:
    static std::vector<std::vector<float>> load_images(const std::string& path);
    static std::vector<int> load_labels(const std::string& path);
};

} // namespace data
} // namespace dlt

