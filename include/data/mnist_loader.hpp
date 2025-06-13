#include <vector>
#include <string>
#include <fstream>

class MNISTLoader {
public:
    static std::vector<std::vector<float>> load_images(const std::string& path);
    static std::vector<int> load_labels(const std::string& path);
};

