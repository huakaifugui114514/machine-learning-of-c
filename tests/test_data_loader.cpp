#include <gtest/gtest.h>
#include "data/data_loader.hpp"
#include "tensor.hpp"
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace dlt;
using namespace dlt::data;
namespace fs = std::filesystem;

// 测试 DataLoader 的基本功能
TEST(DataLoaderTest, BasicFunctionality) {
    std::string data_dir = "../resources"; 
    // 从数据目录动态获取样本数量
    size_t total_samples = 0;
    for (const auto& category : fs::directory_iterator(data_dir)) {
        if (category.is_directory()) {
            for (const auto& entry : fs::directory_iterator(category.path())) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        total_samples++;
                    }
                }
            }
        }
    }

    // 设置批次大小为样本数的最小值（避免越界）
    int batch_size = std::min(32, static_cast<int>(total_samples));
    bool shuffle = true;

    // 创建 DataLoader 实例
    DataLoader data_loader(data_dir, batch_size, shuffle);

    // 加载数据
    data_loader.load_data();

    // 检查是否有下一个批次
    EXPECT_TRUE(data_loader.has_next_batch());

    // 获取下一个批次
    auto batch = data_loader.get_next_batch();
    std::vector<TensorPtr> images = batch.first;
    std::vector<int> labels = batch.second;

    // 检查批次大小
    EXPECT_EQ(images.size(), batch_size);
    EXPECT_EQ(labels.size(), batch_size);

    // 再次检查是否有下一个批次
    bool has_next = data_loader.has_next_batch();
    // 如果数据量大于一个批次，应该还有下一个批次
    if (data_loader.get_data_size() > batch_size) {
        EXPECT_TRUE(has_next);
    } else {
        EXPECT_FALSE(has_next);
    }

    // 重置 DataLoader
    data_loader.reset();
    // 重置后应该又有下一个批次
    EXPECT_TRUE(data_loader.has_next_batch());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}