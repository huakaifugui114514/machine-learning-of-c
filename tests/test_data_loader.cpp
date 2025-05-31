#include <gtest/gtest.h>
#include "data/data_loader.hpp"
#include "tensor.hpp"
#include <vector>

using namespace dlt;
using namespace dlt::data;

// 测试 DataLoader 的基本功能
TEST(DataLoaderTest, BasicFunctionality) {
    std::string data_dir = "../resources"; 
    int batch_size = 32;
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