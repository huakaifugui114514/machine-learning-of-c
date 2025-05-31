// test_flatten.cpp
#include <gtest/gtest.h>
#include "nn/flatten.hpp"
#include "tensor.hpp"
#include <vector>

using namespace dlt;
using namespace dlt::nn;

TEST(FlattenTest, Forward) {
    Flatten flatten;

    // 创建一个4D张量 [1, 3, 4, 4]
    std::vector<float> data(48);
    for (int i = 0; i < 48; ++i) {
        data[i] = static_cast<float>(i);
    }
    TensorPtr x = tensor(data, {1, 3, 4, 4}, false);

    TensorPtr output = flatten.forward(x);

    // 检查输出形状是否为 [1, 48]
    EXPECT_EQ(output->shape()[0], 1);
    EXPECT_EQ(output->shape()[1], 48);

    // 检查数据是否保持不变
    const auto& output_data = output->data();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(output_data[i], data[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}