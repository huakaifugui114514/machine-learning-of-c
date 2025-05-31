// test_dropout2d.cpp
#include <gtest/gtest.h>
#include "nn/dropout2d.hpp"
#include "tensor.hpp"
#include <vector>

using namespace dlt;
using namespace dlt::nn;

TEST(Dropout2dTest, ForwardInTrainMode) {
    Dropout2d dropout(0.5);
    dropout.train(); // 确保处于训练模式

    // 创建一个全1的张量
    std::vector<float> data(100, 1.0f);
    TensorPtr x = tensor(data, {1, 1, 10, 10}, false);

    TensorPtr output = dropout.forward(x);
    const auto& output_data = output->data();

    // 检查输出中约50%的元素为0，其余为2.0 (1.0 / (1 - 0.5))
    int zero_count = 0;
    for (float val : output_data) {
        if (val == 0.0f) {
            zero_count++;
        } else {
            EXPECT_FLOAT_EQ(val, 2.0f);
        }
    }

    // 由于随机性，实际零的数量可能会有波动，但应该接近50%
    EXPECT_NEAR(zero_count, 50, 10);
}

TEST(Dropout2dTest, ForwardInEvalMode) {
    Dropout2d dropout(0.5);
    dropout.eval(); // 切换到评估模式

    // 创建一个全1的张量
    std::vector<float> data(100, 1.0f);
    TensorPtr x = tensor(data, {1, 1, 10, 10}, false);

    TensorPtr output = dropout.forward(x);
    const auto& output_data = output->data();

    // 在评估模式下，dropout应该不改变输入
    for (float val : output_data) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}