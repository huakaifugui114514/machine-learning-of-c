// test_nll_loss.cpp
#include <gtest/gtest.h>
#include "loss/nll_loss.hpp"
#include "tensor.hpp"
#include <vector>

using namespace dlt;
using namespace dlt::loss;

TEST(NLLLossTest, Forward) {
    NLLLoss loss_fn;

    // 创建输入 (log概率)
    std::vector<float> input_data = {
        -1.0f, -2.0f, -3.0f,  // 样本1: 类别0的概率最高
        -3.0f, -2.0f, -1.0f   // 样本2: 类别2的概率最高
    };
    TensorPtr input = tensor(input_data, {2, 3}, true);

    // 创建目标 (类别索引)
    std::vector<float> target_data = {0.0f, 2.0f};
    TensorPtr target = tensor(target_data, {2}, false);

    TensorPtr loss = loss_fn.forward(input, target);

    // 计算预期损失: -(log(predict[0][0]) + log(predict[1][2])) / 2
    float expected_loss = -((-1.0f) + (-1.0f)) / 2.0f;
    EXPECT_FLOAT_EQ(loss->data()[0], expected_loss);
}

TEST(NLLLossTest, Backward) {
    NLLLoss loss_fn;

    // 创建输入
    std::vector<float> input_data = {
        -1.0f, -2.0f, -3.0f,
        -3.0f, -2.0f, -1.0f
    };
    TensorPtr input = tensor(input_data, {2, 3}, true);

    // 创建目标
    std::vector<float> target_data = {0.0f, 2.0f};
    TensorPtr target = tensor(target_data, {2}, false);

    // 前向传播
    TensorPtr loss = loss_fn.forward(input, target);

    // 反向传播
    auto grads = loss_fn.backward();
    TensorPtr grad_input = grads[0];

    // 检查梯度
    std::vector<float> expected_grad = {
        -0.5f, 0.0f, 0.0f,  // 样本1的梯度
        0.0f, 0.0f, -0.5f   // 样本2的梯度
    };

    const auto& grad_data = grad_input->data();
    for (size_t i = 0; i < grad_data.size(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], expected_grad[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}