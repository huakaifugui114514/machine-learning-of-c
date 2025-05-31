#include <gtest/gtest.h>
#include "loss/mse_loss.hpp"
#include "tensor.hpp"
#include <vector>
#include <cmath>

using namespace dlt;
using namespace dlt::loss;

TEST(MSELossTest, ForwardPass) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> target_data = {0.5f, 2.5f, 2.0f};
    TensorPtr input = tensor(input_data, {3,1});
    TensorPtr target = tensor(target_data, {3,1});

    MSELoss mse_loss;
    TensorPtr loss = mse_loss.forward(input, target);

    // 计算预期MSE值: [(1-0.5)^2 + (2-2.5)^2 + (3-2)^2] / 3 = [0.25 + 0.25 + 1] / 3 ≈ 0.5
    float expected_loss = std::pow(1.0f - 0.5f, 2) + 
                         std::pow(2.0f - 2.5f, 2) + 
                         std::pow(3.0f - 2.0f, 2);
    expected_loss /= 3.0f;

    // 检查损失值
    EXPECT_EQ(loss->shape(), std::vector<int>({1}));  // 应为标量
    EXPECT_FLOAT_EQ(loss->data()[0], expected_loss);
}

TEST(MSELossTest, BackwardPass) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> target_data = {0.5f, 2.5f, 2.0f};
    TensorPtr input = tensor(input_data, {3});
    TensorPtr target = tensor(target_data, {3});

    MSELoss mse_loss;
    
    mse_loss.forward(input, target);  
    auto grads = mse_loss.backward();
    // 检查梯度数量
    ASSERT_EQ(grads.size(), 1);
    ASSERT_NE(grads[0], nullptr);

    // 计算预期梯度: 2*(input - target)/n
    // 输入[1,2,3]和目标[0.5,2.5,2], 梯度为:
    // [2*(1-0.5)/3, 2*(2-2.5)/3, 2*(3-2)/3] ≈ [0.333, -0.333, 0.666]
    std::vector<float> expected_grad = {
        2.0f * (1.0f - 0.5f) / 3.0f,
        2.0f * (2.0f - 2.5f) / 3.0f,
        2.0f * (3.0f - 2.0f) / 3.0f
    };

    EXPECT_EQ(grads[0]->shape(), std::vector<int>({3}));
    for (size_t i = 0; i < expected_grad.size(); ++i) {
        EXPECT_FLOAT_EQ(grads[0]->data()[i], expected_grad[i]);
    }
}

TEST(MSELossTest, ShapeMismatch) {
    // 形状不匹配的输入和目标
    TensorPtr input = tensor({1.0f, 2.0f}, {2});
    TensorPtr target = tensor({1.0f}, {1});

    MSELoss mse_loss;
    EXPECT_THROW({
        mse_loss.forward(input, target);
    }, std::invalid_argument);
}

TEST(MSELossTest, WithoutForwardCall) {
    MSELoss mse_loss;
    EXPECT_THROW({
        mse_loss.backward();
    }, std::runtime_error); 
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}