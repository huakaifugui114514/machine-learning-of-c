#include <gtest/gtest.h>
#include "loss/cross_entropy_loss.hpp"
#include "tensor.hpp"
#include <vector>
#include <cmath>

using namespace dlt;
using namespace dlt::loss;

TEST(CrossEntropyLossTest, ForwardPass) {
    std::vector<float> input_data = {0.2f, 0.5f, 0.3f}; 
    std::vector<float> target_data = {0.0f, 1.0f, 0.0f};
    TensorPtr input = tensor(input_data, {3,1});
    TensorPtr target = tensor(target_data, {3,1});

    CrossEntropyLoss ce_loss;
    TensorPtr loss = ce_loss.forward(input, target);

    // 计算预期交叉熵值: -sum(target * log(input))
    // -[0*log(0.2) + 1*log(0.5) + 0*log(0.3)] = -log(0.5) ≈ 0.693147
    float expected_loss = -std::log(0.5f);

    // 检查损失值
    EXPECT_EQ(loss->shape(), std::vector<int>({1}));  
    EXPECT_NEAR(loss->data()[0], expected_loss, 1e-6f);
}

TEST(CrossEntropyLossTest, BackwardPass) {
    std::vector<float> input_data = {0.2f, 0.5f, 0.3f};  
    std::vector<float> target_data = {0.0f, 1.0f, 0.0f}; 
    TensorPtr input = tensor(input_data, {3});
    TensorPtr target = tensor(target_data, {3});

    CrossEntropyLoss ce_loss;
    
    ce_loss.forward(input, target);  
    auto grads = ce_loss.backward();
    
    ASSERT_EQ(grads.size(), 1); 
    ASSERT_NE(grads[0], nullptr);

    // 计算预期梯度: -target / input
    // 对于输入[0.2,0.5,0.3]和目标[0,1,0], 梯度应为:
    // [0, -1/0.5, 0] = [0, -2, 0]
    std::vector<float> expected_grad = {
        0.0f,
        -1.0f / 0.5f,
        0.0f
    };

    // 检查梯度值
    EXPECT_EQ(grads[0]->shape(), std::vector<int>({3}));
    for (size_t i = 0; i < expected_grad.size(); ++i) {
        EXPECT_FLOAT_EQ(grads[0]->data()[i], expected_grad[i]);
    }
}

TEST(CrossEntropyLossTest, ShapeMismatch) {
    TensorPtr input = tensor({0.5f, 0.5f}, {2});
    TensorPtr target = tensor({1.0f}, {1});

    CrossEntropyLoss ce_loss;
    EXPECT_THROW({
        ce_loss.forward(input, target);
    }, std::invalid_argument);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}