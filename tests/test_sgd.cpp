#include <gtest/gtest.h>
#include "optimizer/sgd.hpp"
#include "tensor.hpp"


using namespace dlt;

TEST(SGDTest, BasicStepWithoutMomentum) {

    auto param = tensor({1.0f, 2.0f, 3.0f}, {3}, true);
    param->grad() = {0.1f, 0.2f, 0.3f};  
    optimizer::SGD sgd(0.1f);  // 学习率0.1，无动量
    sgd.add_parameters({param});
    sgd.step();
    
    // 验证参数更新
    const auto& data = param->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f - 0.1f*0.1f);  // 1 - lr*grad
    EXPECT_FLOAT_EQ(data[1], 2.0f - 0.1f*0.2f);
    EXPECT_FLOAT_EQ(data[2], 3.0f - 0.1f*0.3f);
}

TEST(SGDTest, BasicStepWithMomentum) {
    auto param = tensor({1.0f, 2.0f, 3.0f}, {3}, true);
    param->grad() = {0.5f, 1.0f, 1.5f}; 
    optimizer::SGD sgd(0.01f, 0.9f);  
    sgd.add_parameters({param});
    
    // 第一次更新
    sgd.step();
    const auto& data1 = param->data();
    // v = 0 + lr*grad = 0.01*0.5 = 0.005
    // new_param = 1.0 - 0.005 = 0.995
    EXPECT_NEAR(data1[0], 0.995f, 1e-6f);
    
    // 改变梯度并第二次更新
    param->grad() = {0.25f, 0.5f, 0.75f}; 
    sgd.step();
    const auto& data2 = param->data();
    // v = 0.9*0.005 + 0.01*0.25 = 0.0045 + 0.0025 = 0.007
    // new_param = 0.995 - 0.007 = 0.988
    EXPECT_NEAR(data2[0], 0.988f, 1e-6f);
}

TEST(SGDTest, ZeroGrad) {
    auto param = tensor({1.0f}, {1}, true);
    param->grad() = {0.5f};
    
    optimizer::SGD sgd(0.1f);
    sgd.add_parameters({param});
    
    sgd.zero_grad();
    EXPECT_FLOAT_EQ(param->grad()[0], 0.0f);
}

TEST(SGDTest, MultipleParameters) {
    auto param1 = tensor({1.0f}, {1}, true);
    auto param2 = tensor({2.0f}, {1}, true);
    param1->grad() = {0.1f};
    param2->grad() = {0.2f};
    
    optimizer::SGD sgd(0.1f);
    sgd.add_parameters({param1, param2});
    sgd.step();
    
    EXPECT_FLOAT_EQ(param1->data()[0], 1.0f - 0.1f*0.1f);
    EXPECT_FLOAT_EQ(param2->data()[0], 2.0f - 0.1f*0.2f);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}