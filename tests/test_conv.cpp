#include <gtest/gtest.h>
#include "nn/conv.hpp"
#include "tensor.hpp"
#include <vector>
#include <cmath>

using namespace dlt;
using namespace dlt::nn;

TEST(Conv2dTest, Parameters) {
    int in_channels = 3;
    int out_channels = 2;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    bool bias = true;

    Conv2d conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias);
    auto params = conv_layer.parameters();

    ASSERT_EQ(params.size(), 2);
    ASSERT_NE(params[0], nullptr);  
    ASSERT_NE(params[1], nullptr);  
}

TEST(Conv2dTest, ForwardPass) {
    int batch_size = 1;
    int in_channels = 3;
    int in_height = 5;
    int in_width = 5;
    std::vector<float> input_data(batch_size * in_channels * in_height * in_width, 1.0f); 
    std::vector<int> input_shape = {batch_size, in_channels, in_height, in_width};
    TensorPtr x = tensor(input_data, input_shape);  

    int out_channels = 2;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    bool bias = true;
    Conv2d conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias);
    TensorPtr output = conv_layer.forward(x);

    int expected_out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int expected_out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    EXPECT_EQ(output->shape()[0], batch_size);
    EXPECT_EQ(output->shape()[1], out_channels);
    EXPECT_EQ(output->shape()[2], expected_out_height);
    EXPECT_EQ(output->shape()[3], expected_out_width);

}

TEST(Conv2dTest, ManualWeightSetting) {
    int batch_size = 1;
    int in_channels = 1;
    int in_height = 3;
    int in_width = 3;
    std::vector<float> input_data = {1, 2, 3,
                                    4, 5, 6,
                                    7, 8, 9};
    std::vector<int> input_shape = {batch_size, in_channels, in_height, in_width};
    TensorPtr x = tensor(input_data, input_shape);

    int out_channels = 1;
    int kernel_size = 2;
    int stride = 1;
    int padding = 0;
    bool bias = true;
    Conv2d conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias);

    auto params = conv_layer.parameters();
    std::vector<float> manual_weights = {1, 0,
                                        0, 1};  
    std::copy(manual_weights.begin(), manual_weights.end(), params[0]->data().begin());
    if (bias) {
        params[1]->data()[0] = 1.0f;
    }
    
    // 验证输出
    // 输入: 1 2 3  核: 1 0  偏置: 1
    //       4 5 6      0 1
    //       7 8 9
    // 输出计算:
    // (1*1 + 2*0 + 4*0 + 5*1) + 1 = 1 + 5 + 1 = 7
    // (2*1 + 3*0 + 5*0 + 6*1) + 1 = 2 + 6 + 1 = 9
    // (4*1 + 5*0 + 7*0 + 8*1) + 1 = 4 + 8 + 1 = 13
    // (5*1 + 6*0 + 8*0 + 9*1) + 1 = 5 + 9 + 1 = 15
    TensorPtr output = conv_layer.forward(x);
    std::vector<float> expected_output = {7, 9,
                                         13, 15};
    
    // 检查形状和预期数值     (3 - 2 + 0)/1 + 1 = 2
    EXPECT_EQ(output->shape()[0], batch_size);
    EXPECT_EQ(output->shape()[1], out_channels);  
    EXPECT_EQ(output->shape()[2], 2);
    EXPECT_EQ(output->shape()[3], 2);
    
    for (int i = 0; i < expected_output.size(); ++i) {
        EXPECT_FLOAT_EQ(output->data()[i], expected_output[i]);
    }
}


TEST(Conv2dTest, InputChannelMismatch) {
    //错误输入，形状不匹配
    int batch_size = 1;
    int in_channels = 3;
    int in_height = 5;
    int in_width = 5;
    std::vector<float> input_data(batch_size * in_channels * in_height * in_width, 1.0f);
    std::vector<int> input_shape = {batch_size, in_channels, in_height, in_width};
    TensorPtr x = tensor(input_data, input_shape);

    int wrong_in_channels = 4;
    int out_channels = 2;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    bool bias = true;
    
    EXPECT_THROW({
        Conv2d conv_layer(wrong_in_channels, out_channels, kernel_size, stride, padding, bias);
        conv_layer.forward(x);
    }, std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}