// pj/tests/test_conv_transpose.cpp
#include <gtest/gtest.h>
#include "nn/conv_transpose.hpp"
#include "tensor.hpp"
#include <vector>
#include <cmath>

using namespace dlt;
using namespace dlt::nn;

TEST(ConvTranspose2dTest, Parameters) {
    int in_channels = 3;
    int out_channels = 2;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    bool use_bias = true;

    ConvTranspose2d conv_transpose_layer(in_channels, out_channels, kernel_size, stride, padding, use_bias);
    auto params = conv_transpose_layer.parameters();

    if (use_bias) {
        ASSERT_EQ(params.size(), 2);
        ASSERT_NE(params[0], nullptr);
        ASSERT_NE(params[1], nullptr);
    } else {
        ASSERT_EQ(params.size(), 1);
        ASSERT_NE(params[0], nullptr);
    }
}

TEST(ConvTranspose2dTest, ForwardPass) {
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
    bool use_bias = true;
    ConvTranspose2d conv_transpose_layer(in_channels, out_channels, kernel_size, stride, padding, use_bias);
    TensorPtr output = conv_transpose_layer.forward(x);

    int expected_out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int expected_out_width = (in_width - 1) * stride - 2 * padding + kernel_size;

    EXPECT_EQ(output->shape()[0], batch_size);
    EXPECT_EQ(output->shape()[1], out_channels);
    EXPECT_EQ(output->shape()[2], expected_out_height);
    EXPECT_EQ(output->shape()[3], expected_out_width);
}

TEST(ConvTranspose2dTest, InputChannelMismatch) {
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
    bool use_bias = true;

    EXPECT_THROW({
        ConvTranspose2d conv_transpose_layer(wrong_in_channels, out_channels, kernel_size, stride, padding, use_bias);
        conv_transpose_layer.forward(x);
    }, std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}