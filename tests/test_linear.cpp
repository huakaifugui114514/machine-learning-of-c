// pj/tests/test_linear.cpp
#include <gtest/gtest.h>
#include "nn/linear.hpp"
#include "tensor.hpp"
#include <vector>

using namespace dlt;
using namespace dlt::nn;

TEST(LinearTest, Parameters) {
    int in_features = 3;
    int out_features = 2;
    bool has_bias = true;

    Linear linear_layer(in_features, out_features, has_bias);
    auto params = linear_layer.parameters();

    if (has_bias) {
        ASSERT_EQ(params.size(), 2);
        ASSERT_NE(params[0], nullptr);
        ASSERT_NE(params[1], nullptr);
    } else {
        ASSERT_EQ(params.size(), 1);
        ASSERT_NE(params[0], nullptr);
    }
}

TEST(LinearTest, ForwardPass) {
    int in_features = 3;
    int out_features = 2;
    bool has_bias = true;
    Linear linear_layer(in_features, out_features, has_bias);

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
    std::vector<int> input_shape = {1, in_features};
    TensorPtr x = tensor(input_data, input_shape);

    std::cout << "before forward: " << std::endl;
    TensorPtr output = linear_layer.forward(x);
    std::cout << "after forward: " << std::endl;

    EXPECT_EQ(output->shape()[0], 1);
    EXPECT_EQ(output->shape()[1], out_features);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}