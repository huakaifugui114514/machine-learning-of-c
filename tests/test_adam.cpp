// pj/tests/test_adam.cpp
#include <gtest/gtest.h>
#include "optimizer/adam.hpp"
#include "tensor.hpp"
#include <vector>

using namespace dlt;
using namespace dlt::optimizer;

TEST(AdamTest, AddParameters) {
    auto param = tensor({1.0f, 2.0f, 3.0f}, {3}, true);
    Adam adam(0.001f);
    adam.add_parameters({param});

    auto params = adam.get_parameters();
    ASSERT_EQ(params.size(), 1);
    ASSERT_EQ(params[0], param);
}

// TEST(AdamTest, Step) {
//     auto param = tensor({1.0f, 2.0f, 3.0f}, {3}, true);
//     param->grad() = {0.1f, 0.2f, 0.3f};
//     Adam adam(0.1f);
//     adam.add_parameters({param});

//     const auto& before_step_data = param->data();
//     // 输出梯度值
//     std::cout << "Gradient before step: ";
//     for (float g : param->grad()) {
//         std::cout << g << " ";
//     }
//     std::cout << std::endl;

//     adam.step();
//     const auto& after_step_data = param->data();

//     EXPECT_NE(before_step_data, after_step_data);
// }

TEST(AdamTest, ZeroGrad) {
    auto param = tensor({1.0f}, {1}, true);
    param->grad() = {0.5f};
    Adam adam(0.001f);
    adam.add_parameters({param});

    adam.zero_grad();
    EXPECT_FLOAT_EQ(param->grad()[0], 0.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}