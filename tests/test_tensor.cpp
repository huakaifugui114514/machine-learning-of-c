#include <gtest/gtest.h>
#include "tensor.hpp"
#include "ops.hpp"
#include <cmath>

using namespace dlt;

TEST(TensorTest, CreateTensor) {
    auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_EQ(t->shape()[0], 2);
    EXPECT_EQ(t->shape()[1], 2);
    EXPECT_EQ(t->size(), 4);
}

TEST(TensorTest, TensorAddition) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    auto c = (*a) + b;
    
    EXPECT_EQ(c->shape()[0], 2);
    EXPECT_EQ(c->shape()[1], 2);
    
    const auto& data = c->data();
    EXPECT_FLOAT_EQ(data[0], 6.0f);
    EXPECT_FLOAT_EQ(data[1], 8.0f);
    EXPECT_FLOAT_EQ(data[2], 10.0f);
    EXPECT_FLOAT_EQ(data[3], 12.0f);
}

TEST(TensorTest, TensorMultiplication) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    auto c = (*a) * b;
    
    EXPECT_EQ(c->shape()[0], 2);
    EXPECT_EQ(c->shape()[1], 2);
    
    const auto& data = c->data();
    EXPECT_FLOAT_EQ(data[0], 5.0f);
    EXPECT_FLOAT_EQ(data[1], 12.0f);
    EXPECT_FLOAT_EQ(data[2], 21.0f);
    EXPECT_FLOAT_EQ(data[3], 32.0f);
}

TEST(TensorTest, MatrixMultiplication) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    auto c = a->matmul(b);
    
    EXPECT_EQ(c->shape()[0], 2);
    EXPECT_EQ(c->shape()[1], 2);
    
    const auto& data = c->data();
    EXPECT_FLOAT_EQ(data[0], 19.0f);
    EXPECT_FLOAT_EQ(data[1], 22.0f);
    EXPECT_FLOAT_EQ(data[2], 43.0f);
    EXPECT_FLOAT_EQ(data[3], 50.0f);
}

TEST(TensorTest, ReLU) {
    auto a = tensor({-1.0f, 2.0f, -3.0f, 4.0f}, {2, 2});
    auto b = a->relu();
    
    EXPECT_EQ(b->shape()[0], 2);
    EXPECT_EQ(b->shape()[1], 2);
    
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(TensorTest, Sum) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = a->sum();
    
    EXPECT_EQ(b->shape()[0], 1);
    EXPECT_EQ(b->size(), 1);
    
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 10.0f);
}

TEST(TensorTest, Exp) {
    auto a = tensor({0.0f, 1.0f}, {2});
    auto b = ops::exp(a);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], std::exp(0.0f));
    EXPECT_FLOAT_EQ(data[1], std::exp(1.0f));
}

TEST(TensorTest, Log) {
    auto a = tensor({1.0f, std::exp(1.0f)}, {2});
    auto b = ops::log(a);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], std::log(1.0f));
    EXPECT_FLOAT_EQ(data[1], std::log(std::exp(1.0f)));
}

TEST(TensorTest, Sin) {
    auto a = tensor({0.0f, M_PI / 2}, {2});
    auto b = ops::sin(a);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], std::sin(0.0f));
    EXPECT_FLOAT_EQ(data[1], std::sin(M_PI / 2));
}

TEST(TensorTest, Cos) {
    auto a = tensor({0.0f, M_PI}, {2});
    auto b = ops::cos(a);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], std::cos(0.0f));
    EXPECT_FLOAT_EQ(data[1], std::cos(M_PI));
}

TEST(TensorTest, Tan) {
    auto a = tensor({0.0f, M_PI / 4}, {2});
    auto b = ops::tan(a);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], std::tan(0.0f));
    EXPECT_FLOAT_EQ(data[1], std::tan(M_PI / 4));
}

TEST(TensorTest, GlobalMax_) {
    auto a = tensor({1.0f, 2.0f, 3.0f}, {3});
    auto b = ops::max(a);
    EXPECT_EQ(b->shape()[0], 1);
    EXPECT_FLOAT_EQ(b->data()[0], 3.0f);
}

TEST(TensorTest, GlobalMin_) {
    auto a = tensor({1.0f, 2.0f, 3.0f}, {3});
    auto b = ops::min(a);
    EXPECT_EQ(b->shape()[0], 1);
    EXPECT_FLOAT_EQ(b->data()[0], 1.0f);
}

TEST(TensorTest, MaxDim0) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = ops::max(a, 0);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 3.0f);
    EXPECT_FLOAT_EQ(data[1], 4.0f);
}

TEST(TensorTest, MinDim0) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = ops::min(a, 0);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
}

TEST(TensorTest, GlobalMean) {
    auto a = tensor({1.0f, 2.0f, 3.0f}, {3});
    auto b = ops::mean(a);
    EXPECT_EQ(b->shape()[0], 1);
    EXPECT_FLOAT_EQ(b->data()[0], (1.0f + 2.0f + 3.0f) / 3.0f);
}

TEST(TensorTest, MeanDim0) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = ops::mean(a, 0);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], (1.0f + 3.0f) / 2.0f);
    EXPECT_FLOAT_EQ(data[1], (2.0f + 4.0f) / 2.0f);
}

TEST(TensorTest, Reshape) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = ops::reshape(a, {4});
    EXPECT_EQ(b->shape()[0], 4);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(TensorTest, Transpose) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto b = ops::transpose(a, 0, 1);
    EXPECT_EQ(b->shape()[0], 2);
    EXPECT_EQ(b->shape()[1], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 3.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(TensorTest, Concat) {
    auto a = tensor({1.0f, 2.0f}, {2});
    auto b = tensor({3.0f, 4.0f}, {2});
    auto c = ops::concat({a, b}, 0);
    EXPECT_EQ(c->shape()[0], 4);
    const auto& data = c->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(TensorTest, Split) {
    auto a = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto tensors = ops::split(a, 0, 2);
    EXPECT_EQ(tensors.size(), 2);
    const auto& data1 = tensors[0]->data();
    const auto& data2 = tensors[1]->data();
    EXPECT_FLOAT_EQ(data1[0], 1.0f);
    EXPECT_FLOAT_EQ(data1[1], 2.0f);
    EXPECT_FLOAT_EQ(data2[0], 3.0f);
    EXPECT_FLOAT_EQ(data2[1], 4.0f);
}

TEST(TensorTest, Dot) {
    auto a = tensor({1.0f, 2.0f}, {2});
    auto b = tensor({3.0f, 4.0f}, {2});
    auto c = ops::dot(a, b);
    EXPECT_EQ(c->shape()[0], 1);
    EXPECT_FLOAT_EQ(c->data()[0], 1.0f * 3.0f + 2.0f * 4.0f);
}

TEST(TensorTest, Abs) {
    auto a = tensor({-1.0f, 2.0f}, {2});
    auto b = ops::abs(a);
    EXPECT_EQ(b->shape()[0], 2);
    const auto& data = b->data();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
