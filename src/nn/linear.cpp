#include <iostream>
#include "nn/linear.hpp"
#include "ops.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <sstream>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::nn;

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
    // 初始化权重
    std::vector<float> weight_data(in_features * out_features);
    
    // Xavier初始化
    float stdv = 1.0f / std::sqrt(in_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-stdv, stdv);
    
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = dis(gen);
    }
    
    weight_ = tensor(weight_data, {in_features_, out_features_}, true);
    
    // 初始化偏置（如果需要）
    if (bias) {
        std::vector<float> bias_data(out_features, 0.0f);
        bias_ = tensor(bias_data, {1, out_features}, true);
    }
}

TensorPtr Linear::forward(const TensorPtr& x) {
    // 检查输入特征维度
    if (x->shape().size() < 2 || x->shape()[1] != in_features_) {
        std::ostringstream oss;
        oss << "Input features (" << (x->shape().size() < 2 ? "invalid" : std::to_string(x->shape()[1]))
            << ") do not match layer's in_features (" << in_features_ << ")";
        throw std::invalid_argument(oss.str());
    }
    
    // 确保输入是连续的
    auto x_contiguous = contiguous(x);
    
    // 执行矩阵乘法
    auto output = matmul(x_contiguous, weight_);
    
    // 如果有偏置，加上偏置（使用广播）
    if (has_bias_) {
        output = output + bias_;
    }
    
    return output;
}

std::vector<TensorPtr> Linear::parameters() const {
    if (has_bias_) {
        return {weight_, bias_};
    }
    return {weight_};
}