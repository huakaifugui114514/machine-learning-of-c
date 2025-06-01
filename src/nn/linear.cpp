#include "nn/linear.hpp"
#include "ops.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

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
    
    weight_ = tensor(weight_data, {out_features_, in_features_}, true);
    
    // 初始化偏置（如果需要）
    if (bias) {
        std::vector<float> bias_data(out_features, 0.0f);
        bias_ = tensor(bias_data, {1, out_features}, true);
    }
}

TensorPtr Linear::forward(const TensorPtr& x) {
    // 检查输入特征维度
    if (x->shape()[1] != in_features_) {
        throw std::invalid_argument("Input features do not match the layer's in_features.");
    }

    // 执行矩阵乘法
    auto output = matmul(x, weight_);
    
    // 如果有偏置，加上偏置
    if (has_bias_) {
        // 扩展偏置以匹配输出维度
        int batch_size = x->shape()[0];
        std::vector<float> expanded_bias_data(batch_size * out_features_);
        for (int i = 0; i < batch_size; ++i) {
            std::copy(bias_->data().begin(), bias_->data().end(), expanded_bias_data.begin() + i * out_features_);
        }
        
        auto expanded_bias = tensor(expanded_bias_data, {batch_size, out_features_}, false);
        
        output = output + expanded_bias;
    }
    
    return output;
}

std::vector<TensorPtr> Linear::parameters() const {
    if (has_bias_) {
        return {weight_, bias_};
    } else {
        return {weight_};
    }
}
