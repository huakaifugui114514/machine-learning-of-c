#include "nn/linear.hpp"
#include "loss/mse_loss.hpp"
#include "optimizer/sgd.hpp"
#include "tensor.hpp"
#include "ops.hpp"
#include <iostream>
#include <vector>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::nn;
using namespace dlt::loss;
using namespace dlt::optimizer;

int main() {
    // 定义输入特征数和输出特征数
    int in_features = 1;
    int out_features = 1;

    // 创建线性层（全连接层）
    Linear linear_layer(in_features, out_features, true);

    // 创建 MSE 损失函数
    MSELoss mse_loss;

    // 创建 SGD 优化器，学习率设置为 0.01，不使用动量
    SGD optimizer(0.01f);
    optimizer.add_parameters(linear_layer.parameters());

    // 生成一些简单的训练数据
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y_data = {2.0f, 4.0f, 6.0f, 8.0f};

    // 训练轮数
    int num_epochs = 1000;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();
        TensorPtr total_loss = zeros({1});
        
        for (size_t i = 0; i < x_data.size(); ++i) {
            TensorPtr x = tensor({x_data[i]}, {1, in_features});
            TensorPtr y = tensor({y_data[i]}, {1, out_features});
            
            TensorPtr output = linear_layer.forward(x);
            TensorPtr loss = mse_loss.forward(output, y);
            
            // 累积损失
            total_loss = add(total_loss, loss);
        }
        
        // 平均损失
        total_loss = mul(total_loss, tensor({1.0f / x_data.size()}, {1}));
        
        // 反向传播
        total_loss->backward();
        
        // 更新参数
        optimizer.step();

        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Loss: " << total_loss->data()[0] << std::endl;
        }
    }

    // 测试模型
    TensorPtr test_x = tensor({5.0f}, {1, in_features});
    TensorPtr test_output = linear_layer.forward(test_x);
    std::cout << "Prediction for x = 5: " << test_output->data()[0] << std::endl;

    return 0;
}