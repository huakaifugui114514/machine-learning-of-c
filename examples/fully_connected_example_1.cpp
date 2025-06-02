#include "nn/linear.hpp"
#include "loss/mse_loss.hpp"
#include "optimizer/adam.hpp"  // 修改1：包含Adam头文件
#include "ops.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::nn;
using namespace dlt::loss;
using namespace dlt::optimizer;

// 定义简单的神经网络模型
class Net {
public:
    Linear fc1;
    Linear fc2;

    Net(int input_size, int hidden_size, int output_size)
        : fc1(input_size, hidden_size, true), 
          fc2(hidden_size, output_size, true) {}

    TensorPtr forward(const TensorPtr& x) {
        // 第一层：线性变换 + ReLU激活
        auto h = fc1.forward(x);
        h = relu(h);
        
        // 第二层：线性变换（输出层）
        return fc2.forward(h);
    }

    std::vector<TensorPtr> parameters() {
        auto params = fc1.parameters();
        auto fc2_params = fc2.parameters();
        params.insert(params.end(), fc2_params.begin(), fc2_params.end());
        return params;
    }
};

int main() {
    // 配置参数
    const int input_size = 1;
    const int hidden_size = 8;
    const int output_size = 1;
    const float learning_rate = 0.001;  // 修改2：Adam通常使用更小的学习率
    const int num_epochs = 5000;
    const int num_samples = 100;

    // 创建模型
    Net model(input_size, hidden_size, output_size);
    
    // 创建损失函数和优化器
    MSELoss mse_loss;
    // 修改3：使用Adam优化器，设置学习率和默认超参数
    Adam optimizer(learning_rate, 0.9, 0.999, 1e-8);
    optimizer.add_parameters(model.parameters());

    // 生成非线性数据集 (y = sin(2πx) + 0.5*cos(4πx) + 噪声)
    std::vector<float> x_data(num_samples);
    std::vector<float> y_data(num_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    
    for (int i = 0; i < num_samples; ++i) {
        x_data[i] = dist(gen);
        y_data[i] = std::sin(2 * M_PI * x_data[i]) + 
                    0.5 * std::cos(4 * M_PI * x_data[i]) + 
                    noise_dist(gen);
    }

    // 训练循环
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();
        TensorPtr total_loss = zeros({1});
        
        for (int i = 0; i < num_samples; ++i) {
            // 准备数据
            TensorPtr x = tensor({x_data[i]}, {1, input_size});
            TensorPtr y = tensor({y_data[i]}, {1, output_size});
            
            // 前向传播
            TensorPtr output = model.forward(x);
            TensorPtr loss = mse_loss.forward(output, y);
            
            // 累积损失
            total_loss = add(total_loss, loss);
        }
        
        // 计算平均损失
        total_loss = mul(total_loss, tensor({1.0f / num_samples}, {1}));
        
        // 反向传播
        total_loss->backward();
        
        // 更新参数
        optimizer.step();

        // 每100轮打印损失
        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs 
                      << "], Loss: " << total_loss->data()[0] << std::endl;
        }
    }

    // 测试模型
    std::cout << "\nTesting model:\n";
    std::cout << "x\tTrue y\tPredicted y\n";
    
    for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
        TensorPtr test_x = tensor({x}, {1, input_size});
        TensorPtr test_output = model.forward(test_x);
        
        float true_y = std::sin(2 * M_PI * x) + 0.5 * std::cos(4 * M_PI * x);
        
        std::cout << x << "\t" << true_y << "\t" 
                  << test_output->data()[0] << std::endl;
    }

    return 0;
}