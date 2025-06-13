#include "loss/cross_entropy_loss.hpp"
#include "ops.hpp"
#include <iostream>
#include <vector>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::loss;

// 辅助函数，用于打印张量的数据
void printTensor(const TensorPtr& tensor) {
    const auto& data = tensor->data();
    for (float val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    // 模拟输入数据和目标数据
    std::vector<float> input_data = {
        2.0f, 1.0f, 0.1f,
        0.1f, 2.0f, 1.0f
    };
    std::vector<float> target_data = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };

    // 创建输入张量和目标张量
    TensorPtr input = tensor(input_data, {2, 3}, true);
    TensorPtr target = tensor(target_data, {2, 3}, false);

    // 创建交叉熵损失函数对象
    CrossEntropyLoss cross_entropy_loss;

    // 前向传播计算损失
    TensorPtr loss = cross_entropy_loss.forward(input, target);

    // 打印损失值
    std::cout << "Cross Entropy Loss: ";
    printTensor(loss);

    // 反向传播计算梯度
    std::vector<TensorPtr> grads = cross_entropy_loss.backward();

    // 打印梯度
    std::cout << "Gradient of input: ";
    printTensor(grads[0]);

    return 0;
}