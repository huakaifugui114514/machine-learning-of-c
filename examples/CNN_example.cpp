#include "nn/conv.hpp"
#include "nn/pooling.hpp"
#include "nn/linear.hpp"
#include "loss/cross_entropy_loss.hpp"  
#include "optimizer/sgd.hpp"  
#include "ops.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>  

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::nn;
using namespace dlt::loss;  
using namespace dlt::optimizer;  

// 定义CNN模型
class MNISTCNN {
public:
    Conv2d conv1;
    AvgPool2d pool1;
    Linear fc1;

    MNISTCNN()
        : conv1(1, 16, 3, 1, 1), // 输入通道1，输出通道16，卷积核大小3，步长1，填充1
          pool1(2, 2, 0), // 池化核大小2，步长2，填充0
          fc1(14 * 14 * 16, 10) {} // 输入特征数14*14*16，输出特征数10

    TensorPtr forward(const TensorPtr& x) {
        // 第一层卷积 + ReLU激活
        auto h = conv1.forward(x);
        h = relu(h);

        // 池化层
        h = pool1.forward(h);

        // 展平
        int batch_size = x->shape()[0];
        h = reshape(h, {batch_size, 14 * 14 * 16});

        // 全连接层
        return fc1.forward(h);
    }

    std::vector<TensorPtr> parameters() {
        auto params = conv1.parameters();
        auto fc1_params = fc1.parameters();
        params.insert(params.end(), fc1_params.begin(), fc1_params.end());
        return params;
    }
};

// 加载MNIST数据集
std::pair<std::vector<TensorPtr>, std::vector<TensorPtr>> load_mnist(const std::string& image_file, const std::string& label_file) {
    std::ifstream image_stream(image_file, std::ios::binary);
    std::ifstream label_stream(label_file, std::ios::binary);

    int magic_number, num_images, num_rows, num_cols;
    image_stream.read(reinterpret_cast<char*>(&magic_number), 4);
    image_stream.read(reinterpret_cast<char*>(&num_images), 4);
    image_stream.read(reinterpret_cast<char*>(&num_rows), 4);
    image_stream.read(reinterpret_cast<char*>(&num_cols), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    int label_magic, num_labels;
    label_stream.read(reinterpret_cast<char*>(&label_magic), 4);
    label_stream.read(reinterpret_cast<char*>(&num_labels), 4);

    label_magic = __builtin_bswap32(label_magic);
    num_labels = __builtin_bswap32(num_labels);

    std::vector<TensorPtr> images;
    std::vector<TensorPtr> labels;

    for (int i = 0; i < num_images; ++i) {
        std::vector<float> image_data(num_rows * num_cols);
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel;
            image_stream.read(reinterpret_cast<char*>(&pixel), 1);
            image_data[j] = static_cast<float>(pixel) / 255.0f;
        }
        images.push_back(tensor(image_data, {1, 1, num_rows, num_cols}, true));

        unsigned char label;
        label_stream.read(reinterpret_cast<char*>(&label), 1);
        std::vector<float> label_data(10, 0.0f);
        label_data[label] = 1.0f;
        labels.push_back(tensor(label_data, {1, 10}, false));
    }

    return {images, labels};
}

int main() {
    // 配置参数
    const float learning_rate = 0.01;
    const int num_epochs = 10;

    // 创建模型
    MNISTCNN model;

    // 创建损失函数和优化器
    CrossEntropyLoss cross_entropy_loss;  
    SGD optimizer(learning_rate);  

    // 加载MNIST数据集
    auto [train_images, train_labels] = load_mnist("resources/mnist/train-images-idx3-ubyte/train-images.idx3-ubyte", "resources/mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    auto [test_images, test_labels] = load_mnist("resources/mnist/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", "resources/mnist/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    const int batch_size = 64;
    const int num_batches = train_images.size() / batch_size;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            optimizer.zero_grad();
            TensorPtr batch_loss = zeros({1});
            
            // 处理小批量
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch_idx * batch_size + i;
                TensorPtr output = model.forward(train_images[idx]);
                TensorPtr loss = cross_entropy_loss.forward(output, train_labels[idx]);
                batch_loss = add(batch_loss, loss);
            }
            
            // 平均损失
            batch_loss = mul(batch_loss, tensor({1.0f / batch_size}, {1}));
            batch_loss->backward();
            optimizer.step();
            
            epoch_loss += batch_loss->data()[0];
        }
        
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                  << "], Loss: " << epoch_loss / num_batches << std::endl;
    }

    // 评估模型
    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        TensorPtr output = model.forward(test_images[i]);
        const auto& output_data = output->data();
        // 修正：确保使用正确的迭代器范围
        int predicted = std::distance(output_data.begin(), 
                                     std::max_element(output_data.begin(), output_data.end()));
        const auto& label_data = test_labels[i]->data();
        int label = std::distance(label_data.begin(), 
                                 std::max_element(label_data.begin(), label_data.end()));
        if (predicted == label) {
            correct++;
        }
    }

    std::cout << "Test Accuracy: " << static_cast<float>(correct) / test_images.size() << std::endl;

    return 0;
}