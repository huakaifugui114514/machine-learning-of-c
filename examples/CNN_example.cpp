#include "nn/conv.hpp"
#include "nn/pooling.hpp"
#include "nn/linear.hpp"
#include "loss/cross_entropy_loss.hpp"  
#include "optimizer/adam.hpp"  
#include "ops.hpp"
#include "tensor.hpp"
#include "data/mnist_loader.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <chrono>

using namespace dlt;
using namespace dlt::ops;
using namespace dlt::nn;
using namespace dlt::loss;  
using namespace dlt::optimizer;  
using namespace dlt::data;

class MNISTCNN {
public:
    Conv2d conv1;
    AvgPool2d pool1;
    Linear fc1;

    MNISTCNN()
        : conv1(1, 16, 3, 1, 1),
          pool1(2, 2, 0),
          fc1(16 * 14 * 14, 10) {}

    TensorPtr forward(const TensorPtr& x) {
        auto h = conv1.forward(x);
        h = relu(h);
        h = pool1.forward(h);
        int batch_size = x->shape()[0];
        h = reshape(h, {batch_size, 16 * 14 * 14});
        return fc1.forward(h);
    }

    std::vector<TensorPtr> parameters() {
        auto params = conv1.parameters();
        auto fc1_params = fc1.parameters();
        params.insert(params.end(), fc1_params.begin(), fc1_params.end());
        return params;
    }
};

std::vector<float> to_one_hot(int label, int num_classes = 10) {
    std::vector<float> one_hot(num_classes, 0.0f);
    if (label >= 0 && label < num_classes) {
        one_hot[label] = 1.0f;
    }
    return one_hot;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
create_batches(const std::vector<std::vector<float>>& images, 
               const std::vector<int>& labels, 
               int batch_size, bool shuffle = true) {
    int num_samples = images.size();
    if (num_samples != static_cast<int>(labels.size())) {
        throw std::runtime_error("Number of images and labels must be equal");
    }

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    std::vector<std::vector<float>> batch_images;
    std::vector<std::vector<float>> batch_labels;
    
    for (int i = 0; i < num_samples; i += batch_size) {
        int current_batch_size = std::min(batch_size, num_samples - i);
        
        std::vector<float> image_batch;
        image_batch.reserve(current_batch_size * 1 * 28 * 28);
        
        std::vector<float> label_batch;
        label_batch.reserve(current_batch_size * 10);
        
        for (int j = 0; j < current_batch_size; ++j) {
            int idx = indices[i + j];
            image_batch.insert(image_batch.end(), images[idx].begin(), images[idx].end());
            
            auto one_hot = to_one_hot(labels[idx]);
            label_batch.insert(label_batch.end(), one_hot.begin(), one_hot.end());
        }
        
        batch_images.push_back(image_batch);
        batch_labels.push_back(label_batch);
    }
    
    return {batch_images, batch_labels};
}

int main() {
    // 配置参数
    const float learning_rate = 0.001;
    const int num_epochs = 10;
    const int batch_size = 64;

    // 创建模型
    MNISTCNN model;

    // 创建损失函数和优化器（改为Adam）
    CrossEntropyLoss cross_entropy_loss;  
    Adam optimizer(learning_rate);  // 使用Adam优化器
    optimizer.add_parameters(model.parameters());  // 添加模型参数

    // 使用MNISTLoader加载数据集 - 恢复缺失的代码
    std::cout << "Loading training data...\n";
    auto train_images = MNISTLoader::load_images("../resources/mnist/train-images-idx3-ubyte/train-images.idx3-ubyte");
    auto train_labels = MNISTLoader::load_labels("../resources/mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    
    std::cout << "Loading test data...\n";
    auto test_images = MNISTLoader::load_images("../resources/mnist/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    auto test_labels = MNISTLoader::load_labels("../resources/mnist/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    
    std::cout << "Creating batches...\n";
    auto [train_image_batches, train_label_batches] = 
        create_batches(train_images, train_labels, batch_size, true);
    
    auto [test_image_batches, test_label_batches] = 
        create_batches(test_images, test_labels, batch_size, false);

    const int num_batches = train_image_batches.size();  // 定义num_batches
    std::cout << "Training batches: " << num_batches << std::endl;

    // 训练循环
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            optimizer.zero_grad();
            
            auto batch_images = tensor(train_image_batches[batch_idx], 
                                      {static_cast<int>(train_image_batches[batch_idx].size()) / (28 * 28), 
                                       1, 28, 28}, true);
            
            auto batch_labels = tensor(train_label_batches[batch_idx], 
                                      {static_cast<int>(train_label_batches[batch_idx].size()) / 10, 10}, 
                                      false);
            
            TensorPtr output = model.forward(batch_images);
            TensorPtr loss = cross_entropy_loss.forward(output, batch_labels);
            loss->backward();
            optimizer.step();  // 使用Adam更新参数

            epoch_loss += loss->data()[0];
            
            if (batch_idx % 100 == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                          << "], Batch [" << batch_idx << "/" << num_batches
                          << "], Loss: " << loss->data()[0] << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                  << "], Average Loss: " << epoch_loss / num_batches
                  << ", Time: " << epoch_duration.count() << "ms" << std::endl;
    }

    // 评估模型
    std::cout << "Evaluating model...\n";
    int correct = 0;
    int total = 0;
    
    for (size_t i = 0; i < test_image_batches.size(); ++i) {
        auto batch_images = tensor(test_image_batches[i], 
                                  {static_cast<int>(test_image_batches[i].size()) / (28 * 28), 
                                   1, 28, 28}, true);
        
        auto batch_labels = tensor(test_label_batches[i], 
                                  {static_cast<int>(test_label_batches[i].size()) / 10, 10}, 
                                  false);
        
        TensorPtr output = model.forward(batch_images);
        const auto& output_data = output->data();
        const auto& label_data = batch_labels->data();
        
        int batch_size = output->shape()[0];
        for (int j = 0; j < batch_size; ++j) {
            int predicted = 0;
            float max_val = -std::numeric_limits<float>::max();
            
            // 找到预测类别
            for (int k = 0; k < 10; ++k) {
                if (output_data[j * 10 + k] > max_val) {
                    max_val = output_data[j * 10 + k];
                    predicted = k;
                }
            }
            
            // 找到真实类别
            int label = 0;
            max_val = -std::numeric_limits<float>::max();
            for (int k = 0; k < 10; ++k) {
                if (label_data[j * 10 + k] > max_val) {
                    max_val = label_data[j * 10 + k];
                    label = k;
                }
            }
            
            if (predicted == label) {
                correct++;
            }
            total++;
        }
    }

    std::cout << "Test Accuracy: " << static_cast<float>(correct) / total 
              << " (" << correct << "/" << total << ")" << std::endl;

    return 0;
}