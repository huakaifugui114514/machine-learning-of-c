#include <gtest/gtest.h>
#include "data/data_loader.hpp"
#include <filesystem>
#include <fstream>
#include "../examples/CNN_example.cpp" //使用该测试时，需要先注释CNN_examples.cpp的main()。 

namespace fs = std::filesystem;

// 测试 load_mnist 函数
TEST(MNISTLoaderTest, LoadMNIST) {
    std::string image_file = "../resources/mnist/train-images-idx3-ubyte/train-images.idx3-ubyte";
    std::string label_file = "../resources/mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte";

    // 检查文件是否存在
    if (!fs::exists(image_file)) {
        std::cerr << "Image file not found: " << image_file << std::endl;
    }
    if (!fs::exists(label_file)) {
        std::cerr << "Label file not found: " << label_file << std::endl;
    }

    // 调用 load_mnist 函数
    auto [images, labels] = load_mnist(image_file, label_file);

    // 验证图像和标签的数量是否一致
    EXPECT_EQ(images.size(), labels.size());

    // 验证图像和标签的数量是否大于 0
    EXPECT_GT(images.size(), 0);

    // 遍历部分图像和标签进行更详细的验证
    for (size_t i = 0; i < std::min(images.size(), static_cast<size_t>(10)); ++i) {
        const auto& image = images[i];
        const auto& label = labels[i];

        // 验证图像的形状是否正确
        EXPECT_EQ(image->shape().size(), 4);
        EXPECT_EQ(image->shape()[0], 1);
        EXPECT_EQ(image->shape()[1], 1);
        EXPECT_EQ(image->shape()[2], 28);
        EXPECT_EQ(image->shape()[3], 28);

        // 验证图像数据的范围是否在 [0, 1] 之间
        const auto& image_data = image->data();
        for (float pixel : image_data) {
            EXPECT_GE(pixel, 0.0f);
            EXPECT_LE(pixel, 1.0f);
        }

        // 验证标签的形状是否正确
        EXPECT_EQ(label->shape().size(), 2);
        EXPECT_EQ(label->shape()[0], 1);
        EXPECT_EQ(label->shape()[1], 10);

        // 验证标签数据是否为 one-hot 编码
        const auto& label_data = label->data();
        int one_count = 0;
        for (float value : label_data) {
            EXPECT_TRUE(value == 0.0f || value == 1.0f);
            if (value == 1.0f) {
                one_count++;
            }
        }
        EXPECT_EQ(one_count, 1);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}