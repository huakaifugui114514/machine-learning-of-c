# DeepLearningFramework

`DeepLearningFramework` 是一个基于 C++ 开发的深度学习框架，它实现了一些基础的深度学习功能，如全连接层、自动微分、优化器和损失函数等。

## 环境要求
- **操作系统**：建议使用 Linux 系统（如 Ubuntu），因为代码中使用了一些 Linux 特定的脚本和库。
- **编译器**：支持 C++17 标准的编译器，如 GCC 7 及以上版本。
- **CMake**：版本 3.10 及以上，用于项目的构建和配置。
- **Google Test**：用于单元测试，需要安装并配置好该库。

## 构建与测试步骤
### 1. 运行脚本
运行 `run_tests.sh` 脚本可以自动完成tests项目的构建和测试过程。
```bash
./run_tests.sh

运行 `run_examples.sh` 脚本可以自动完成examples项目的构建和测试过程。
```bash
./run_examples.sh