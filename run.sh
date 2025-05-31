#!/bin/bash

set -e  # 如果任何命令失败，立即退出

# 颜色定义，用于美化输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

echo -e "${YELLOW}==== 开始构建深度学习框架 ====${NC}"

# 创建构建目录（如果不存在）
if [ ! -d "build" ]; then
    mkdir build
    echo -e "${GREEN}创建 build 目录${NC}"
fi

cd build

# 清理旧的构建文件
echo -e "${YELLOW}清理旧的构建文件...${NC}"
rm -rf *

# 运行 CMake 配置
echo -e "${YELLOW}运行 CMake 配置...${NC}"
cmake ..

# 编译项目
echo -e "${YELLOW}开始编译...${NC}"
make

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo -e "${GREEN}编译成功！${NC}"
else
    echo -e "${RED}编译失败！${NC}"
    exit 1
fi

# 运行测试
echo -e "${YELLOW}运行测试...${NC}"
echo -e "${YELLOW}运行adam测试...${NC}"./test_adam
./test_adam

echo -e "${YELLOW}运行conv_transpose测试...${NC}"./test_conv_transpose
./test_conv_transpose

echo -e "${YELLOW}运行conv测试...${NC}"./test_conv
./test_conv

echo -e "${YELLOW}运行cross_entropy_loss测试...${NC}"./test_cross_entropy_loss
./test_cross_entropy_loss

echo -e "${YELLOW}运行data_loader测试...${NC}"./test_data_loader     
./test_data_loader

echo -e "${YELLOW}运行dropout2d测试...${NC}"./test_dropout2d
./test_dropout2d

echo -e "${YELLOW}运行flatten测试...${NC}"./test_flatten
./test_flatten

echo -e "${YELLOW}运行linear测试...${NC}"./test_linear
./test_linear

echo -e "${YELLOW}运行mseloss测试...${NC}"./test_mseloss
./test_mseloss

echo -e "${YELLOW}运行nll_loss测试...${NC}"./test_nll_loss  
./test_nll_loss

echo -e "${YELLOW}运行sgd测试...${NC}"./test_sgd    
./test_sgd

echo -e "${YELLOW}运行tensor测试...${NC}"./test_tensor
./test_tensor


# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo -e "${GREEN}所有测试通过！${NC}"
else
    echo -e "${RED}测试失败！${NC}"
    exit 1
fi

echo -e "${GREEN}==== 项目构建和测试完成 ====${NC}"