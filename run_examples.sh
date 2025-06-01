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

# 运行 example_usage 目录下生成的可执行文件
echo -e "${YELLOW}运行示例代码...${NC}"
for file in ./*; do
    if [ -x "$file" ] && [ ! -d "$file" ]; then  # 检查文件是否为可执行文件且不是目录
        echo -e "${YELLOW}运行 $file...${NC}"
        ./$file
        if [ $? -ne 0 ]; then
            echo -e "${RED}$file 运行失败！${NC}"
            exit 1
        fi
    fi
done

echo -e "${GREEN}==== 项目构建和示例代码运行完成 ====${NC}"