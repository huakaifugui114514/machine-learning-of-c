echo "切换到build 目录"
cd build
echo "运行gdb"
gdb ./CNN_example
echo "输入c命令继续运行程序"
c
echo "输入run命令运行程序"
run
echo "获取backstrace"
bt
