cmake_minimum_required(VERSION 3.10)
project(DeepLearningFramework)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 设置调试信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

# 设置链接库的搜索路径(系统默认的)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/lib/x86_64-linux-gnu")

# 查找 Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# 添加头文件路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include   # 项目头文件目录
    ${CMAKE_SOURCE_DIR}/third_party
)

# 收集源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB TEST_FILES "tests/*.cpp")
file(GLOB EXAMPLE_FILES "examples/*.cpp")


# 示例编译
foreach(example_file ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file} ${SOURCES})
endforeach()

# # 测试编译 
# foreach(test_file ${TEST_FILES})
#     get_filename_component(test_name ${test_file} NAME_WE)
#     add_executable(${test_name} ${test_file} ${SOURCES})
#     target_link_libraries(${test_name} ${GTEST_LIBRARIES} pthread)
#     add_test(NAME ${test_name} COMMAND ${test_name})
# endforeach()

# enable_testing()