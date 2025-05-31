// pj/include/optimizer/adam.hpp
#ifndef ADAM_HPP
#define ADAM_HPP

#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace dlt {
namespace optimizer {

class Adam {
public:
    Adam(float lr = 0.001f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f,
         // 减小 eps_ 的值
         float eps = 1e-10f); 
    
    void add_parameters(const std::vector<TensorPtr>& params);
    void step();
    void zero_grad();
    // 添加 get_parameters 函数声明
    const std::vector<TensorPtr>& get_parameters() const; 

private:
    float lr_;     // 学习率
    float beta1_;  // 一阶矩估计的指数衰减率
    float beta2_;  // 二阶矩估计的指数衰减率
    float eps_;    // 数值稳定性常数
    
    std::vector<TensorPtr> parameters_;
    std::vector<std::vector<float>> m_;  // 一阶矩估计
    std::vector<std::vector<float>> v_;  // 二阶矩估计
    int t_;  
};

} // namespace optimizer
} // namespace dlt

#endif // ADAM_HPP