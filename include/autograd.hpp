#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include "tensor.hpp"
#include <memory>
#include <vector>
#include <string>

namespace dlt {

// 自动微分基类
class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;
    
    // 前向传播
    virtual TensorPtr apply(const std::vector<TensorPtr>& inputs) = 0;
    
    // 反向传播
    virtual std::vector<TensorPtr> backward(const TensorPtr& grad_output) = 0;
    
    // 获取函数名称
    virtual std::string name() const = 0;
    
    // 输入和输出张量
    std::vector<TensorPtr> inputs_;
    TensorPtr output_;
};

// 加法操作的自动微分
class AddFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "AddFunction"; }
};


// 乘法操作的自动微分
class MulFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MulFunction"; }
};

// 矩阵乘法操作的自动微分
class MatMulFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MatMulFunction"; }
};

// ReLU操作的自动微分
class ReLUFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ReLUFunction"; }
};

// 求和操作的自动微分
class SumFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SumFunction"; }
};



} // namespace dlt

#endif // AUTOGRAD_HPP    