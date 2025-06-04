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

class SubFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SubFunction"; }
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

// Exp操作的自动微分
class ExpFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ExpFunction"; }
};

// Log操作的自动微分
class LogFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "LogFunction"; }
};

// Sin操作的自动微分
class SinFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SinFunction"; }
};

// Cos操作的自动微分
class CosFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "CosFunction"; }
};

// Tan操作的自动微分
class TanFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "TanFunction"; }
};

// Max操作的自动微分
class MaxFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MaxFunction"; }
};

// Min操作的自动微分
class MinFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MinFunction"; }
};

// Max按维度操作的自动微分
class MaxDimFunction : public Function {
public:
    MaxDimFunction(int dim) : dim_(dim) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MaxDimFunction"; }
private:
    int dim_;
};

// Min按维度操作的自动微分
class MinDimFunction : public Function {
public:
    MinDimFunction(int dim) : dim_(dim) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MinDimFunction"; }
private:
    int dim_;
};

// Mean操作的自动微分
class MeanFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MeanFunction"; }
};

// Mean按维度操作的自动微分
class MeanDimFunction : public Function {
public:
    MeanDimFunction(int dim) : dim_(dim) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MeanDimFunction"; }
private:
    int dim_;
};

// Reshape操作的自动微分
class ReshapeFunction : public Function {
public:
    ReshapeFunction(const std::vector<int>& new_shape) : new_shape_(new_shape) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ReshapeFunction"; }
private:
    std::vector<int> new_shape_;
};

// Transpose操作的自动微分
class TransposeFunction : public Function {
public:
    TransposeFunction(int dim0, int dim1) : dim0_(dim0), dim1_(dim1) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "TransposeFunction"; }
private:
    int dim0_;
    int dim1_;
};

// Concat操作的自动微分
class ConcatFunction : public Function {
public:
    ConcatFunction(int dim) : dim_(dim) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ConcatFunction"; }
private:
    int dim_;
};

// Split操作的自动微分
class SplitFunction : public Function {
public:
    SplitFunction(int dim, int sections) : dim_(dim), sections_(sections) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SplitFunction"; }
private:
    int dim_;
    int sections_;
};

// Dot操作的自动微分
class DotFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "DotFunction"; }
};

// Abs操作的自动微分
class AbsFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "AbsFunction"; }
};

// ContiguousFunction 类
class ContiguousFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ContiguousFunction"; }
};

// Sigmoid操作的自动微分
class SigmoidFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SigmoidFunction"; }
};

// Tanh操作的自动微分
class TanhFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "TanhFunction"; }
};

// Softmax操作的自动微分（带维度参数）
class SoftmaxFunction : public Function {
public:
    SoftmaxFunction(int dim = -1) : dim_(dim) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SoftmaxFunction"; }
private:
    int dim_;  // 归一化的维度
};

} // namespace dlt

#endif // AUTOGRAD_HPP