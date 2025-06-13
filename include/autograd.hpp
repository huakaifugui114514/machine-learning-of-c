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

TensorPtr reduce_grad(const TensorPtr& grad_output, const std::vector<int>& target_shape);

// ================== 基本数学运算 ==================
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

class MulFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MulFunction"; }
};

class MatMulFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MatMulFunction"; }
};

// ================== 激活函数 ==================
class ReLUFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ReLUFunction"; }
};

class SigmoidFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SigmoidFunction"; }
};

class TanhFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "TanhFunction"; }
};

class SoftmaxFunction : public Function {
public:
    SoftmaxFunction(int dim = -1) : dim_(dim) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SoftmaxFunction"; }

private:
    int dim_; // Softmax计算的维度
};

// ================== 张量操作 ==================
class SumFunction : public Function {
public:
    SumFunction(int dim = -1, bool keepdims = false) 
        : dim_(dim), keepdims_(keepdims) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SumFunction"; }

private:
    int dim_;        // 求和的维度
    bool keepdims_;  // 是否保持维度
};

class ExpFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ExpFunction"; }
};

class LogFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "LogFunction"; }
};

class SinFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SinFunction"; }
};

class CosFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "CosFunction"; }
};

class TanFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "TanFunction"; }
};

class MaxFunction : public Function {
public:
    MaxFunction(int dim = -1, bool keepdims = false) 
        : dim_(dim), keepdims_(keepdims) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MaxFunction"; }

private:
    int dim_;
    bool keepdims_;
    std::vector<size_t> argmax_indices_; // 存储最大值的索引
};

class MinFunction : public Function {
public:
    MinFunction(int dim = -1, bool keepdims = false) 
        : dim_(dim), keepdims_(keepdims) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MinFunction"; }

private:
    int dim_;
    bool keepdims_;
    std::vector<size_t> argmin_indices_; // 存储最小值的索引
};

class MeanFunction : public Function {
public:
    MeanFunction(int dim = -1, bool keepdims = false) 
        : dim_(dim), keepdims_(keepdims) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MeanFunction"; }

private:
    int dim_;
    bool keepdims_;
};

class ReshapeFunction : public Function {
public:
    ReshapeFunction(const std::vector<int>& new_shape) : new_shape_(new_shape) {}
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ReshapeFunction"; }
    
private:
    std::vector<int> new_shape_;
};

class TransposeFunction : public Function {
public:
    TransposeFunction(int dim0 = 0, int dim1 = 1) 
        : dim0_(dim0), dim1_(dim1) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "TransposeFunction"; }

private:
    int dim0_;
    int dim1_;
};

class ConcatFunction : public Function {
public:
    ConcatFunction(int dim = 0) : dim_(dim) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ConcatFunction"; }

private:
    int dim_;
};

class SplitFunction : public Function {
public:
    SplitFunction(int dim, int sections, int index) 
        : dim_(dim), sections_(sections), index_(index) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "SplitFunction"; }

private:
    int dim_;
    int sections_;
    int index_; // 当前分割部分的索引
    size_t slice_index_; // 当前分割部分的起始位置
};

class DotFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "DotFunction"; }
};

class AbsFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "AbsFunction"; }
};

class ContiguousFunction : public Function {
public:
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ContiguousFunction"; }
};

class ExpandFunction : public Function {
public:
    ExpandFunction(const std::vector<int>& new_shape) 
        : new_shape_(new_shape) {}
    
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "ExpandFunction"; }

private:
    std::vector<int> new_shape_;
};

// ================== 神经网络操作 ==================
class AvgPool2dFunction : public Function {
public:
    AvgPool2dFunction(int kernel_size, int stride, int padding);
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "AvgPool2dFunction"; }
    
private:
    int kernel_size_;
    int stride_;
    int padding_;
    std::vector<int> input_shape_; // 用于存储前向传播时的输入形状
};

class MaxPool2dFunction : public Function {
public:
    MaxPool2dFunction(int kernel_size, int stride, int padding);
    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "MaxPool2dFunction"; }
    
private:
    int kernel_size_;
    int stride_;
    int padding_;
    std::vector<int> input_shape_; // 用于存储前向传播时的输入形状
    std::vector<int> argmax_indices_; // 用于存储最大值的索引
};

class Conv2dFunction : public Function {
public:
    Conv2dFunction(int in_channels, int out_channels, int kernel_size, int stride, int padding)
        : in_channels_(in_channels), out_channels_(out_channels), 
          kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

    TensorPtr apply(const std::vector<TensorPtr>& inputs) override;
    std::vector<TensorPtr> backward(const TensorPtr& grad_output) override;
    std::string name() const override { return "Conv2dFunction"; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    std::vector<int> input_shape_; // 用于存储前向传播时的输入形状
};

} // namespace dlt

#endif // AUTOGRAD_HPP