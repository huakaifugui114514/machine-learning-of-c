#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <functional>
#include <string>

namespace dlt {

class Tensor;
class Function;

using TensorPtr = std::shared_ptr<Tensor>;
using FunctionPtr = std::shared_ptr<Function>;

// Tensor类 - 框架的核心数据结构
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad = false);
    Tensor(std::vector<float>&& data, const std::vector<int>& shape, bool requires_grad = false);
    
    // 基本属性访问
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& grad() const { return grad_; }
    std::vector<float>& grad() { return grad_; }
    const std::vector<int>& shape() const { return shape_; }
    int size() const { return size_; }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    // 操作符重载
    TensorPtr operator-(const TensorPtr& other) ;
    TensorPtr operator+(const TensorPtr& other) const;
    TensorPtr operator-(const TensorPtr& other) const;
    TensorPtr operator*(const TensorPtr& other) const;
    
    // 常用操作
    TensorPtr matmul(const TensorPtr& other) const;
    TensorPtr relu() const;
    TensorPtr sum() const;
    
    // 自动微分相关
    void backward(const TensorPtr& grad_output = nullptr);
    void zero_grad();
    
    // 辅助函数
    void print() const;
    
protected:
    // 设置梯度函数和子节点（供Function类使用）
    void set_grad_fn(const FunctionPtr& grad_fn) { grad_fn_ = grad_fn; }
    void add_child(const TensorPtr& child) { children_.push_back(child); }
    void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }

private:
    std::vector<float> data_;       // 张量数据
    std::vector<float> grad_;       // 梯度数据
    std::vector<int> shape_;        // 张量形状
    int size_;                      // 张量元素总数
    bool requires_grad_;            // 是否需要计算梯度
    FunctionPtr grad_fn_;           // 计算梯度的函数
    std::vector<TensorPtr> children_; // 子张量（用于构建计算图）
    
    // 用于标记该张量是否是叶子节点（用户创建的张量）
    bool is_leaf_;
    
    // 友元类可以访问私有成员
    friend class Function;
    friend class AddFunction;
    friend class MulFunction;
    friend class MatMulFunction;
    friend class ReLUFunction;
    friend class SumFunction;
};

// 工厂函数，用于创建张量
TensorPtr tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad = false);
TensorPtr zeros(const std::vector<int>& shape, bool requires_grad = false);
TensorPtr ones(const std::vector<int>& shape, bool requires_grad = false);
TensorPtr randn(const std::vector<int>& shape, bool requires_grad = false);

} // namespace dlt

#endif // TENSOR_HPP    