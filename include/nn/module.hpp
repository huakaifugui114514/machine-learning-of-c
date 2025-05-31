#ifndef NN_MODULE_HPP
#define NN_MODULE_HPP

#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>

namespace dlt {
namespace nn {

// 模块基类
class Module {
public:
    virtual ~Module() = default;
    
    // 前向传播
    virtual TensorPtr forward(const TensorPtr& x) = 0;
    
    // 获取所有参数
    virtual std::vector<TensorPtr> parameters() const = 0;
    
    // 设置为训练模式
    virtual void train() { is_training_ = true; }
    
    // 设置为评估模式
    virtual void eval() { is_training_ = false; }
    
    // 获取模块名称
    virtual std::string name() const = 0;
    
protected:
    bool is_training_ = true;
};

using ModulePtr = std::shared_ptr<Module>;

} // namespace nn
} // namespace dlt

#endif // NN_MODULE_HPP    