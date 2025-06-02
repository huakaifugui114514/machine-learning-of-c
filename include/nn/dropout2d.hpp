#ifndef NN_DROPOUT2D_HPP
#define NN_DROPOUT2D_HPP

#include "module.hpp"
#include <memory>

namespace dlt {
namespace nn {

class Dropout2d : public Module {
public:
    Dropout2d(float p = 0.5);
    TensorPtr forward(const TensorPtr& x) override;
    std::vector<TensorPtr> parameters() const override;
    std::string name() const override { return "Dropout2d"; }

    // 显式覆盖基类函数
    void train() override { is_training_ = true; }
    void eval() override { is_training_ = false; }

private:
    float p_;
    using Module::is_training_; // 显式声明访问基类成员
};

} // namespace nn
} // namespace dlt

#endif // NN_DROPOUT2D_HPP