#ifndef NN_LINEAR_HPP
#define NN_LINEAR_HPP

#include "module.hpp"
#include <memory>

namespace dlt {
namespace nn {

// 全连接层
class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true);
    
    TensorPtr forward(const TensorPtr& x) override;
    std::vector<TensorPtr> parameters() const override;
    std::string name() const override { return "Linear"; }
    
private:
    TensorPtr weight_;
    TensorPtr bias_;
    bool has_bias_;
    int in_features_;
    int out_features_;
};

} // namespace nn
} // namespace dlt

#endif // NN_LINEAR_HPP    