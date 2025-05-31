#ifndef NN_CONV2D_HPP
#define NN_CONV2D_HPP

#include "module.hpp"
#include <memory>

namespace dlt {
namespace nn {

class Conv2d : public Module {
public:
    Conv2d(int in_channels, int out_channels, 
           int kernel_size, int stride = 1, 
           int padding = 0, bool bias = true);
    
    TensorPtr forward(const TensorPtr& x) override;
    std::vector<TensorPtr> parameters() const override;
    std::string name() const override { return "Conv2d"; }
    
private:
    TensorPtr weight_;
    TensorPtr bias_;
    bool has_bias_;
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    
};

} // namespace nn
} // namespace dlt

#endif // NN_CONV2D_HPP