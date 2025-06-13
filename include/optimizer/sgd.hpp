// sgd.hpp
#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"

namespace dlt {
namespace optimizer {

class SGD {
public:
    // 动量参数可选
    SGD(float lr, float momentum = 0.0f, float clip_value = 1.0f);
    void add_parameters(const std::vector<std::shared_ptr<Tensor>>& params);
    void step();
    void zero_grad();

private:
    float lr_;
    float momentum_;
    float clip_value_;
    std::vector<std::shared_ptr<Tensor>> parameters_;
    std::vector<std::vector<float>> velocities_;
};

} // namespace optimizer
} // namespace dlt