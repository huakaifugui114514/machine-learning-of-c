#ifndef OPTIMIZER_SGD_HPP
#define OPTIMIZER_SGD_HPP

#include "optimizer.hpp"

namespace dlt {
namespace optimizer {

class SGD : public Optimizer {
public:
    SGD(float lr = 0.01, float momentum = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameters(const std::vector<TensorPtr>& params) override;
    
private:
    std::vector<TensorPtr> parameters_;
    float lr_;
    float momentum_;
    std::vector<std::vector<float>> velocities_;
};

} // namespace optimizer
} // namespace dlt

#endif // OPTIMIZER_SGD_HPP