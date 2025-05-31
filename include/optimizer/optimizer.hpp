#ifndef OPTIMIZER_OPTIMIZER_HPP
#define OPTIMIZER_OPTIMIZER_HPP

#include "../tensor.hpp"
#include <vector>
#include <memory>

namespace dlt {
namespace optimizer {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual void add_parameters(const std::vector<TensorPtr>& params) = 0;
};

} // namespace optimizer
} // namespace dlt

#endif // OPTIMIZER_OPTIMIZER_HPP