#ifndef LOSS_LOSS_HPP
#define LOSS_LOSS_HPP

#include "../tensor.hpp"
#include <memory>

namespace dlt {
namespace loss {

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual TensorPtr forward(const TensorPtr& input, const TensorPtr& target) = 0;
    virtual std::vector<TensorPtr> backward() = 0;
};

} // namespace loss
} // namespace dlt

#endif // LOSS_LOSS_HPP