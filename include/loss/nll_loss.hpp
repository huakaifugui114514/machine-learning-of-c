#ifndef NLL_LOSS_HPP
#define NLL_LOSS_HPP

#include "loss.hpp"

namespace dlt {
namespace loss {

class NLLLoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& input, const TensorPtr& target) override;
    std::vector<TensorPtr> backward() override;

private:
    TensorPtr input_;
    TensorPtr target_;
};

} // namespace loss
} // namespace dlt

#endif // NLL_LOSS_HPP