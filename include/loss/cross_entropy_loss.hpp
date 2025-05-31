#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include "loss.hpp"

namespace dlt {
namespace loss {

class CrossEntropyLoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& input, const TensorPtr& target) override;
    std::vector<TensorPtr> backward() override;
    
private:
    TensorPtr input_;
    TensorPtr target_;
    TensorPtr softmax_output_;
};

} // namespace loss
} // namespace dlt

#endif // CROSS_ENTROPY_LOSS_HPP