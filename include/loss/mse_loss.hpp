#ifndef LOSS_MSE_LOSS_HPP
#define LOSS_MSE_LOSS_HPP

#include "loss.hpp"

namespace dlt {
namespace loss {

class MSELoss : public LossFunction {
public:
    TensorPtr forward(const TensorPtr& input, const TensorPtr& target) override;
    std::vector<TensorPtr> backward() override;
    
private:
    TensorPtr input_;
    TensorPtr target_;
    TensorPtr diff_; 
};

} // namespace loss
} // namespace dlt

#endif // LOSS_MSE_LOSS_HPP