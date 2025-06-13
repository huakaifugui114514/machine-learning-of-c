// mse_loss.hpp
#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"

namespace dlt {
namespace loss {

class MSELoss {
public:
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, 
                                   const std::shared_ptr<Tensor>& target);
    std::vector<std::shared_ptr<Tensor>> backward();

private:
    std::shared_ptr<Tensor> input_;
    std::shared_ptr<Tensor> target_;
    std::shared_ptr<Tensor> diff_;
};

} // namespace loss
} // namespace dlt