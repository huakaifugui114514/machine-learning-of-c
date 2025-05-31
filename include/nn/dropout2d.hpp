#ifndef NN_DROPOUT2D_HPP
#define NN_DROPOUT2D_HPP

#include "module.hpp"
#include <memory>

namespace dlt {
namespace nn {

class Dropout2d : public Module {
public:
    Dropout2d(float p = 0.5);
    TensorPtr forward(const TensorPtr& x) override;
    std::vector<TensorPtr> parameters() const override;
    std::string name() const override { return "Dropout2d"; }

private:
    float p_;
};

} // namespace nn
} // namespace dlt

#endif // NN_DROPOUT2D_HPP