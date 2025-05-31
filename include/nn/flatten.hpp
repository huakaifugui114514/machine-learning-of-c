#ifndef NN_FLATTEN_HPP
#define NN_FLATTEN_HPP

#include "module.hpp"
#include <memory>

namespace dlt {
namespace nn {

class Flatten : public Module {
public:
    Flatten();
    TensorPtr forward(const TensorPtr& x) override;
    std::vector<TensorPtr> parameters() const override;
    std::string name() const override { return "Flatten"; }
};

} // namespace nn
} // namespace dlt

#endif // NN_FLATTEN_HPP