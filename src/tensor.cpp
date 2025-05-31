#include "tensor.hpp"
#include "autograd.hpp"
#include "ops.hpp"
#include <iostream>
#include <random>
#include <numeric>

namespace dlt {

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad)
    : data_(data), shape_(shape), requires_grad_(requires_grad), is_leaf_(true) {
    size_ = 1;
    for (int dim : shape_) {
        size_ *= dim;
    }
    grad_.resize(size_, 0.0f);
    
    // 确保数据大小与形状一致
    if (data_.size() != static_cast<size_t>(size_)) {
        throw std::invalid_argument("Data size does not match shape");
    }
}

Tensor::Tensor(std::vector<float>&& data, const std::vector<int>& shape, bool requires_grad)
    : data_(std::move(data)), shape_(shape), requires_grad_(requires_grad), is_leaf_(true) {
    size_ = 1;
    for (int dim : shape_) {
        size_ *= dim;
    }
    grad_.resize(size_, 0.0f);
    
    // 确保数据大小与形状一致
    if (data_.size() != static_cast<size_t>(size_)) {
        throw std::invalid_argument("Data size does not match shape");
    }
}




TensorPtr Tensor::operator+(const TensorPtr& other) const {
    return ops::add(const_cast<Tensor*>(this)->shared_from_this(), other);// const 转换为 non-const， 以符合函数签名要求
}

TensorPtr Tensor::operator*(const TensorPtr& other) const {
    return ops::mul(const_cast<Tensor*>(this)->shared_from_this(), other);
}

TensorPtr Tensor::matmul(const TensorPtr& other) const {
    return ops::matmul(const_cast<Tensor*>(this)->shared_from_this(), other);
}

TensorPtr Tensor::relu() const {
    return ops::relu(const_cast<Tensor*>(this)->shared_from_this());
}

TensorPtr Tensor::sum() const {
    return ops::sum(const_cast<Tensor*>(this)->shared_from_this());
}

void Tensor::backward(const TensorPtr& grad_output) {
    // 如果没有提供梯度输出，假设是标量损失
    if (grad_output == nullptr) {
        if (size_ != 1) {
            throw std::invalid_argument("backward should be called only on scalar tensors when grad_output is not provided");
        }
        grad_ = {1.0f};
    } else {
        grad_ = grad_output->data();
    }
    
    // 执行反向传播
    if (grad_fn_) {
        grad_fn_->backward(shared_from_this());
    }
}

void Tensor::zero_grad() {
    std::fill(grad_.begin(), grad_.end(), 0.0f);
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Tensor data: [";
    for (size_t i = 0; i < data_.size(); ++i) {
        std::cout << data_[i];
        if (i < data_.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    if (requires_grad_) {
        std::cout << "Tensor grad: [";
        for (size_t i = 0; i < grad_.size(); ++i) {
            std::cout << grad_[i];
            if (i < grad_.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

// 工厂函数实现
TensorPtr tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad) {
    return std::make_shared<Tensor>(data, shape, requires_grad);
}

TensorPtr zeros(const std::vector<int>& shape, bool requires_grad) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    return std::make_shared<Tensor>(std::vector<float>(size, 0.0f), shape, requires_grad);
}

TensorPtr ones(const std::vector<int>& shape, bool requires_grad) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    return std::make_shared<Tensor>(std::vector<float>(size, 1.0f), shape, requires_grad);
}

TensorPtr randn(const std::vector<int>& shape, bool requires_grad) {
    int size = 1;
    for (int dim : shape) {
        size *= dim;
    }
    
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    
    return std::make_shared<Tensor>(data, shape, requires_grad);
}

} // namespace dlt    