#include "tensor.hpp"
#include "autograd.hpp"
#include "ops.hpp"
#include <iostream>
#include <random>
#include <numeric>

namespace dlt {

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, bool requires_grad)
    : data_(data), shape_(shape), requires_grad_(requires_grad), is_leaf_(true) {
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
    }
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
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
    }
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

void Tensor::backward(const TensorPtr& grad_output) {
    // 步骤1：创建或验证梯度张量
    TensorPtr grad_tensor;
    if (grad_output == nullptr) {
        if (size_ != 1) throw std::invalid_argument("grad_output must not be null for non-scalar tensor");
        grad_tensor = std::make_shared<Tensor>(
            std::vector<float>{1.0f}, std::vector<int>{1}, false
        );
    } else {
        // 验证梯度形状匹配
        if (grad_output->size() != size_) throw std::invalid_argument("grad_output size does not match tensor size");
        grad_tensor = grad_output;
    }

    // 步骤2：正确累加梯度
    if (grad_.empty()) {
        grad_ = grad_tensor->data();  // 初始化梯度
    } else {
        const auto& grad_data = grad_tensor->data();
        for (size_t i = 0; i < grad_.size(); ++i) {
            grad_[i] += grad_data[i];  // 累加梯度
        }
    }

    // 步骤3：执行反向传播
    if (grad_fn_) {
        // 传递正确的梯度张量
        auto grads = grad_fn_->backward(grad_tensor);
        
        for (size_t i = 0; i < children_.size(); ++i) {
            if (children_[i]->requires_grad() && grads[i]) {
                children_[i]->backward(grads[i]);  // 递归反向传播
            }
        }
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