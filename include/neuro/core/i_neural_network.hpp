#pragma once

#include <memory>
#include <vector>

namespace neuro {

  class INeuralNetwork {
   public:
    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual std::vector<float> feedforward(const std::vector<float> inputs) const = 0;

    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

    virtual std::unique_ptr<INeuralNetwork> clone() const = 0;
  };

} // namespace neuro
