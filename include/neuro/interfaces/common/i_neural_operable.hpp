#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_layer.hpp"

namespace neuro {

  class INeuralOperable {
   public:
    INeuralOperable() = default;
    virtual ~INeuralOperable() = default;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual neuro_layer_t feedforward(const neuro_layer_t& inputs) const = 0;

    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

    virtual void setAllWeights(const std::vector<layer_weight_t>&) = 0;
    virtual void setAllBiases(const std::vector<layer_bias_t>&) = 0;

    virtual void setLayers(std::vector<std::unique_ptr<ILayer>>) = 0;

    virtual std::vector<layer_weight_t> getAllWeights() const = 0;
    virtual std::vector<layer_bias_t> getAllBiases() const = 0;

    virtual size_t sizeLayers() const = 0;

    virtual std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>::iterator begin() = 0;

    virtual std::vector<std::unique_ptr<ILayer>>::const_iterator end() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>::iterator end() = 0;

    virtual neuro_layer_t operator()(const neuro_layer_t& inputs) const = 0;

    virtual const ILayer& operator[](size_t index) const = 0;
    virtual ILayer& operator[](size_t index) = 0;
  };

};  // namespace neuro
