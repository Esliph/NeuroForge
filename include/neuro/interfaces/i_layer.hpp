#pragma once

#include <memory>
#include <random>
#include <vector>

#include "neuro/utils/activation.hpp"
#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

  typedef std::vector<float> neuro_layer_t;

  typedef std::vector<neuro_layer_t> layer_weight_t;
  typedef neuro_layer_t layer_bias_t;

  class ILayer {
   public:
    ILayer() = default;
    virtual ~ILayer() = default;

    virtual neuro_layer_t feedforward(const neuro_layer_t& inputs) const = 0;

    virtual void reset() = 0;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

    virtual RefProxy<float> weight(size_t indexX, size_t indexY) = 0;
    virtual RefProxy<float> bias(size_t index) = 0;

    virtual void setActivationFunction(const ActivationFunction&) = 0;

    virtual void setWeights(const layer_weight_t&) = 0;
    virtual void setBiases(const layer_bias_t&) = 0;

    virtual void setWeight(size_t indexX, size_t indexY, float value) = 0;
    virtual void setBias(size_t index, float value) = 0;

    virtual const layer_weight_t& getWeights() const = 0;
    virtual const layer_bias_t& getBiases() const = 0;

    virtual layer_weight_t& getWeights() = 0;
    virtual layer_bias_t& getBiases() = 0;

    virtual const ActivationFunction& getActivationFunction() const = 0;

    virtual std::unique_ptr<ILayer> clone() const = 0;
  };

};  // namespace neuro
