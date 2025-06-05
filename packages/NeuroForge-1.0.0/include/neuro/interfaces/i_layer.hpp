#pragma once

#include <random>
#include <vector>

#include "neuro/activation.hpp"

namespace neuro {

typedef std::vector<float> neuro_layer_t;

typedef std::vector<neuro_layer_t> layer_weight_t;
typedef neuro_layer_t layer_bias_t;

class ILayer {
 public:
  ILayer(ILayer&) = default;
  virtual ~ILayer() = default;

  virtual std::vector<float> feedforward(const neuro_layer_t& input) = 0;

  virtual void mutate(float rate, float intensity) = 0;
  virtual ILayer crossover(const ILayer& partner) const = 0;

  virtual void randomizeWeights(float max = 1.0f) {
    randomizeWeights(-max, max);
  }

  virtual void randomizeBias(float max = 1.0f) {
    randomizeBias(-max, max);
  }

  virtual void randomizeWeights(float min, float max) = 0;
  virtual void randomizeBias(float min, float max) = 0;

  virtual void setActivationFunction(const ActivationFunction&) = 0;

  virtual void setWeights(const layer_weight_t&) = 0;
  virtual void setBias(const layer_bias_t&) = 0;

  virtual const layer_weight_t& getWeights() const = 0;
  virtual const layer_bias_t& getBias() const = 0;

  virtual size_t inputSize() const = 0;
  virtual size_t outputSize() const = 0;

  virtual ILayer& operator=(const ILayer&) = default;
};

};  // namespace neuro
