#pragma once

#include <random>
#include <vector>

#include "neuro/utils/activation.hpp"
#include "neuro/utils/interfaces/i_iterable.hpp"

namespace neuro {

typedef std::vector<float> neuro_layer_t;

typedef std::vector<neuro_layer_t> layer_weight_t;
typedef neuro_layer_t layer_bias_t;

class ILayer {
 public:
  ILayer(ILayer&) = default;
  virtual ~ILayer() = default;

  virtual std::vector<float> feedforward(const neuro_layer_t& input) = 0;

  virtual void randomizeWeights(float max = 1.0f) {
    randomizeWeights(-max, max);
  }

  virtual void randomizeBiases(float max = 1.0f) {
    randomizeBiases(-max, max);
  }

  virtual void randomizeWeights(float min, float max) = 0;
  virtual void randomizeBiases(float min, float max) = 0;

  virtual void setActivationFunction(const ActivationFunction&) = 0;

  virtual void setWeights(const layer_weight_t&) = 0;
  virtual void setBiases(const layer_bias_t&) = 0;

  virtual void setWeight(int indexX, int indexY, float value) = 0;
  virtual void setBias(int index, float value) = 0;

  virtual const layer_weight_t& getWeights() const = 0;
  virtual const layer_bias_t& getBiases() const = 0;

  virtual layer_weight_t& getWeights() = 0;
  virtual layer_bias_t& getBiases() = 0;

  virtual float getWeight(int indexX, int indexY) const = 0;
  virtual float getBias(int index) const = 0;

  virtual size_t inputSize() const = 0;
  virtual size_t outputSize() const = 0;

  virtual ILayer& operator=(const ILayer&) = default;
};

};  // namespace neuro
