#pragma once

#include "neuro/interfaces/layer/i_layer.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

class DenseLayer : public ILayer {
  layer_weight_t weights{};
  layer_bias_t biases{};

  ActivationFunction activation;

 public:
  DenseLayer() = default;
  DenseLayer(const DenseLayer&) = default;

  DenseLayer(int inputSize, int outputSize, ActivationFunction& activation);
  DenseLayer(const layer_weight_t& weights, ActivationFunction& activation);
  DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, ActivationFunction& activation);

  virtual ~DenseLayer() = default;
};

};  // namespace neuro
