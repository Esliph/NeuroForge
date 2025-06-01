#pragma once

#include <vector>

#include "neuro/activation.hpp"
#include "neuro/layer.hpp"

namespace neuro {

class NeuralNetwork {
  std::vector<Layer> layers;

 public:
  NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork&) = default;
  NeuralNetwork(const std::vector<Layer>& layers);
  NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation);
  NeuralNetwork(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

  inline void loadWeights(float range = 1.0f) { return loadWeights(-range, range); }
  void loadWeights(float rangeMin, float rangeMax);

  inline void loadBias(float range = 1.0f) { return loadBias(-range, range); }
  void loadBias(float rangeMin, float rangeMax);

  neuro_layer_t feedforward(const neuro_layer_t& input) const;

  NeuralNetwork& operator=(const NeuralNetwork&) = default;

  const std::vector<Layer>&
  getLayers() const;
};

};  // namespace neuro
