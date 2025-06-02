#pragma once

#include <random>
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

  inline void loadWeights(float range = 1.0f) {
    loadWeights(-range, range);
  }

  void loadWeights(float rangeMin, float rangeMax);

  inline void loadBias(float range = 1.0f) {
    loadBias(-range, range);
  }

  void loadBias(float rangeMin, float rangeMax);

  neuro_layer_t feedforward(const neuro_layer_t& input) const;

  void mutate(float rate, float strength);
  NeuralNetwork crossover(const NeuralNetwork& partner) const;

  void mutate(float rate, float strength, std::default_random_engine& engine);
  NeuralNetwork crossover(const NeuralNetwork& partner, std::default_random_engine& engine) const;

  NeuralNetwork& operator=(const NeuralNetwork&) = default;

  std::vector<Layer>::const_iterator begin() const;
  std::vector<Layer>::const_iterator end() const;

  std::vector<Layer>::iterator begin();
  std::vector<Layer>::iterator end();

  const std::vector<Layer>& getLayers() const;

  inline size_t size() const {
    return layers.size();
  }
};

};  // namespace neuro
