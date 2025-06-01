#pragma once

#include <vector>

#include "neuro/activation.hpp"
#include "neuro/layer.hpp"
#include "neuro/neural_network.hpp"

namespace neuro {

class Individual {
  NeuralNetwork neuralNetwork;
  float fitness{};

 public:
  Individual() = default;
  Individual(const Individual&) = default;
  Individual(NeuralNetwork& neuralNetwork);
  Individual(const std::vector<Layer>& layers);
  Individual(const std::vector<int>& structure, const ActivationFunction& activation);
  Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

  inline void loadWeights(float range = 1.0f);
  void loadWeights(float rangeMin, float rangeMax);

  inline void loadBias(float range = 1.0f);
  void loadBias(float rangeMin, float rangeMax);

  void mutate(float rate, float strength);
  Individual crossover(const Individual& partner) const;

  Individual& operator=(const Individual&) = default;

  const NeuralNetwork& getNeuralNetwork() const;
  float getFitness() const;
};

};  // namespace neuro
