#include "neuro/individual.hpp"

#include <random>
#include <vector>

#include "neuro/activation.hpp"
#include "neuro/layer.hpp"
#include "neuro/neural_network.hpp"

namespace neuro {

Individual::Individual(NeuralNetwork& neuralNetwork)
    : neuralNetwork(neuralNetwork) {}

Individual::Individual(const std::vector<Layer>& layers)
    : neuralNetwork(layers) {}

Individual::Individual(const std::vector<int>& structure, const ActivationFunction& activation)
    : neuralNetwork(structure, activation) {}

Individual::Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations)
    : neuralNetwork(structure, activations) {}

inline void Individual::loadWeights(float range) {
  neuralNetwork.loadWeights(range);
}

void Individual::loadWeights(float rangeMin, float rangeMax) {
  neuralNetwork.loadWeights(rangeMin, rangeMax);
}

inline void Individual::loadBias(float range) {
  neuralNetwork.loadBias(range);
}

void Individual::loadBias(float rangeMin, float rangeMax) {
  neuralNetwork.loadBias(rangeMin, rangeMax);
}

neuro_layer_t Individual::predict(const neuro_layer_t& input) const {
  return neuralNetwork.feedforward(input);
}

void Individual::mutate(float rate, float strength) {
  std::random_device rd;
  std::default_random_engine engine(rd());

  neuralNetwork.mutate(rate, strength, engine);
}

Individual Individual::crossover(const Individual& partner) const {
  std::random_device rd;
  std::default_random_engine engine(rd());

  auto newNeuralNetwork(neuralNetwork.crossover(partner.neuralNetwork, engine));

  return Individual(newNeuralNetwork);
}

const NeuralNetwork& Individual::getNeuralNetwork() const {
  return neuralNetwork;
}

float Individual::getFitness() const {
  return fitness;
}

void Individual::setFitness(float newFitness) {
  fitness = newFitness;
}

};  // namespace neuro
