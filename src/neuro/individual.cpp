#include "neuro/individual.hpp"

#include <random>
#include <vector>

#include "neuro/activation.hpp"
#include "neuro/layer.hpp"
#include "neuro/neural_network.hpp"
#include "neuro/util.hpp"

namespace neuro {

Individual::Individual(NeuralNetwork& neuralNetwork)
    : neuralNetwork(neuralNetwork) {}

Individual::Individual(const std::vector<Layer>& layers)
    : neuralNetwork(layers) {}

Individual::Individual(const std::vector<int>& structure, const ActivationFunction& activation)
    : neuralNetwork(structure, activation) {}

Individual::Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations)
    : neuralNetwork(structure, activations) {}

neuro_layer_t Individual::predict(const neuro_layer_t& input) const {
  return neuralNetwork.feedforward(input);
}

void Individual::mutate(float rate, float strength) {
  neuralNetwork.mutate(rate, strength, random_engine);
}

Individual Individual::crossover(const Individual& partner) const {
  auto newNeuralNetwork(neuralNetwork.crossover(partner.neuralNetwork, random_engine));

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
