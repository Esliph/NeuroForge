#include "neuro/impl/individual.hpp"

#include <memory>
#include <vector>

#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

Individual::Individual(const Individual& individual)
    : IIndividual(),
      fitness(individual.fitness),
      neuralNetwork(std::move(individual.getNeuralNetwork().clone())) {}

Individual::Individual(int fitness)
    : IIndividual(),
      fitness(fitness) {}

Individual::Individual(std::unique_ptr<INeuralNetwork> neuralNetwork)
    : IIndividual(),
      neuralNetwork(std::move(neuralNetwork)) {}

Individual::Individual(std::unique_ptr<INeuralNetwork>, int fitness)
    : IIndividual(),
      neuralNetwork(std::move(neuralNetwork)),
      fitness(fitness) {}

Individual::Individual(const std::vector<int>& structure, const ActivationFunction& activation)
    : IIndividual(),
      neuralNetwork(std::make_unique<NeuralNetwork>(structure, activation)) {}

Individual::Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations)
    : IIndividual(),
      neuralNetwork(std::make_unique<NeuralNetwork>(structure, activations)) {}

void Individual::evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) {
  auto newFitness = evaluateFunction(*neuralNetwork);

  fitness = newFitness;
}

neuro_layer_t Individual::feedforward(const neuro_layer_t& inputs) const {
  return neuralNetwork->feedforward(inputs);
}

void Individual::setNeuralNetwork(const INeuralNetwork& neuralNetwork) {
  this->neuralNetwork = std::move(neuralNetwork.clone());
}

void Individual::setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) {
  this->neuralNetwork = std::move(neuralNetwork);
}

void Individual::setFitness(float fitness) {
  this->fitness = fitness;
}

const INeuralNetwork& Individual::getNeuralNetwork() const {
  return *neuralNetwork;
}

INeuralNetwork& Individual::getNeuralNetwork() {
  return *neuralNetwork;
}

float Individual::getFitness() const {
  return fitness;
}

std::vector<layer_weight_t> Individual::getAllWeights() const {
  return neuralNetwork->getAllWeights();
}

std::vector<layer_bias_t> Individual::getAllBiases() const {
  return neuralNetwork->getAllBiases();
}

neuro_layer_t Individual::operator()(const neuro_layer_t& inputs) const {
  return feedforward(inputs);
}

};  // namespace neuro
