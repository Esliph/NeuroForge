#include "neuro/impl/individual.hpp"

#include <functional>
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

void Individual::randomizeWeights(float min, float max) {
  neuralNetwork->randomizeWeights(min, max);
}

void Individual::randomizeBiases(float min, float max) {
  neuralNetwork->randomizeBiases(min, max);
}

neuro_layer_t Individual::feedforward(const neuro_layer_t& inputs) const {
  return neuralNetwork->feedforward(inputs);
}

size_t Individual::inputSize() const {
  return neuralNetwork->inputSize();
}

size_t Individual::outputSize() const {
  return neuralNetwork->outputSize();
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

void Individual::setAllWeights(const std::vector<layer_weight_t>& weights) {
  neuralNetwork->setAllWeights(weights);
}

void Individual::setAllBiases(const std::vector<layer_bias_t>& biases) {
  neuralNetwork->setAllBiases(biases);
}

void Individual::setLayers(std::vector<std::unique_ptr<ILayer>> layers) {
  neuralNetwork->setLayers(std::move(layers));
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

size_t Individual::sizeLayers() const {
  return neuralNetwork->sizeLayers();
}

std::vector<std::unique_ptr<ILayer>>::const_iterator Individual::begin() const {
  return neuralNetwork->begin();
}

std::vector<std::unique_ptr<ILayer>>::iterator Individual::begin() {
  return neuralNetwork->begin();
}

std::vector<std::unique_ptr<ILayer>>::const_iterator Individual::end() const {
  return neuralNetwork->end();
}

std::vector<std::unique_ptr<ILayer>>::iterator Individual::end() {
  return neuralNetwork->end();
}

neuro_layer_t Individual::operator()(const neuro_layer_t& inputs) const {
  return feedforward(inputs);
}

const ILayer& Individual::operator[](int index) const {
  return (*neuralNetwork)[index];
}

ILayer& Individual::operator[](int index) {
  return (*neuralNetwork)[index];
}

};  // namespace neuro
