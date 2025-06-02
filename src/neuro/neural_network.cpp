#include "neuro/neural_network.hpp"

#include <random>
#include <vector>

#include "neuro/activation.hpp"
#include "neuro/layer.hpp"

namespace neuro {

NeuralNetwork::NeuralNetwork(const std::vector<Layer>& layers)
    : layers(layers) {}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation) {
  for (size_t i = 0; i < structure.size() - 1; ++i) {
    layers.emplace_back(structure[0], structure[i + 1], activation);
  }
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations) {
  for (size_t i = 0; i < structure.size() - 1; ++i) {
    layers.emplace_back(structure[0], structure[i + 1], activations[i]);
  }
}

void NeuralNetwork::loadWeights(float rangeMin, float rangeMax) {
  for (size_t i = 0; i < layers.size(); ++i) {
    layers[i].loadWeights(rangeMin, rangeMax);
  }
}

void NeuralNetwork::loadBias(float rangeMin, float rangeMax) {
  for (size_t i = 0; i < layers.size(); ++i) {
    layers[i].loadBias(rangeMin, rangeMax);
  }
}

neuro_layer_t NeuralNetwork::feedforward(const neuro_layer_t& input) const {
  neuro_layer_t current = input;

  for (const auto& layer : layers) {
    current = layer.process(current);
  }

  return current;
}

void NeuralNetwork::mutate(float rate, float strength, std::default_random_engine& engine) {
  for (auto& layer : layers) {
    layer.mutate(rate, strength, engine);
  }
}

NeuralNetwork NeuralNetwork::crossover(const NeuralNetwork& partner, std::default_random_engine& engine) const {
  std::vector<Layer> newLayers;

  for (size_t i = 0; i < layers.size(); ++i) {
    newLayers.emplace_back(layers[i].crossover(partner.layers[i], engine));
  }

  return NeuralNetwork(newLayers);
}

std::vector<Layer>::const_iterator NeuralNetwork::begin() const {
  return layers.begin();
}

std::vector<Layer>::const_iterator NeuralNetwork::end() const {
  return layers.end();
}

std::vector<Layer>::iterator NeuralNetwork::begin() {
  return layers.begin();
}

std::vector<Layer>::iterator NeuralNetwork::end() {
  return layers.end();
}

const std::vector<Layer>& NeuralNetwork::getLayers() const {
  return layers;
}

};  // namespace neuro
