#include "neuro/impl/individual.hpp"

#include <memory>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

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

void Individual::setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) {
  this->neuralNetwork = std::move(neuralNetwork);
}

float Individual::setFitness(float fitness) {
  this->fitness = fitness;
}

const INeuralNetwork& Individual::getNeuralNetwork() const {
  return *neuralNetwork;
}

INeuralNetwork& Individual::getNeuralNetworkMutable() {
  return *neuralNetwork;
}

float Individual::getFitness() const {
  return fitness;
}

};  // namespace neuro
