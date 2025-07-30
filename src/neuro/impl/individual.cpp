#include "neuro/impl/individual.hpp"

#include <functional>
#include <memory>
#include <vector>

#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

  Individual::Individual()
    : IIndividual(),
      neuralNetwork(std::make_unique<NeuralNetwork>()) {}

  Individual::Individual(const Individual& individual)
    : IIndividual(),
      neuralNetwork(std::move(individual.getNeuralNetwork().clone())),
      fitness(individual.fitness) {}

  Individual::Individual(int fitness)
    : IIndividual(),
      neuralNetwork(std::make_unique<NeuralNetwork>()),
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

}; // namespace neuro
