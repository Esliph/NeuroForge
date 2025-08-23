#include "neuro/impl/individual.hpp"

#include <functional>
#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

  Individual::Individual()
    : IIndividual(),
      neuralNetwork(std::make_unique<NeuralNetwork>()) {}

  Individual::Individual(const INeuralNetwork& neuralNetwork)
    : IIndividual(),
      neuralNetwork(neuralNetwork.clone()) {}

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

  FORCE_INLINE void Individual::evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) {
    fitness = evaluateFunction(*neuralNetwork);
  }

  FORCE_INLINE const INeuralNetwork& Individual::getNeuralNetwork() const {
    return *neuralNetwork;
  }

  FORCE_INLINE INeuralNetwork& Individual::getNeuralNetwork() {
    return *neuralNetwork;
  }

  FORCE_INLINE void Individual::setNeuralNetwork(const INeuralNetwork& neuralNetwork) {
    this->neuralNetwork = std::move(neuralNetwork.clone());
  }

  FORCE_INLINE void Individual::setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) {
    this->neuralNetwork = std::move(neuralNetwork);
  }

  FORCE_INLINE float Individual::getFitness() const {
    return fitness;
  }

  FORCE_INLINE void Individual::setFitness(float fitness) {
    this->fitness = fitness;
  }

  FORCE_INLINE std::unique_ptr<IIndividual> Individual::clone() const {
    return std::make_unique<Individual>(*this);
  };

}; // namespace neuro
