#pragma once

#include <memory>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

class Individual : public IIndividual {
  std::unique_ptr<INeuralNetwork> neuralNetwork{};
  float fitness{};

 public:
  Individual() = default;
  Individual(const Individual&) = default;
  Individual(int fitness);
  Individual(std::unique_ptr<INeuralNetwork>);
  Individual(std::unique_ptr<INeuralNetwork>, int fitness);

  virtual ~Individual() = default;

  IIndividual crossover(const IIndividual& partner) const override;

  void setNeuralNetwork(std::unique_ptr<INeuralNetwork>) override;
  float setFitness(float) override;

  const INeuralNetwork& getNeuralNetwork() const override;
  INeuralNetwork& getNeuralNetworkMutable() override;
  float getFitness() const override;
};

};  // namespace neuro
