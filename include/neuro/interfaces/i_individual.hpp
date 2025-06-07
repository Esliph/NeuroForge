#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/interfaces/i_crossable.hpp"

namespace neuro {

class IIndividual : ICrossable<IIndividual> {
 public:
  IIndividual(IIndividual&) = default;
  virtual ~IIndividual() = default;

  virtual void evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) = 0;

  virtual void mutate(float rate, float intensity) = 0;

  virtual void setNeuralNetwork(std::unique_ptr<INeuralNetwork>) = 0;
  virtual float setFitness(float) = 0;

  virtual const INeuralNetwork& getNeuralNetwork() const = 0;
  virtual INeuralNetwork& getNeuralNetworkMutable() = 0;
  virtual float getFitness() const = 0;

  virtual IIndividual& operator=(IIndividual&) = default;
};

};  // namespace neuro
