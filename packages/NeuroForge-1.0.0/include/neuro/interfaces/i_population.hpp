#pragma once

#include "neuro/interfaces/i_individual.hpp"

namespace neuro {

class IPopulation {
 public:
  IPopulation(IPopulation&) = default;
  virtual ~IPopulation() = default;

  virtual void evaluate(const std::function<float(const INeuralNetwork&)>& evaluateFunction) = 0;
  virtual void evolve() = 0;

  virtual const std::vector<std::unique_ptr<IIndividual>>& getIndividuals() const = 0;
  virtual const IIndividual& getBestIndividual() const = 0;

  virtual float size() const = 0;
};

};  // namespace neuro
