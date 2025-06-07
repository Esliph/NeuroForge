#pragma once

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/utils/interfaces/i_iterable.hpp"

namespace neuro {

class IPopulation : IIterable<IIndividual> {
 public:
  IPopulation() = default;
  IPopulation(const IPopulation&) = default;
  virtual ~IPopulation() = default;

  virtual void evolve() = 0;

  virtual const std::vector<std::unique_ptr<IIndividual>>& getIndividuals() const = 0;
  virtual std::vector<std::unique_ptr<IIndividual>>& getIndividualsMutable() = 0;

  virtual const IIndividual& getBestIndividual() const = 0;

  virtual const IIndividual& get(size_t index) const = 0;
  virtual IIndividual& get(size_t index) = 0;

  virtual float size() const = 0;
};

};  // namespace neuro
