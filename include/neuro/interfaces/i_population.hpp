#pragma once

#include <memory>

#include "neuro/interfaces/i_individual.hpp"

namespace neuro {

class IPopulation {
 public:
  IPopulation() = default;
  IPopulation(const IPopulation&) = default;
  virtual ~IPopulation() = default;

  virtual void removeIndividual(size_t index) = 0;
  virtual void popIndividual() = 0;

  virtual const IIndividual& getBestIndividual() const = 0;

  virtual const std::vector<std::unique_ptr<IIndividual>>& getIndividuals() const = 0;
  virtual std::vector<std::unique_ptr<IIndividual>>& getIndividuals() = 0;

  virtual const IIndividual& get(size_t index) const = 0;
  virtual IIndividual& get(size_t index) = 0;

  virtual size_t size() const = 0;

  virtual std::unique_ptr<IPopulation> clone() const = 0;
  virtual IPopulation& operator=(const IPopulation&) = default;
};

};  // namespace neuro
