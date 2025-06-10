#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_individual.hpp"

namespace neuro {

class IPopulation {
 public:
  IPopulation() = default;
  IPopulation(const IPopulation&) = default;
  virtual ~IPopulation() = default;

  virtual void addIndividual(std::unique_ptr<IIndividual>) = 0;
  virtual void removeIndividual(size_t index) = 0;
  virtual void clearIndividuals() = 0;
  virtual void popIndividual() = 0;

  virtual const IIndividual& getBestIndividual() const = 0;

  virtual const std::vector<std::unique_ptr<IIndividual>>& getIndividuals() const = 0;
  virtual std::vector<std::unique_ptr<IIndividual>>& getIndividuals() = 0;

  virtual const IIndividual& get(size_t index) const = 0;
  virtual IIndividual& get(size_t index) = 0;

  virtual size_t size() const = 0;

  virtual std::vector<std::unique_ptr<IIndividual>>::const_iterator begin() const = 0;
  virtual std::vector<std::unique_ptr<IIndividual>>::iterator begin() = 0;

  virtual std::vector<std::unique_ptr<IIndividual>>::const_iterator end() const = 0;
  virtual std::vector<std::unique_ptr<IIndividual>>::iterator end() = 0;

  virtual const IIndividual& operator[](int index) const = 0;
  virtual IIndividual& operator[](int index) = 0;

  virtual std::unique_ptr<IPopulation> clone() const = 0;
};

};  // namespace neuro
