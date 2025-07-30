#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_individual.hpp"

namespace neuro {

  class IPopulation {
   public:
    IPopulation() = default;
    virtual ~IPopulation() = default;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual void addIndividuals(const std::vector<IIndividual>&) = 0;
    virtual void addIndividual(const IIndividual&) = 0;
    virtual void addIndividuals(std::vector<std::shared_ptr<IIndividual>>&) = 0;
    virtual void addIndividual(std::shared_ptr<IIndividual>) = 0;

    virtual void removeIndividual(size_t index) = 0;
    virtual void clearIndividuals() = 0;
    virtual void popIndividual() = 0;

    virtual void reserve(size_t size) = 0;

    virtual const IIndividual& getBestIndividual() const = 0;

    virtual const std::vector<std::shared_ptr<IIndividual>>& getIndividuals() const = 0;
    virtual std::vector<std::shared_ptr<IIndividual>>& getIndividuals() = 0;

    virtual const IIndividual& get(size_t index) const = 0;
    virtual IIndividual& get(size_t index) = 0;

    virtual size_t size() const = 0;

    virtual bool empty() const = 0;

    virtual std::vector<std::shared_ptr<IIndividual>>::const_iterator begin() const = 0;
    virtual std::vector<std::shared_ptr<IIndividual>>::iterator begin() = 0;

    virtual std::vector<std::shared_ptr<IIndividual>>::const_iterator end() const = 0;
    virtual std::vector<std::shared_ptr<IIndividual>>::iterator end() = 0;

    virtual const IIndividual& operator[](size_t index) const = 0;
    virtual IIndividual& operator[](size_t index) = 0;

    virtual std::unique_ptr<IPopulation> clone() const = 0;
  };

}; // namespace neuro
