#pragma once

#include <memory>
#include <vector>

#include "neuro/core/i_individual.hpp"

namespace neuro {

  template <typename TIIndividual = IIndividual<>>
  class IPopulation {
   public:
    IPopulation() = default;
    virtual ~IPopulation() = default;

    virtual void addIndividuals(const std::vector<TIIndividual>&) = 0;
    virtual void addIndividual(const TIIndividual&) = 0;
    virtual void addIndividuals(std::vector<std::shared_ptr<TIIndividual>>&) = 0;
    virtual void addIndividual(std::shared_ptr<TIIndividual>) = 0;

    virtual void removeIndividual(size_t index) = 0;
    virtual void clearIndividuals() = 0;
    virtual void popIndividual() = 0;

    virtual void reserve(size_t size) = 0;

    virtual const TIIndividual& getBestIndividual() const = 0;

    virtual const std::vector<std::shared_ptr<TIIndividual>>& getIndividuals() const = 0;
    virtual std::vector<std::shared_ptr<TIIndividual>>& getIndividuals() = 0;

    virtual const TIIndividual& get(size_t index) const = 0;
    virtual TIIndividual& get(size_t index) = 0;

    virtual size_t size() const = 0;

    virtual bool empty() const = 0;

    virtual std::vector<std::shared_ptr<TIIndividual>>::const_iterator begin() const = 0;
    virtual std::vector<std::shared_ptr<TIIndividual>>::iterator begin() = 0;

    virtual std::vector<std::shared_ptr<TIIndividual>>::const_iterator end() const = 0;
    virtual std::vector<std::shared_ptr<TIIndividual>>::iterator end() = 0;

    virtual const TIIndividual& operator[](size_t index) const = 0;
    virtual TIIndividual& operator[](size_t index) = 0;

    virtual std::unique_ptr<IPopulation<TIIndividual>> clone() const = 0;
  };

} // namespace neuro
