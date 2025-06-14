#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

class Population : public IPopulation {
  std::vector<std::unique_ptr<IIndividual>> individuals{};

 public:
  Population() = default;
  Population(std::vector<std::unique_ptr<IIndividual>>& individuals);
  Population(int size, const std::vector<int>& structure, const ActivationFunction& activation);
  Population(int size, const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

  virtual ~Population() = default;

  void addIndividual(std::unique_ptr<IIndividual>) override;
  void removeIndividual(size_t index) override;
  void clearIndividuals() override;
  void popIndividual() override;

  const IIndividual& getBestIndividual() const override;

  const std::vector<std::unique_ptr<IIndividual>>& getIndividuals() const override;
  std::vector<std::unique_ptr<IIndividual>>& getIndividuals() override;

  const IIndividual& get(size_t index) const override;
  IIndividual& get(size_t index) override;

  size_t size() const override;

  std::vector<std::unique_ptr<IIndividual>>::const_iterator begin() const override;
  std::vector<std::unique_ptr<IIndividual>>::iterator begin() override;

  std::vector<std::unique_ptr<IIndividual>>::const_iterator end() const override;
  std::vector<std::unique_ptr<IIndividual>>::iterator end() override;

  const IIndividual& operator[](int index) const override;
  IIndividual& operator[](int index) override;
};

};  // namespace neuro
