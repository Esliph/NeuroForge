#pragma once

#include <vector>

#include "neuro/individual.hpp"

namespace neuro {

class Population {
  std::vector<Individual> individuals;

 public:
  Population() = default;
  Population(const Population&) = default;
  Population(const std::vector<Individual>& individuals);
  Population(int size, const std::vector<int>& structure, const ActivationFunction& activation);
  Population(int size, const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

  const Individual& getBest() const;
  void evolve(float mutationRate, float mutationStrength, unsigned int eliteCount);

  Population& operator=(const Population&) = default;

  const std::vector<Individual> getIndividuals() const;
  inline float size() const;

 private:
  const Individual& tournamentSelect(int rounds = 3) const;
};

};  // namespace neuro
