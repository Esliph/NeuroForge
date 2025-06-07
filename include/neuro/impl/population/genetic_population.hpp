#pragma once

#include "neuro/impl/population/population.hpp"
#include "neuro/interfaces/population/i_genetic_population.hpp"

namespace neuro {

class GeneticPopulation : Population, IGeneticPopulation {
 public:
  GeneticPopulation(const GeneticPopulation&) = default;
  virtual ~GeneticPopulation() = default;
};

};  // namespace neuro
