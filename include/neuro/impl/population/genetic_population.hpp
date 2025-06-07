#pragma once

#include "neuro/impl/population/population.hpp"
#include "neuro/interfaces/population/i_genetic_population.hpp"

namespace neuro {

class GeneticPopulation : public Population, public IGeneticPopulation {
 public:
  GeneticPopulation() = default;
  GeneticPopulation(const GeneticPopulation&) = default;
  virtual ~GeneticPopulation() = default;
};

};  // namespace neuro
