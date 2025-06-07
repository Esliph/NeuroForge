#pragma once

#include "neuro/interfaces/population/i_population.hpp"

namespace neuro {

class Population : IPopulation {
 public:
  Population() = default;
  Population(const Population&) = default;
  virtual ~Population() = default;
};

};  // namespace neuro
