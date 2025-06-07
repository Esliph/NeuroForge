#pragma once

#include "neuro/interfaces/i_individual.hpp"

namespace neuro {

class Individual : IIndividual {
 public:
  Individual(const Individual&) = default;
  virtual ~Individual() = default;
};

};  // namespace neuro
