#pragma once

#include "neuro/interfaces/i_individual.hpp"

namespace neuro {

class Individual : IIndividual {
 public:
  Individual() = default;
  Individual(const Individual&) = default;
  virtual ~Individual() = default;
};

};  // namespace neuro
