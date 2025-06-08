#pragma once

#include <vector>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"

namespace neuro {

class ReinforcementTrainable : public IStrategyEvolution {
 public:
  ReinforcementTrainable() = default;
  virtual ~ReinforcementTrainable() = default;

  virtual void updateFromReward(float reward) = 0;

  virtual void observe(const neuro_layer_t& state) = 0;
  virtual void takeAction(const neuro_layer_t& action) = 0;
};

};  // namespace neuro
