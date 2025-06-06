#pragma once

#include <vector>

#include "neuro/interfaces/layer/i_layer.hpp"

namespace neuro {

class IReinforcementTrainable {
 public:
  virtual ~IReinforcementTrainable() = default;

  virtual void updateFromReward(float reward) = 0;

  virtual void observe(const neuro_layer_t& state) = 0;
  virtual void takeAction(const neuro_layer_t& action) = 0;
};

};  // namespace neuro
