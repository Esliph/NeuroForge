#pragma once

#include <random>
#include <vector>

#include "neuro/interfaces/layer/i_layer.hpp"

namespace neuro {

class IStatefulLayer : public ILayer {
 public:
  IStatefulLayer(IStatefulLayer&) = default;
  virtual ~IStatefulLayer() = default;

  virtual void compute() = 0;

  virtual void setInput(const neuro_layer_t& input) = 0;

  virtual const neuro_layer_t& getOutput() const = 0;
};

};  // namespace neuro
