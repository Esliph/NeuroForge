#pragma once

#include "neuro/interfaces/layer/i_layer.hpp"

namespace neuro {

class DenseLayer : ILayer {
 public:
  DenseLayer(DenseLayer&) = default;
  virtual ~DenseLayer() = default;
};

};  // namespace neuro