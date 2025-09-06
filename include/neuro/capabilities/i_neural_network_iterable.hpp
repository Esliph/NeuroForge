#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_layer.hpp"

namespace neuro {

  class INeuralNetworkIterable {
   public:
    virtual ~INeuralNetworkIterable() = default;

    virtual std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>::iterator begin() = 0;

    virtual std::vector<std::unique_ptr<ILayer>>::const_iterator end() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>::iterator end() = 0;
  };

} // namespace neuro
