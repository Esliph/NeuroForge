#pragma once

namespace neuro {

template <typename T>
class ICrossable {
  virtual T crossover(const T& partner) const = 0;
};

};  // namespace neuro
