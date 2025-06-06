#pragma once

#include <vector>

namespace neuro {

template <typename T>
class IIterable {
  std::vector<T>::const_iterator begin() const;
  std::vector<T>::const_iterator end() const;

  std::vector<T>::iterator begin();
  std::vector<T>::iterator end();
};

};  // namespace neuro
