#pragma once

#include <vector>

namespace neuro {

template <typename T>
class IIterable {
  virtual typename std::vector<T>::const_iterator begin() const;
  virtual typename std::vector<T>::const_iterator end() const;

  virtual typename std::vector<T>::iterator begin();
  virtual typename std::vector<T>::iterator end();
};

};  // namespace neuro
