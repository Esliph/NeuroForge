#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

template <typename T>
RefProxy<T>::RefProxy(T& value) : ref(value) {}

template <typename T>
RefProxy<T>::operator T() const {
  return ref;
}

template <typename T>
RefProxy<T>& RefProxy<T>::operator=(const T& val) {
  ref = val;
  return *this;
}

}  // namespace neuro
