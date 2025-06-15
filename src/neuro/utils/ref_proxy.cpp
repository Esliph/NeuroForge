#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

template <typename T>
RefProxy<T>::operator T() const {
  return ref;
}

template <typename T>
RefProxy<T>::operator T*() {
  return &ref;
}

template <typename T>
RefProxy<T>& RefProxy<T>::operator=(const T& val) {
  ref = val;
  return *this;
}

template <typename T>
RefProxy<T>& RefProxy<T>::operator=(T* val) {
  if (val) {
    ref = *val;
  }
  return *this;
}

template <typename T>
T& RefProxy<T>::operator*() {
  return ref;
}

template <typename T>
T* RefProxy<T>::operator->() {
  return &ref;
}

}  // namespace neuro
