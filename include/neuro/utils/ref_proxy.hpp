#pragma once

namespace neuro {

template <typename T>
class RefProxy {
  T& ref;

 public:
  RefProxy() = delete;
  RefProxy(T& value) : ref(value) {}

  operator T() const;
  operator T*();

  RefProxy<T>& operator=(const T& val);
  RefProxy<T>& operator=(T* val);

  T& operator*();
  T* operator->();
};

};  // namespace neuro
