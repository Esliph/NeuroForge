#pragma once

namespace neuro {

  template <typename T>
  class RefProxy {
    T& ref;

   public:
    RefProxy() = delete;
    RefProxy(T& value)
      : ref(value) {}

    operator T() const {
      return ref;
    }

    operator T*() {
      return &ref;
    }

    RefProxy<T>& operator=(const T& val) {
      ref = val;
      return *this;
    }

    RefProxy<T>& operator=(T* val) {
      if (val) {
        ref = *val;
      }
      return *this;
    }

    T& operator*() {
      return ref;
    }

    T* operator->() {
      return &ref;
    }
  };

}; // namespace neuro
