#pragma once

namespace neuro {

template <typename T>
class RefProxy {
  T& ref;

 public:
  RefProxy() = delete;
  RefProxy(T&);

  operator T() const;
  RefProxy<T>& operator=(const T& val);
};

};  // namespace neuro
