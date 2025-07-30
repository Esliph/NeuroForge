#include "neuro/exceptions/exception.hpp"

#include <exception>
#include <string>

namespace neuro {

  namespace exception {

    NeuroException::NeuroException(const std::string& message)
      : message(message) {}

    const char* NeuroException::what() const noexcept {
      return message.c_str();
    }

  }; // namespace exception

}; // namespace neuro
