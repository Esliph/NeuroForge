#pragma once

#include <string>

#include "neuro/exceptions/exception.hpp"

namespace neuro {

  namespace exception {

    class InvalidNetworkArchitectureException : public NeuroException {
     public:
      explicit InvalidNetworkArchitectureException(const std::string& message);
    };

  }; // namespace exception

}; // namespace neuro
