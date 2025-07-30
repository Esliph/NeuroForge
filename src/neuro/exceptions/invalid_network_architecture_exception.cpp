#include "neuro/exceptions/invalid_network_architecture_exception.hpp"

#include <string>

#include "neuro/exceptions/exception.hpp"

namespace neuro {

  namespace exception {

    InvalidNetworkArchitectureException::InvalidNetworkArchitectureException(const std::string& message)
      : NeuroException("Invalid Network Architecture: " + message) {}

  }; // namespace exception

}; // namespace neuro
