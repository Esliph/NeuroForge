#pragma once

#include <exception>
#include <string>

namespace neuro {

  namespace exception {

    class NeuroException : public std::exception {
     protected:
      std::string message;

     public:
      explicit NeuroException(const std::string& message);

      const char* what() const noexcept override;
    };

  }; // namespace exception

}; // namespace neuro
