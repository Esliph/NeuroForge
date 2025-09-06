#include "internal/random_engine.hpp"

namespace neuro {

  std::default_random_engine random_engine(std::random_device{}());

} // namespace neuro
