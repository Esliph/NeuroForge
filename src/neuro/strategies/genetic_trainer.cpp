#include "neuro/strategies/genetic_trainer.hpp"

#include "internal/attribute.hpp"

namespace neuro {

  GeneticTrainer::GeneticTrainer(float rate, float intensity, size_t eliteCount)
    : options({rate, intensity, eliteCount}) {}

  GeneticTrainer::GeneticTrainer(const GeneticOptions& options)
    : options(options) {}

  FORCE_INLINE void GeneticTrainer::setOptions(const GeneticOptions& options) {
    this->options = options;
  }

  FORCE_INLINE void GeneticTrainer::setRate(float rate) {
    options.rate = rate;
  }

  FORCE_INLINE void GeneticTrainer::setIntensity(float intensity) {
    options.intensity = intensity;
  }

  FORCE_INLINE void GeneticTrainer::setEliteCount(size_t eliteCount) {
    options.eliteCount = eliteCount;
  }

  FORCE_INLINE const GeneticOptions& GeneticTrainer::getOptions() const {
    return options;
  }

}; // namespace neuro
