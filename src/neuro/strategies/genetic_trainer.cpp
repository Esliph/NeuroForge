#include "neuro/strategies/genetic_trainer.hpp"

#include <memory>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_population.hpp"

namespace neuro {

  GeneticTrainer::GeneticTrainer(const std::shared_ptr<IPopulation>& population)
    : population(population) {}

  GeneticTrainer::GeneticTrainer(const std::shared_ptr<IPopulation>& population, float rate, float intensity, size_t eliteCount)
    : population(population),
      options({rate, intensity, eliteCount}) {}

  GeneticTrainer::GeneticTrainer(const std::shared_ptr<IPopulation>& population, const GeneticOptions& options)
    : population(population),
      options(options) {}

  FORCE_INLINE IPopulation& GeneticTrainer::getPopulation() {
    return *population;
  }

  FORCE_INLINE void GeneticTrainer::setPopulation(const std::shared_ptr<IPopulation>& population) {
    this->population = population;
  }

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
