#pragma once

#include <memory>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"

namespace neuro {

  struct GeneticOptions {
    float rate = 0.5f;
    float intensity = 0.5f;
    size_t eliteCount = 5;

    bool operator!=(const GeneticOptions& other) const {
      return !(*this == other);
    }

    bool operator==(const GeneticOptions& other) const {
      return rate == other.rate && intensity == other.intensity && eliteCount == other.eliteCount;
    }
  };

  class GeneticTrainer : public IStrategyEvolution {
   protected:
    std::shared_ptr<IPopulation> population;

    GeneticOptions options{};

   public:
    GeneticTrainer() = delete;
    GeneticTrainer(const GeneticTrainer&) = default;

    GeneticTrainer(const std::shared_ptr<IPopulation>&);
    GeneticTrainer(const std::shared_ptr<IPopulation>&, float rate, float intensity = 0.5f, size_t eliteCount = 5);
    GeneticTrainer(const std::shared_ptr<IPopulation>&, const GeneticOptions& options);

    virtual ~GeneticTrainer() = default;

    virtual void evolve();

    virtual void mutate();

    virtual void crossover();

    FORCE_INLINE std::shared_ptr<IPopulation> getPopulation() {
      return population;
    }

    FORCE_INLINE void setPopulation(const std::shared_ptr<IPopulation>& population) {
      this->population = population;
    }

    FORCE_INLINE void setOptions(const GeneticOptions& options) {
      this->options = options;
    }

    FORCE_INLINE void setRate(float rate) {
      options.rate = rate;
    }

    FORCE_INLINE void setIntensity(float intensity) {
      options.intensity = intensity;
    }

    FORCE_INLINE void setEliteCount(size_t eliteCount) {
      options.eliteCount = eliteCount;
    }

    FORCE_INLINE const GeneticOptions& getOptions() const {
      return options;
    }
  };

}; // namespace neuro
