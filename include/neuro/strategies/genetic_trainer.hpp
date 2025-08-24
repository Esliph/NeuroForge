#pragma once

#include "neuro/strategies/i_strategy_evolution.hpp"

namespace neuro {

  struct GeneticOptions {
    float rate = 0.5f;
    float intensity = 0.5f;
    size_t eliteCount = 5;
  };

  class GeneticTrainer : public IStrategyEvolution {
    GeneticOptions options{};

   public:
    GeneticTrainer() = default;
    GeneticTrainer(const GeneticTrainer&) = default;

    GeneticTrainer(float rate, float intensity = 0.5f, size_t eliteCount = 5);
    GeneticTrainer(const GeneticOptions& options);

    virtual ~GeneticTrainer() = default;

    virtual const GeneticOptions& getOptions() const;

    virtual void setOptions(const GeneticOptions&);
    virtual void setRate(float);
    virtual void setIntensity(float);
    virtual void setEliteCount(size_t);
  };

}; // namespace neuro
