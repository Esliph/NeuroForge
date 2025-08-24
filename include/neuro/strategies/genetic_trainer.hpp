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
  };

  class GeneticTrainer : public IStrategyEvolution {
    std::shared_ptr<IPopulation> population;

    GeneticOptions options{};

   public:
    GeneticTrainer() = delete;
    GeneticTrainer(const GeneticTrainer&) = default;

    GeneticTrainer(const std::shared_ptr<IPopulation>&);
    GeneticTrainer(const std::shared_ptr<IPopulation>&, float rate, float intensity = 0.5f, size_t eliteCount = 5);
    GeneticTrainer(const std::shared_ptr<IPopulation>&, const GeneticOptions& options);

    virtual ~GeneticTrainer() = default;

    virtual IPopulation& getPopulation();
    virtual void setPopulation(const std::shared_ptr<IPopulation>&);

    virtual const GeneticOptions& getOptions() const;

    virtual void setOptions(const GeneticOptions&);
    virtual void setRate(float);
    virtual void setIntensity(float);
    virtual void setEliteCount(size_t);
  };

}; // namespace neuro
