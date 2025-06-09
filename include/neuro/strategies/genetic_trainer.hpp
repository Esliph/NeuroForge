#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"

namespace neuro {

struct GeneticTrainerOptions {
  float rate = 0.5f;
  float intensity = 0.5f;
  size_t eliteCount = 5;
};

class GeneticTrainer : public IStrategyEvolution {
  GeneticTrainerOptions options{};

 public:
  GeneticTrainer() = default;
  GeneticTrainer(float rate, float intensity = 0.5f, size_t eliteCount = 5);
  GeneticTrainer(const GeneticTrainerOptions& options);

  virtual ~GeneticTrainer() = default;

  virtual void mutate(IIndividual& individual) const;
  virtual void mutate(INeuralNetwork& neuralNetwork) const;
  virtual void mutate(std::vector<ILayer>& layers) const;
  virtual void mutate(std::vector<layer_weight_t>& layerWeights, std::vector<layer_bias_t>& layerBiases) const;

  virtual void mutate(IIndividual& individual, float rate, float intensity) const;
  virtual void mutate(INeuralNetwork& neuralNetwork, float rate, float intensity) const;
  virtual void mutate(std::vector<ILayer>& layers, float rate, float intensity) const;
  virtual void mutate(std::vector<layer_weight_t>& layerWeights, std::vector<layer_bias_t>& layerBiases, float rate, float intensity) const;

  virtual std::unique_ptr<IIndividual> crossover(const IIndividual& partnerA, const IIndividual& partnerB) const;
  virtual std::unique_ptr<INeuralNetwork> crossover(const INeuralNetwork& partnerA, const INeuralNetwork& partnerB) const;
  virtual std::vector<std::unique_ptr<ILayer>> crossover(const std::vector<ILayer>& partnerLayersA, const std::vector<ILayer>& partnerLayersB) const;
  virtual std::vector<std::unique_ptr<ILayer>> crossover(
      const std::vector<layer_weight_t>& parentLayerWeightsA,
      const std::vector<layer_bias_t>& parentLayerBiasesA,
      const std::vector<layer_weight_t>& parentLayerWeightsB,
      const std::vector<layer_bias_t>& parentLayerBiasesB) const;

  virtual std::unique_ptr<IPopulation> evolve(const IPopulation& population) const;
  virtual std::unique_ptr<IPopulation> evolve(const std::vector<IIndividual>& individuals) const;

  virtual std::vector<std::unique_ptr<IIndividual>> select(const IPopulation& population) const;
  virtual std::vector<std::unique_ptr<IIndividual>> select(const std::vector<IIndividual>& individuals) const;

  virtual std::vector<std::unique_ptr<IIndividual>> select(const IPopulation& population, size_t eliteCount) const;
  virtual std::vector<std::unique_ptr<IIndividual>> select(const std::vector<IIndividual>& individuals, size_t eliteCount) const;

  virtual void setOptions(const GeneticTrainerOptions&);
  virtual void setRate(float);
  virtual void setIntensity(float);
  virtual void setEliteCount(size_t);

  virtual const GeneticTrainerOptions& getOptions() const;
};

};  // namespace neuro
