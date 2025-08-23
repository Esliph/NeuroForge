#pragma once

#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/impl/dense_layer.hpp"
#include "neuro/impl/individual.hpp"
#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"
#include "neuro/types.hpp"

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

    GeneticTrainer(float rate, float intensity = 0.5f, size_t eliteCount = 5);
    GeneticTrainer(const GeneticOptions& options);

    virtual ~GeneticTrainer() = default;

    FORCE_INLINE void mutate(IIndividual& individual) const {
      for (auto& layer : individual.getNeuralNetwork()) {
        mutateWeights(layer->getWeights());
        mutateBiases(layer->getBiases());
      }
    }

    FORCE_INLINE void mutate(INeuralNetwork& neuralNetwork) const {
      for (auto& layer : neuralNetwork) {
        mutateWeights(layer->getWeights());
        mutateBiases(layer->getBiases());
      }
    }

    FORCE_INLINE void mutate(std::vector<ILayer>& layers) const {
      for (auto& layer : layers) {
        mutateWeights(layer.getWeights());
        mutateBiases(layer.getBiases());
      }
    }

    FORCE_INLINE void mutate(ILayer& layer) const {
      mutateWeights(layer.getWeights());
      mutateBiases(layer.getBiases());
    }

    virtual void mutateWeights(layer_weight_t& layerWeights) const;
    virtual void mutateBiases(layer_bias_t& layerBiases) const;

    FORCE_INLINE std::unique_ptr<IIndividual> crossover(const IIndividual& partnerA, const IIndividual& partnerB) const {
      return std::make_unique<Individual>(crossover(partnerA.getNeuralNetwork(), partnerB.getNeuralNetwork()));
    }

    FORCE_INLINE std::unique_ptr<INeuralNetwork> crossover(const INeuralNetwork& partnerA, const INeuralNetwork& partnerB) const {
      std::unique_ptr<INeuralNetwork> neuralNetwork = std::make_unique<NeuralNetwork>();

      for (size_t i = 0; i < partnerA.sizeLayers() && i < partnerB.sizeLayers(); i++) {
        neuralNetwork->addLayer(crossover(partnerA[i], partnerB[i]));
      }

      return neuralNetwork;
    }

    FORCE_INLINE std::unique_ptr<INeuralNetwork>
      crossover(const std::vector<ILayer>& partnerA, const std::vector<ILayer>& partnerB) const {
      std::unique_ptr<INeuralNetwork> neuralNetwork = std::make_unique<NeuralNetwork>();

      for (size_t i = 0; i < partnerA.size() && i < partnerB.size(); i++) {
        neuralNetwork->addLayer(crossover(partnerA[i], partnerB[i]));
      }

      return neuralNetwork;
    }

    FORCE_INLINE std::unique_ptr<ILayer> crossover(const ILayer& partnerA, const ILayer& partnerB) const {
      auto weights = crossoverWeights(partnerA.getWeights(), partnerB.getWeights());
      auto biases = crossoverBiases(partnerA.getBiases(), partnerB.getBiases());

      return std::make_unique<DenseLayer>(weights, biases, partnerA.getActivationFunction());
    }

    virtual layer_weight_t crossoverWeights(const layer_weight_t& partnerA, const layer_weight_t& partnerB) const;
    virtual layer_bias_t crossoverBiases(const layer_bias_t& partnerA, const layer_bias_t& partnerB) const;

    virtual std::unique_ptr<IPopulation> evolve(const IPopulation& population) const;
    virtual std::vector<IIndividual> evolve(const std::vector<IIndividual>& individuals) const;

    virtual const IIndividual* select(const IPopulation& population) const;
    virtual const IIndividual* select(const std::vector<IIndividual>& individuals) const;

    virtual void setOptions(const GeneticOptions&);
    virtual void setRate(float);
    virtual void setIntensity(float);
    virtual void setEliteCount(size_t);

    virtual const GeneticOptions& getOptions() const;
  };

}; // namespace neuro
