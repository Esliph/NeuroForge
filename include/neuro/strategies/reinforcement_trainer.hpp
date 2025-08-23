#pragma once

#include <vector>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"
#include "neuro/types.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  struct ReinforcementOptions {
    float learningRate = 0.01f;
    float discountFactor = 0.99f;
    float explorationRate = 0.1f;
    float explorationDecay = 0.99f;
  };

  class ReinforcementTrainer : public IStrategyEvolution {
    ReinforcementOptions options{};

   public:
    ReinforcementTrainer() = default;

    ReinforcementTrainer(float learningRate, float discountFactor = 0.99f, float explorationRate = 0.1f, float explorationDecay = 0.99f);
    ReinforcementTrainer(const ReinforcementOptions& options);

    virtual ~ReinforcementTrainer() = default;

    virtual void train(
      IIndividual& individual,
      const neuro_layer_t& state,
      int action,
      float reward,
      const neuro_layer_t& nextState,
      bool done) const;

    virtual void train(
      INeuralNetwork& network,
      const neuro_layer_t& state,
      int action,
      float reward,
      const neuro_layer_t& nextState,
      bool done) const;

    virtual void train(std::vector<ILayer>& layers,
                       const neuro_layer_t& state,
                       int action,
                       float reward,
                       const neuro_layer_t& nextState,
                       bool done,
                       const ActivationFunction& activation) const;

    virtual void train(std::vector<ILayer>& layers,
                       const neuro_layer_t& state,
                       int action,
                       float reward,
                       const neuro_layer_t& nextState,
                       bool done,
                       const std::vector<ActivationFunction>& activations) const;

    virtual void train(std::vector<layer_weight_t>& weights,
                       std::vector<layer_bias_t>& biases,
                       const neuro_layer_t& state,
                       int action,
                       float reward,
                       const neuro_layer_t& nextState,
                       bool done,
                       const ActivationFunction& activation) const;

    virtual void train(std::vector<layer_weight_t>& weights,
                       std::vector<layer_bias_t>& biases,
                       const neuro_layer_t& state,
                       int action,
                       float reward,
                       const neuro_layer_t& nextState,
                       bool done,
                       const std::vector<ActivationFunction>& activations) const;

    virtual int selectAction(const IIndividual& individual, const neuro_layer_t& state) const;
    virtual int selectAction(const INeuralNetwork& network, const neuro_layer_t& state) const;
    virtual int selectAction(const std::vector<ILayer>& layers, const neuro_layer_t& state, const ActivationFunction& activation) const;
    virtual int selectAction(const std::vector<ILayer>& layers,
                             const neuro_layer_t& state,
                             const std::vector<ActivationFunction>& activations) const;

    virtual int selectAction(const std::vector<layer_weight_t>& weights,
                             const std::vector<layer_bias_t>& biases,
                             const neuro_layer_t& state,
                             const ActivationFunction& activation) const;

    virtual int selectAction(const std::vector<layer_weight_t>& weights,
                             const std::vector<layer_bias_t>& biases,
                             const neuro_layer_t& state,
                             const std::vector<ActivationFunction>& activations) const;

    virtual void setOptions(const ReinforcementOptions&);
    virtual void setLearningRate(float);
    virtual void setDiscountFactor(float);
    virtual void setExplorationRate(float);
    virtual void setExplorationDecay(float);

    virtual const ReinforcementOptions& getOptions() const;
  };

}; // namespace neuro
