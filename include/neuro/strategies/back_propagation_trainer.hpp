#pragma once

#include <vector>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"
#include "neuro/types.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  struct BackPropagationOptions {
    float learningRate = 0.01f;
    float momentum = 0.0f;
    float minLoss = 0.001f;
    size_t maxEpochs = 1000;
  };

  class BackPropagationTrainer : public IStrategyEvolution {
    BackPropagationOptions options{};

   public:
    BackPropagationTrainer() = default;

    BackPropagationTrainer(float learningRate, float momentum = 0.0f, float minLoss = 0.001f, size_t maxEpochs = 1000);
    BackPropagationTrainer(const BackPropagationOptions& options);

    virtual ~BackPropagationTrainer() = default;

    virtual void train(IIndividual& individual,
                       const std::vector<neuro_layer_t>& inputs,
                       const std::vector<neuro_layer_t>& expectedOutputs) const;

    virtual void train(INeuralNetwork& network,
                       const std::vector<neuro_layer_t>& inputs,
                       const std::vector<neuro_layer_t>& expectedOutputs) const;

    virtual void train(std::vector<ILayer>& layers,
                       const std::vector<neuro_layer_t>& inputs,
                       const std::vector<neuro_layer_t>& expectedOutputs,
                       const ActivationFunction& activation) const;

    virtual void train(std::vector<ILayer>& layers,
                       const std::vector<neuro_layer_t>& inputs,
                       const std::vector<neuro_layer_t>& expectedOutputs,
                       const std::vector<ActivationFunction>& activations) const;

    virtual void train(std::vector<layer_weight_t>& weights,
                       std::vector<layer_bias_t>& biases,
                       const std::vector<neuro_layer_t>& inputs,
                       const std::vector<neuro_layer_t>& expectedOutputs,
                       const ActivationFunction& activation) const;

    virtual void train(std::vector<layer_weight_t>& weights,
                       std::vector<layer_bias_t>& biases,
                       const std::vector<neuro_layer_t>& inputs,
                       const std::vector<neuro_layer_t>& expectedOutputs,
                       const std::vector<ActivationFunction>& activations) const;

    virtual void setOptions(const BackPropagationOptions&);
    virtual void setLearningRate(float);
    virtual void setMomentum(float);
    virtual void setMinLoss(float);
    virtual void setMaxEpochs(size_t);

    virtual const BackPropagationOptions& getOptions() const;
  };

} // namespace neuro
