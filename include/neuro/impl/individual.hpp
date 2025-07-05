#pragma once

#include <functional>
#include <memory>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class Individual : public IIndividual {
    std::unique_ptr<INeuralNetwork> neuralNetwork;
    float fitness{};

   public:
    Individual();
    Individual(const Individual&);

    Individual(int fitness);
    Individual(std::unique_ptr<INeuralNetwork>);
    Individual(std::unique_ptr<INeuralNetwork>, int fitness);
    Individual(const std::vector<int>& structure, const ActivationFunction& activation);
    Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

    virtual ~Individual() = default;

    FORCE_INLINE void evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) override {
      fitness = evaluateFunction(*neuralNetwork);
    }

    FORCE_INLINE void randomizeWeights(float min, float max) override {
      neuralNetwork->randomizeWeights(min, max);
    }

    FORCE_INLINE void randomizeBiases(float min, float max) override {
      neuralNetwork->randomizeBiases(min, max);
    }

    FORCE_INLINE neuro_layer_t feedforward(const neuro_layer_t& inputs) const override {
      return neuralNetwork->feedforward(inputs);
    }

    FORCE_INLINE size_t inputSize() const override {
      return neuralNetwork->inputSize();
    }

    FORCE_INLINE size_t outputSize() const override {
      return neuralNetwork->outputSize();
    }

    FORCE_INLINE void setNeuralNetwork(const INeuralNetwork& neuralNetwork) override {
      this->neuralNetwork = std::move(neuralNetwork.clone());
    }

    FORCE_INLINE void setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) override {
      this->neuralNetwork = std::move(neuralNetwork);
    }

    FORCE_INLINE void setFitness(float fitness) override {
      this->fitness = fitness;
    }

    FORCE_INLINE void setAllWeights(const std::vector<layer_weight_t>& weights) override {
      neuralNetwork->setAllWeights(weights);
    }

    FORCE_INLINE void setAllBiases(const std::vector<layer_bias_t>& biases) override {
      neuralNetwork->setAllBiases(biases);
    }

    FORCE_INLINE void setLayers(std::vector<std::unique_ptr<ILayer>> layers) override {
      neuralNetwork->setLayers(std::move(layers));
    }

    FORCE_INLINE const INeuralNetwork& getNeuralNetwork() const override {
      return *neuralNetwork;
    }

    FORCE_INLINE INeuralNetwork& getNeuralNetwork() override {
      return *neuralNetwork;
    }

    FORCE_INLINE float getFitness() const override {
      return fitness;
    }

    FORCE_INLINE std::vector<layer_weight_t> getAllWeights() const override {
      return neuralNetwork->getAllWeights();
    }

    FORCE_INLINE std::vector<layer_bias_t> getAllBiases() const override {
      return neuralNetwork->getAllBiases();
    }

    FORCE_INLINE size_t sizeLayers() const override {
      return neuralNetwork->sizeLayers();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const override {
      return neuralNetwork->begin();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator begin() override {
      return neuralNetwork->begin();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator end() const override {
      return neuralNetwork->end();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator end() override {
      return neuralNetwork->end();
    }

    FORCE_INLINE neuro_layer_t operator()(const neuro_layer_t& inputs) const override {
      return feedforward(inputs);
    }

    FORCE_INLINE const ILayer& operator[](int index) const override {
      return (*neuralNetwork)[index];
    }

    FORCE_INLINE ILayer& operator[](int index) override {
      return (*neuralNetwork)[index];
    }

    FORCE_INLINE std::unique_ptr<IIndividual> clone() const override {
      return std::make_unique<Individual>(*this);
    };
  };

};  // namespace neuro
