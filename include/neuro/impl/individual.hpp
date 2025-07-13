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

    FORCE_INLINE void setNeuralNetwork(const INeuralNetwork& neuralNetwork) override {
      this->neuralNetwork = std::move(neuralNetwork.clone());
    }

    FORCE_INLINE void setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) override {
      this->neuralNetwork = std::move(neuralNetwork);
    }

    FORCE_INLINE void setFitness(float fitness) override {
      this->fitness = fitness;
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

    FORCE_INLINE std::unique_ptr<IIndividual> clone() const override {
      return std::make_unique<Individual>(*this);
    };
  };

};  // namespace neuro
