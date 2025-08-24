#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/impl/neural_network.hpp"
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

    Individual(const INeuralNetwork&);

    Individual(std::unique_ptr<INeuralNetwork>);
    Individual(int fitness);

    Individual(std::unique_ptr<INeuralNetwork>, int fitness);

    Individual(const std::vector<int>& structure, const ActivationFunction& activation);
    Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

    virtual ~Individual() = default;

    FORCE_INLINE void evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) {
      fitness = evaluateFunction(*neuralNetwork);
    }

    FORCE_INLINE const INeuralNetwork& getNeuralNetwork() const {
      return *neuralNetwork;
    }

    FORCE_INLINE INeuralNetwork& getNeuralNetwork() {
      return *neuralNetwork;
    }

    FORCE_INLINE void setNeuralNetwork(const INeuralNetwork& neuralNetwork) {
      this->neuralNetwork = std::move(neuralNetwork.clone());
    }

    FORCE_INLINE void setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) {
      this->neuralNetwork = std::move(neuralNetwork);
    }

    FORCE_INLINE float getFitness() const {
      return fitness;
    }

    FORCE_INLINE void setFitness(float fitness) {
      this->fitness = fitness;
    }

    FORCE_INLINE std::unique_ptr<IIndividual> clone() const {
      return std::make_unique<Individual>(*this);
    };
  };

}; // namespace neuro
