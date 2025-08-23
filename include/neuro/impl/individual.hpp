#pragma once

#include <functional>
#include <memory>

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

    void evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) override;

    const INeuralNetwork& getNeuralNetwork() const override;
    INeuralNetwork& getNeuralNetwork() override;

    void setNeuralNetwork(const INeuralNetwork& neuralNetwork) override;
    void setNeuralNetwork(std::unique_ptr<INeuralNetwork> neuralNetwork) override;

    float getFitness() const override;

    void setFitness(float fitness) override;

    std::unique_ptr<IIndividual> clone() const override;
  };

}; // namespace neuro
