#pragma once

#include <functional>
#include <memory>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

class Individual : public IIndividual {
  std::unique_ptr<INeuralNetwork> neuralNetwork{};
  float fitness{};

 public:
  Individual() = default;
  Individual(const Individual&);
  Individual(int fitness);
  Individual(std::unique_ptr<INeuralNetwork>);
  Individual(std::unique_ptr<INeuralNetwork>, int fitness);
  Individual(const std::vector<int>& structure, const ActivationFunction& activation);
  Individual(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

  virtual ~Individual() = default;

  void evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) override;
  neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

  void setNeuralNetwork(const INeuralNetwork&) override;
  void setNeuralNetwork(std::unique_ptr<INeuralNetwork>) override;

  void setFitness(float) override;

  const INeuralNetwork& getNeuralNetwork() const override;
  INeuralNetwork& getNeuralNetwork() override;

  float getFitness() const override;

  std::vector<layer_weight_t> getAllWeights() const override;
  std::vector<layer_bias_t> getAllBiases() const override;

  neuro_layer_t operator()(const neuro_layer_t& inputs) const override;
  std::unique_ptr<IIndividual> clone() const override;
};

};  // namespace neuro
