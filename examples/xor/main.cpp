#define POPULATION_SIZE 100
#define GENERATIONS 100
#define MUTATION_RATE 0.2f
#define MUTATION_STRENGTH 0.8f
#define ELITE_COUNT 5

#include <iomanip>
#include <iostream>
#include <neuro/neuro.hpp>
#include <vector>

const std::vector<std::pair<neuro::neuro_layer_t, float>> DATASET = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 0.0f},
};

float evaluateIndividual(const neuro::IIndividual& individual) {
  float totalError = 0.0f;

  for (const auto& [input, expected] : DATASET) {
    float output = 0.0f;
    float error = expected - output;

    totalError += error * error;
  }

  return 1.0f / (1.0f + totalError);
}

int main() {
  std::vector<int> structure = {2, 4, 1};
  neuro::ActivationFunction activation = neuro::makeSigmoid();

  neuro::IPopulation* population = new neuro::Population(POPULATION_SIZE, structure, activation);

  population->randomizeWeights(-5, 5);
  population->randomizeBiases(-5, 5);

  for (int generation = 0; generation <= GENERATIONS; ++generation) {
    for (const auto& individual : (*population)) {
      float fitness = evaluateIndividual(*individual);

      individual->setFitness(fitness);
    }

    const auto& best = population->getBestIndividual();
    float bestFitness = best.getFitness();

    std::cout << "Generation " << std::setw(3) << generation << " | Best fitness: " << std::fixed << std::setprecision(6) << bestFitness << std::endl;

    // population->evolve(MUTATION_RATE, MUTATION_STRENGTH, ELITE_COUNT);
  }

  const auto& best = population->getBestIndividual();

  std::cout << std::endl
            << "Best individual (XOR):" << std::endl;

  for (const auto& [input, expected] : DATASET) {
    float output = best.getNeuralNetwork().feedforward(input)[0];

    std::cout << "Input: [" << input[0] << ", " << input[1] << "] | Expect: " << expected << " | Result: " << std::fixed << std::setprecision(4) << output << std::endl;
  }

  delete population;

  return 0;
}
