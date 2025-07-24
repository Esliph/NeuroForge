#include <doctest/doctest.h>

#include "neuro/makers/makers.hpp"

TEST_CASE("Activation - Sigmoid") {
  auto sigmoid(neuro::maker::makeSigmoid());

  CHECK(sigmoid.activate(0.0f) == doctest::Approx(0.5f));
  CHECK(sigmoid.activate(100.0f) == doctest::Approx(1.0f).epsilon(0.01));
  CHECK(sigmoid.activate(-100.0f) == doctest::Approx(0.0f).epsilon(0.01));

  float out = sigmoid.activate(0.5f);
  CHECK(sigmoid.derivate(out) == doctest::Approx(out * (1.0f - out)));
}

TEST_CASE("Activation - ReLU") {
  auto relu(neuro::maker::makeRelu());

  CHECK(relu.activate(-1.0f) == doctest::Approx(0.0f));
  CHECK(relu.activate(0.0f) == doctest::Approx(0.0f));
  CHECK(relu.activate(2.5f) == doctest::Approx(2.5f));

  CHECK(relu.derivate(0.0f) == doctest::Approx(0.0f));
  CHECK(relu.derivate(2.5f) == doctest::Approx(1.0f));
}

TEST_CASE("Activation - Tanh") {
  auto tanh_fn(neuro::maker::makeTanh_fn());

  CHECK(tanh_fn.activate(0.0f) == doctest::Approx(0.0f));
  CHECK(tanh_fn.activate(10.0f) == doctest::Approx(1.0f).epsilon(0.01));
  CHECK(tanh_fn.activate(-10.0f) == doctest::Approx(-1.0f).epsilon(0.01));

  float out = tanh_fn.activate(0.7f);
  CHECK(tanh_fn.derivate(out) == doctest::Approx(1.0f - out * out));
}

TEST_CASE("Activation - LeakyReLU") {
  auto leaky(neuro::maker::makeLeaky_relu());

  CHECK(leaky.activate(-1.0f) == doctest::Approx(-0.01f));
  CHECK(leaky.activate(0.0f) == doctest::Approx(0.0f));
  CHECK(leaky.activate(2.0f) == doctest::Approx(2.0f));

  CHECK(leaky.derivate(-1.0f) == doctest::Approx(0.01f));
  CHECK(leaky.derivate(2.0f) == doctest::Approx(1.0f));
}

TEST_CASE("Activation - ELU") {
  auto elu(neuro::maker::makeElu());

  CHECK(elu.activate(-1.0f) == doctest::Approx(std::exp(-1.0f) - 1.0f));
  CHECK(elu.activate(0.0f) == doctest::Approx(0.0f));
  CHECK(elu.activate(2.0f) == doctest::Approx(2.0f));

  float y = elu.activate(-1.0f);
  CHECK(elu.derivate(y) == doctest::Approx(y + 1.0f));
  CHECK(elu.derivate(2.0f) == doctest::Approx(1.0f));
}

TEST_CASE("Activation - Swish") {
  auto swish(neuro::maker::makeSwish());

  float x = 1.0f;
  float expected = x / (1.0f + std::exp(-x));
  CHECK(swish.activate(x) == doctest::Approx(expected));

  float y = expected;
  float expected_derivative = y + (1.0f - y) * y;
  CHECK(swish.derivate(y) == doctest::Approx(expected_derivative));
}

TEST_CASE("Activation - Softplus") {
  auto softplus(neuro::maker::makeSoftplus());

  float x = 1.0f;
  CHECK(softplus.activate(x) == doctest::Approx(std::log1p(std::exp(x))));

  float y = softplus.activate(x);
  CHECK(softplus.derivate(y) == doctest::Approx(1.0f - std::exp(-y)));
}

TEST_CASE("Activation - Hard Sigmoid") {
  auto hard_sigmoid(neuro::maker::makeHard_sigmoid());

  CHECK(hard_sigmoid.activate(-10.0f) == doctest::Approx(0.0f));
  CHECK(hard_sigmoid.activate(0.0f) == doctest::Approx(0.5f));
  CHECK(hard_sigmoid.activate(10.0f) == doctest::Approx(1.0f));

  CHECK(hard_sigmoid.derivate(0.0f) == doctest::Approx(0.0f));
  CHECK(hard_sigmoid.derivate(0.5f) == doctest::Approx(0.2f));
  CHECK(hard_sigmoid.derivate(-1.0f) == doctest::Approx(0.0f));
  CHECK(hard_sigmoid.derivate(1.0f) == doctest::Approx(0.0f));
}

TEST_CASE("Activation - Identity") {
  auto identity(neuro::maker::makeIdentity());

  CHECK(identity.activate(-10.0f) == doctest::Approx(-10.0f));
  CHECK(identity.activate(0.0f) == doctest::Approx(0.0f));
  CHECK(identity.activate(10.0f) == doctest::Approx(10.0f));

  CHECK(identity.derivate(0.0f) == doctest::Approx(1.0f));
  CHECK(identity.derivate(0.5f) == doctest::Approx(1.0f));
  CHECK(identity.derivate(-1.0f) == doctest::Approx(1.0f));
  CHECK(identity.derivate(1.0f) == doctest::Approx(1.0f));
}
