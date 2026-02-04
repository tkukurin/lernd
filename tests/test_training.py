"""Integration tests for training loop and gradient computation."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lernd.experiments import setup_predecessor
from lernd.lernd_loss import Lernd
from lernd.main import generate_weight_matrices


class TestGradientComputation:
    """Tests to ensure gradients are computed correctly."""

    @pytest.fixture
    def model_and_weights(self):
        """Set up a simple ILP problem for testing."""
        ilp, program_template = setup_predecessor()
        model = Lernd(ilp, program_template)
        weights = generate_weight_matrices(model.clauses)
        return model, weights

    def test_gradients_not_none(self, model_and_weights):
        """Gradients should not be None for any weight."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        _, grads = jax.value_and_grad(loss_fn)(weights)

        for pred, grad in grads.items():
            assert grad is not None, f"Gradient for {pred} is None"

    def test_gradients_not_zero(self, model_and_weights):
        """Gradients should not be all zeros (indicates broken backprop)."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        _, grads = jax.value_and_grad(loss_fn)(weights)

        for pred, grad in grads.items():
            grad_norm = jnp.linalg.norm(grad)
            assert grad_norm > 0, f"Gradient for {pred} is all zeros"

    def test_gradients_finite(self, model_and_weights):
        """Gradients should not contain NaN or Inf."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        _, grads = jax.value_and_grad(loss_fn)(weights)

        for pred, grad in grads.items():
            assert jnp.all(jnp.isfinite(grad)), f"Gradient for {pred} contains NaN/Inf"

    def test_gradients_same_structure_as_weights(self, model_and_weights):
        """Gradient structure should match weight structure."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        _, grads = jax.value_and_grad(loss_fn)(weights)

        assert set(grads.keys()) == set(weights.keys()), "Gradient keys don't match weight keys"

        for pred in weights.keys():
            assert grads[pred].shape == weights[pred].shape, \
                f"Gradient shape {grads[pred].shape} != weight shape {weights[pred].shape} for {pred}"


class TestTrainingLoop:
    """Tests to ensure training actually learns."""

    @pytest.fixture
    def model_and_weights(self):
        """Set up a simple ILP problem for testing."""
        ilp, program_template = setup_predecessor()
        model = Lernd(ilp, program_template)
        weights = generate_weight_matrices(model.clauses)
        return model, weights

    def test_loss_decreases(self, model_and_weights):
        """Loss should decrease after a few training steps."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        initial_loss = float(loss_fn(weights))

        # Train for a few steps
        opt = optax.rmsprop(0.5)
        opt_state = opt.init(weights)

        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(weights)
            updates, opt_state = opt.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)

        final_loss = float(loss_fn(weights))

        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss} -> {final_loss}"

    def test_predictions_improve(self, model_and_weights):
        """Predictions for positive examples should increase after training."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        # Get initial predictions for positive examples
        initial_valuation = model(weights)
        pos_indices, _ = model.big_lambda
        initial_pos_mean = float(jnp.mean(initial_valuation[pos_indices]))

        # Train for a few steps
        opt = optax.rmsprop(0.5)
        opt_state = opt.init(weights)

        for _ in range(20):
            loss, grads = jax.value_and_grad(loss_fn)(weights)
            updates, opt_state = opt.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)

        # Get final predictions
        final_valuation = model(weights)
        final_pos_mean = float(jnp.mean(final_valuation[pos_indices]))

        assert final_pos_mean > initial_pos_mean, \
            f"Positive example predictions did not improve: {initial_pos_mean} -> {final_pos_mean}"

    def test_convergence(self, model_and_weights):
        """Model should converge to low loss on predecessor problem."""
        model, weights = model_and_weights

        def loss_fn(w):
            valuation = model(w)
            return model.loss(valuation, w)

        opt = optax.rmsprop(0.5)
        opt_state = opt.init(weights)

        # Train until convergence or max steps
        for _ in range(100):
            loss, grads = jax.value_and_grad(loss_fn)(weights)
            updates, opt_state = opt.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)

        final_loss = float(loss_fn(weights))

        # Predecessor is a simple problem, should converge to very low loss
        assert final_loss < 0.01, f"Did not converge: final loss = {final_loss}"
