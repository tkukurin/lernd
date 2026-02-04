"""Property-based tests using Hypothesis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings

from lernd import util as u
from lernd.lernd_types import Constant, Predicate, GroundAtom


# Strategies for generating valid inputs
predicate_names = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz',
    min_size=1, max_size=10
)

arities = st.integers(min_value=0, max_value=2)

constant_names = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz0123456789',
    min_size=1, max_size=5
)


class TestStringParsingRoundTrips:
    @given(name=predicate_names, arity=arities)
    def test_pred_roundtrip(self, name, arity):
        """str2pred(pred2str(pred)) == pred"""
        pred = Predicate(name, arity)
        pred_str = u.pred2str(pred)
        parsed = u.str2pred(pred_str)
        assert parsed == pred

    @given(
        name=predicate_names,
        arity=arities,
        const_names=st.lists(constant_names, min_size=0, max_size=2)
    )
    def test_ground_atom_roundtrip(self, name, arity, const_names):
        """str2ground_atom(ground_atom2str(ga)) == ga"""
        assume(len(const_names) == arity)
        consts = tuple(Constant(c) for c in const_names)
        pred = Predicate(name, arity)
        ga = GroundAtom(pred, consts)
        ga_str = u.ground_atom2str(ga)
        parsed = u.str2ground_atom(ga_str)
        assert parsed == ga


class TestNumericalStability:
    @settings(max_examples=50)
    @given(
        probs=st.lists(
            st.floats(min_value=1e-10, max_value=1.0-1e-10),
            min_size=4, max_size=10
        ),
        labels=st.lists(
            st.sampled_from([0.0, 1.0]),
            min_size=4, max_size=10
        )
    )
    def test_bce_loss_finite(self, probs, labels):
        """Binary cross-entropy should always produce finite values."""
        assume(len(probs) == len(labels))
        preds = jnp.array(probs)
        targets = jnp.array(labels)
        # Binary cross-entropy with epsilon
        loss = -jnp.mean(
            targets * jnp.log(preds + 1e-12) +
            (1 - targets) * jnp.log(1 - preds + 1e-12)
        )
        assert jnp.isfinite(loss)

    @given(weights=st.lists(st.floats(min_value=-100, max_value=100), min_size=2, max_size=10))
    def test_softmax_stability(self, weights):
        """Softmax should be numerically stable."""
        w = jnp.array(weights)
        probs = jax.nn.softmax(w)
        assert jnp.all(jnp.isfinite(probs))
        assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-5)

    @given(
        a=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=10),
        b=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=10)
    )
    def test_probabilistic_sum_bounds(self, a, b):
        """Probabilistic sum (t-conorm) should stay in [0, 1]."""
        assume(len(a) == len(b))
        arr_a = jnp.array(a)
        arr_b = jnp.array(b)
        # t-conorm: a + b - a*b
        result = arr_a + arr_b - jnp.multiply(arr_a, arr_b)
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)
