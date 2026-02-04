"""Tests for lernd.inferrer module."""

import jax.numpy as jnp
import numpy as np
import pytest

from lernd import classes as c
from lernd import inferrer as i
from lernd import util as u
from lernd.classes import GroundAtoms, LanguageModel, ProgramTemplate
from lernd.lernd_types import Constant, RuleTemplate


class TestMakeXc:
    def test_make_xc(self, simple_language_model, simple_program_template):
        ground_atoms = GroundAtoms(simple_language_model, simple_program_template)
        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        f = u.str2ground_atom
        expected = [
            (f('r(a,a)'), [(1, 5), (2, 7)]),
            (f('r(a,b)'), [(1, 6), (2, 8)]),
            (f('r(b,a)'), [(3, 5), (4, 7)]),
            (f('r(b,b)'), [(3, 6), (4, 8)])
        ]
        assert i.make_xc(clause, ground_atoms) == expected


class TestMakeXcTensor:
    def test_make_xc_tensor(self, simple_language_model, simple_program_template):
        ground_atoms = GroundAtoms(simple_language_model, simple_program_template)
        f = u.str2ground_atom
        xc = [
            (f('p(a,a)'), []), (f('p(a,b)'), []), (f('p(b,a)'), []),
            (f('p(b,b)'), []), (f('q(a,a)'), []), (f('q(a,b)'), []),
            (f('q(b,a)'), []), (f('q(b,b)'), []),
            (f('r(a,a)'), [(1, 5), (2, 7)]),
            (f('r(a,b)'), [(1, 6), (2, 8)]),
            (f('r(b,a)'), [(3, 5), (4, 7)]),
            (f('r(b,b)'), [(3, 6), (4, 8)])
        ]
        constants = [Constant('a'), Constant('b')]
        tau = RuleTemplate(1, False)
        expected_tensor = np.array([
            [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)],
            [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)],
            [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)],
            [(1, 5), (2, 7)], [(1, 6), (2, 8)],
            [(3, 5), (4, 7)], [(3, 6), (4, 8)]
        ])
        result = i.make_xc_tensor(xc, constants, tau, ground_atoms)
        assert np.asarray(result).tolist() == expected_tensor.tolist()


class TestForwardChaining:
    def test_fc(self, predecessor_setup):
        language_model, program_template, ground_atoms = predecessor_setup
        a = jnp.array([0, 1, 0.9, 0, 0, 0.1, 0, 0.2, 0.8, 0, 0, 0, 0])
        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        tau = RuleTemplate(1, False)
        expected_a_apostrophe = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18, 0.72, 0, 0])

        xc = i.make_xc(clause, ground_atoms)
        xc_tensor = i.make_xc_tensor(
            xc, language_model.constants, tau, ground_atoms)
        a_apostrophe = i.fc(a, xc_tensor)

        np.testing.assert_array_almost_equal(
            np.asarray(a_apostrophe), expected_a_apostrophe)
