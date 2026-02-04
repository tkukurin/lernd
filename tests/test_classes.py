"""Tests for lernd.classes module."""

import pytest

from lernd import classes as c
from lernd import util as u
from lernd.classes import GroundAtoms, LanguageModel, MaybeGroundAtom, ProgramTemplate
from lernd.lernd_types import Atom, Constant, Predicate, RuleTemplate, Variable


class TestClause:
    def test_str(self):
        pred1 = Atom(Predicate('p', 2), (Variable('X'), Variable('Y')))
        pred2 = Atom(Predicate('q', 2), (Variable('X'), Variable('Z')))
        pred3 = Atom(Predicate('t', 2), (Variable('Y'), Variable('X')))
        clause = c.Clause(pred1, (pred2, pred3))
        assert str(clause) == 'p(X,Y)<-q(X,Z), t(Y,X)'

    def test_from_str_roundtrip(self):
        clause_strs = ['p(X,Y)<-q(X,Z), t(Y,X)']
        for clause_str in clause_strs:
            clause = c.Clause.from_str(clause_str)
            assert str(clause) == clause_str


class TestGroundAtoms:
    def test_all_ground_atom_generator(self, simple_language_model, simple_program_template):
        ground_atoms = GroundAtoms(simple_language_model, simple_program_template)
        f = u.str2ground_atom
        expected_ground_atoms = [
            f('p(a,a)'),
            f('p(a,b)'),
            f('p(b,a)'),
            f('p(b,b)'),
            f('q(a,a)'),
            f('q(a,b)'),
            f('q(b,a)'),
            f('q(b,b)'),
            f('r(a,a)'),
            f('r(a,b)'),
            f('r(b,a)'),
            f('r(b,b)')
        ]
        actual_ground_atoms = list(ground_atoms.all_ground_atom_generator())
        assert actual_ground_atoms == expected_ground_atoms

    def test_ground_atom_generator(self, simple_language_model, simple_program_template):
        target_pred = simple_language_model.target
        ground_atoms = GroundAtoms(simple_language_model, simple_program_template)
        maybe_ground_atom = MaybeGroundAtom.from_atom(
            Atom(target_pred, (Variable('C'), Variable('C'))))
        actual_ground_atoms = list(
            list(zip(*(ground_atoms.ground_atom_generator(maybe_ground_atom))))[0])
        f = u.str2ground_atom
        expected_ground_atoms = [f('r(a,a)'), f('r(b,b)')]
        assert actual_ground_atoms == expected_ground_atoms

    def test_index_uniqueness(self, ground_atoms):
        """Each ground atom should have a unique index."""
        indices = set()
        for ga in ground_atoms.all_ground_atom_generator():
            idx = ground_atoms.get_ground_atom_index(ga)
            assert idx not in indices, f"Duplicate index {idx}"
            indices.add(idx)

    def test_indices_are_contiguous(self, ground_atoms):
        """Indices should be 1 to len without gaps."""
        indices = sorted(
            ground_atoms.get_ground_atom_index(ga)
            for ga in ground_atoms.all_ground_atom_generator()
        )
        expected = list(range(1, len(indices) + 1))
        assert indices == expected
