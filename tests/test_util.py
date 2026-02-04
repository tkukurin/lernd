"""Tests for lernd.util module."""

import pytest

from lernd import util as u
from lernd.lernd_types import Atom, Constant, Predicate, Variable


class TestStringParsing:
    def test_arity(self):
        p = Predicate('p', 2)
        assert u.arity(p) == 2

    def test_atom2str(self):
        p = Predicate('p', 2)
        var1 = Variable('X')
        var2 = Variable('Y')
        pred = Atom(p, (var1, var2))
        assert u.atom2str(pred) == 'p(X,Y)'

    def test_ground_atom2str_roundtrip(self):
        string = 'r(a,a)'
        ground_atom = u.str2ground_atom(string)
        assert u.ground_atom2str(ground_atom) == string

    def test_str2pred(self):
        pred_str = 'q/2'
        pred = Predicate('q', 2)
        assert u.str2pred(pred_str) == pred

    @pytest.mark.parametrize("atom_str,expected_arity", [
        ('pred()', 0),
        ('q(X)', 1),
        ('pred1(X,Y,Z)', 3),
    ])
    def test_str2atom(self, atom_str, expected_arity):
        atom = u.str2atom(atom_str)
        assert atom[0][1] == expected_arity
        assert u.atom2str(atom) == atom_str

    def test_pred2str(self):
        pred = Predicate('test', 2)
        assert u.pred2str(pred) == 'test/2'
