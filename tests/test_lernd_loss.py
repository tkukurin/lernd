"""Tests for lernd.lernd_loss module."""

import pytest

from lernd import lernd_loss as l
from lernd import util as u
from lernd.classes import LanguageModel, ProgramTemplate
from lernd.lernd_types import Constant


class TestGetGroundAtoms:
    def test_get_ground_atoms(self):
        target_pred = u.str2pred('q/2')
        preds_ext = [u.str2pred('p/0')]
        preds_aux = [u.str2pred('t/1')]
        language_model = LanguageModel(
            target_pred, preds_ext, [Constant(x) for x in ['a', 'b', 'c']])
        program_template = ProgramTemplate(preds_aux, {}, 0)
        f = u.str2ground_atom
        expected_ground_atoms = [
            f('p()'),
            f('t(a)'),
            f('t(b)'),
            f('t(c)'),
            f('q(a,a)'),
            f('q(a,b)'),
            f('q(a,c)'),
            f('q(b,a)'),
            f('q(b,b)'),
            f('q(b,c)'),
            f('q(c,a)'),
            f('q(c,b)'),
            f('q(c,c)')
        ]
        assert l.get_ground_atoms(language_model, program_template) == expected_ground_atoms
