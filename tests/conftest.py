"""Pytest fixtures for LERND tests."""

import pytest

from lernd.classes import GroundAtoms, LanguageModel, ProgramTemplate
from lernd.lernd_types import Constant, RuleTemplate
from lernd.util import str2pred


@pytest.fixture
def simple_language_model():
    """Two binary predicates with two constants."""
    target_pred = str2pred('r/2')
    preds_ext = [str2pred('p/2'), str2pred('q/2')]
    constants = [Constant('a'), Constant('b')]
    return LanguageModel(target_pred, preds_ext, constants)


@pytest.fixture
def simple_program_template():
    """Minimal program template with no aux predicates."""
    return ProgramTemplate(preds_aux=[], rules={}, forward_chaining_steps=0)


@pytest.fixture
def ground_atoms(simple_language_model, simple_program_template):
    """Ground atoms for simple language model."""
    return GroundAtoms(simple_language_model, simple_program_template)


@pytest.fixture
def predecessor_setup():
    """Setup for the predecessor ILP problem."""
    target_pred = str2pred('r/2')
    preds_ext = [str2pred('p/2'), str2pred('q/2')]
    preds_aux = []
    language_model = LanguageModel(
        target_pred, preds_ext, [Constant('a'), Constant('b')])
    rules = {target_pred: (RuleTemplate(1, True), RuleTemplate(1, False))}
    program_template = ProgramTemplate(preds_aux, rules, 0)
    ground_atoms = GroundAtoms(language_model, program_template)
    return language_model, program_template, ground_atoms
