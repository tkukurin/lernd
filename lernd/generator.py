from __future__ import annotations

import functools as ft
import itertools as it
import string
from operator import add
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from ordered_set import OrderedSet

from lernd.classes import Clause, LanguageModel, ProgramTemplate
from lernd.lernd_types import Atom, Predicate, RuleTemplate, Variable


def f_generate(t: ProgramTemplate, lm: LanguageModel) -> Dict[Predicate, ClausePair]:
  preds_intensional = t.preds_aux + [lm.target]
  return {
    p: ClausePair.cl(preds_intensional, lm.preds_ext, p, *t.rules[p])
    for p in preds_intensional
  }


class ClausePair(NamedTuple):
  c1: ClausesAndRule
  c2: ClausesAndRule

  @staticmethod
  def cl(
      preds_int: List[Predicate],
      preds_ext: List[Predicate],
      pred: Predicate,
      tau1: Optional[RuleTemplate],
      tau2: Optional[RuleTemplate]) -> ClausePair:
    c1 = ClausesAndRule.cl(preds_int, preds_ext, pred, tau1)
    c2 = ClausesAndRule.cl(preds_int, preds_ext, pred, tau2)
    return ClausePair(c1, c2)


class ClausesAndRule(NamedTuple):
  clauses: OrderedSet[Clause]
  rule: Optional[RuleTemplate]

  @staticmethod
  def cl(
      preds_int: List[Predicate],
      preds_ext: List[Predicate],
      pred: Predicate,
      tau: Optional[RuleTemplate]) -> ClausesAndRule:
    """Generates all possible clauses adhering to the restrictions.

    1. Only clauses of atoms involving free variables (no constants)
    2. Only predicates of arity 0-2.
    3. Exactly 2 atoms in the body.
    4. No unsafe (which have a variable used in the head but not the body)
    5. No circular (head atom appears in the body)
    6. No duplicate (same but different order of body atoms)
    7. None with an intensional predicate in the body if int flag is 0.
    """
    if tau is None:
      return ClausesAndRule(OrderedSet(), None)

    v, int_ = tau
    pred_arity = pred[1]
    total_vars = pred_arity + v

    assert total_vars <= len(
        string.ascii_uppercase
    ), 'Handling of more than 26 variables not implemented!'

    variables = [Variable(string.ascii_uppercase[i]) for i in range(total_vars)]
    head = Atom(pred, tuple([variables[i] for i in range(pred_arity)]))

    possible_preds = list(preds_ext) + (preds_int if int_ else [])

    clauses = OrderedSet()
    for p1, p2 in it.product(possible_preds, possible_preds):
      a1s = (Atom(p1, tuple(c)) for c in it.product(variables, repeat=p1[1]))
      a2s = (Atom(p2, tuple(c)) for c in it.product(variables, repeat=p2[1]))
      for atom1, atom2 in it.product(a1s, a2s):
        clause = Clause(head, (atom1, atom2))
        if any([ClauseRule.unsafe(clause),
                ClauseRule.circular(clause),
                ClauseRule.int_flag(clause, int_, preds_int)]):
          continue
        clauses.add(clause)
    return ClausesAndRule(clauses, tau)


class ClauseRule:

  @staticmethod
  def unsafe(clause: Clause) -> bool:
    """Unsafe clause: variable used in the head but not the body."""
    head_vars: List[Variable] = clause.head[1]
    preds: List[Atom] = list(clause.body)
    body_vars = ft.reduce(add, map(lambda x: list(x[1]), preds))
    return any(head_var not in body_vars for head_var in head_vars)

  @staticmethod
  def circular(clause: Clause) -> bool:
    return clause.head in clause.body

  @staticmethod
  def int_flag(clause: Clause, int_: bool, preds: List[Predicate]) -> bool:
    # When int_=True: require at least one intensional predicate in body
    # When int_=False: don't allow intensional predicates in body
    has_intensional = any(pred in preds for pred, vars in clause.body)
    if int_:
      return not has_intensional  # Filter if no intensional (require at least one)
    return has_intensional  # Filter if any intensional (don't allow)


# Wrapper functions for backward compatibility with tests
def check_clause_unsafe(clause: Clause) -> bool:
  return ClauseRule.unsafe(clause)


def check_circular(clause: Clause) -> bool:
  return ClauseRule.circular(clause)


# Convenience function for clause generation (used by tests)
def cl(
    preds_int: List[Predicate],
    preds_ext: List[Predicate],
    pred: Predicate,
    tau: Optional[RuleTemplate]) -> OrderedSet[Clause]:
  result = ClausesAndRule.cl(preds_int, preds_ext, pred, tau)
  return result.clauses if result else OrderedSet()

