from typing import NamedTuple, Tuple


class Constant(str):
  pass


class Predicate(NamedTuple):
  name: str
  arity: int

  def __str__(self):
    return f'{name}/{arity}'


class RuleTemplate(NamedTuple):
  v: int  # number of exist. quantified vars allowed in the clause
  int: bool  # whether intensional predicates are allowed


class Variable(NamedTuple):
  name: str


class Atom(NamedTuple):
  pred: Predicate
  vars: Tuple[Variable, ...]


class GroundAtom(NamedTuple):
  pred: Predicate
  consts: Tuple[Constant, ...]


class ClausesAndRule(NamedTuple):
  clauses: OrderedSet[Clause]
  rule: RuleTemplate

