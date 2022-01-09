from __future__ import annotations

import dataclasses as dcl
import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Union, NamedTuple

from ordered_set import OrderedSet


import lernd.util as u

from lernd.lernd_types import Atom, Constant, GroundAtom, Predicate, RuleTemplate, Variable


class ClausesAndRule(NamedTuple):
  clauses: OrderedSet[Clause]
  rule: RuleTemplate


class Clause:

  def __init__(self, head: Atom, body: Tuple[Atom, ...]):
    self.head = head
    self.body = tuple(sorted(body))

  def __eq__(self, other):
    return self.head == other.head and self.body == other.body

  def __hash__(self):
    return hash(self.head) ^ hash(self.body)

  def __str__(self):
    return '{0}<-{1}'.format(
        u.atom2str(self.head), ', '.join(map(u.atom2str, self.body)))

  def __repr__(self):
    return f'"{self.__str__()}"'

  @classmethod
  def from_str(cls, s: str):
    head, body = s.split('<-')
    head = u.str2atom(head)
    body = map(str.strip, body.split(','))
    return cls(u.str2atom(head), tuple(map(u.str2atom, body)))

  def to_latex(self) -> str:
    return '${0}\\leftarrow {1}$'.format(
        u.atom2str(self.head), ', '.join(map(u.atom2str, self.body)))


@dcl.dataclass
class LanguageModel:
  target: Predicate
  preds_ext: List[Predicate]
  constants: List[Constant]


@dcl.dataclass
class ProgramTemplate:
  preds_aux: List[Predicate]
  rules: Dict[Predicate, Tuple[RuleTemplate, Optional[RuleTemplate]]]
  forward_chaining_steps: int


@dcl.dataclass
class ILP:
  name: str  # L, B, P, N
  language_model: LanguageModel
  background_axioms: List[GroundAtom]
  positive_examples: List[GroundAtom]
  negative_examples: List[GroundAtom]


class Substitution:

  def __init__(self, vs: Iterable[Variable], cs: Iterable[Constant]):
    self.subst = {var: const for var, const in zip(vs, cs)}
    self.subst_reverse = {const: var for var, const in zip(vs, cs)}

  def get_constant(self, c: Constant) -> Optional[Variable]:
    if c in self.subst_reverse:
      return self.subst_reverse[c]

  def add(self, v: Variable, c: Constant):
    self.subst[v] = c
    self.subst_reverse[c] = v

  def apply_to_atom(self, a: Atom) -> GroundAtom:
    m = MaybeGroundAtom.from_atom(a)
    m.apply_substitutions(self)
    return m.to_ground_atom()


class MaybeGroundAtom:

  def __init__(self, pred: Predicate, args, groundedness: Iterable[bool]):
    self._pred = pred
    self._groundedness = groundedness
    self._len = len(args)
    self._arg0 = (args[0], groundedness[0]) if self._len >= 1 else None
    self._arg1 = (args[1], groundedness[1]) if self._len >= 2 else None
    self._args = [self._arg0, self._arg1]

  def arg_at(self, index: int) -> Union[Constant, Variable]:
    return self._args[index][0]

  def const_at(self, index: int) -> bool:
    return self._args[index][1]

  def is_ground(self) -> bool:
    return (self._args[0] is None or
            self._args[0][1]) and (self._args[1] is None or self._args[1][1])

  def to_ground_atom(self) -> GroundAtom:
    if self.is_ground():
      return GroundAtom(
          self._pred, tuple((self.arg_at(i) for i in range(self._len))))
    raise Exception('TODO: something better')

  def apply_substitutions(
      self, substitutions: Union[Substitution, Dict[Variable, Constant]]):
    if isinstance(substitutions, Substitution):
      substitutions = substitutions.subst
    for i in range(self._len):
      if not self.const_at(i) and self.arg_at(i) in substitutions:
        self._args[i] = (substitutions[self.arg_at(i)], True)

  @property
  def pred(self) -> Predicate:
    return self._pred

  @property
  def arity(self) -> int:
    return u.arity(self._pred)

  @classmethod
  def from_ground_atom(cls, ground_atom: GroundAtom):
    pred, args = ground_atom
    groundedness = [True] * u.arity(pred)
    return cls(pred, args, groundedness)

  @classmethod
  def from_atom(cls, atom: Atom):
    pred, args = atom
    groundedness = [False] * u.arity(pred)
    return cls(pred, args, groundedness)

  @classmethod
  def from_pred(cls, pred: Predicate):
    args = [Variable(f'tmp{i}') for i in range(u.arity(pred))]
    groundedness = [False] * u.arity(pred)
    return cls(pred, args, groundedness)

  def __str__(self):
    string = ''
    string += self._pred[0] + '('
    for i in range(self.arity):
      string += self._args[i][0]
    string += ')'
    return string

  def copy(self):
    return type(self)(
        self._pred, [self.arg_at(i) for i in range(self._len)],
        [self.const_at(i) for i in range(self._len)])


class GroundAtoms:

  def __init__(
      self, language_model: LanguageModel, program_template: ProgramTemplate):
    self._constants = OrderedSet(language_model.constants)
    self._number_of_constants = len(language_model.constants)
    self._preds = language_model.preds_ext + program_template.preds_aux + [
        language_model.target
    ]
    preds = self._preds

    # First element (index 0) is falsum
    # key: predicate,
    # value: index of predicate's first ground atom (amongst all ground atoms)
    self._ground_atom_base_index = {preds[0]: 1}

    for i in range(1, len(preds)):
      prev_pred = preds[i - 1]
      pred = preds[i]
      self._ground_atom_base_index[pred] = (
          self._ground_atom_base_index[prev_pred] +
          len(self._constants)**u.arity(prev_pred))
    self.len = (
        self._ground_atom_base_index[preds[-1]] +
        len(self._constants)**u.arity(preds[-1]))

  def __len__(self):
    return self.len

  def ground_atom_generator(
      self, maybe_ground_atom: MaybeGroundAtom
  ) -> Iterable[Tuple[GroundAtom, Dict[Variable, Constant]]]:
    # TODO: maybe doesn't need to return ground_atoms at all?
    if maybe_ground_atom.is_ground():
      return [(maybe_ground_atom.to_ground_atom(), {})]
    pred = maybe_ground_atom.pred
    arity = maybe_ground_atom.arity
    if arity == 1:
      return ((GroundAtom(pred, (c,)), {
          maybe_ground_atom.arg_at(0): c
      }) for c in self._constants)
    elif arity == 2:
      if maybe_ground_atom.const_at(0):
        return ((
            GroundAtom(pred, (maybe_ground_atom.arg_at(0), c)), {
                maybe_ground_atom.arg_at(1): c
            }) for c in self._constants)
      elif maybe_ground_atom.const_at(1):
        return ((
            GroundAtom(pred, (c, maybe_ground_atom.arg_at(1))), {
                maybe_ground_atom.arg_at(0): c
            }) for c in self._constants)
      else:
        if not maybe_ground_atom.const_at(0) and not maybe_ground_atom.const_at(
            1) and maybe_ground_atom.arg_at(0) == maybe_ground_atom.arg_at(1):
          return ((GroundAtom(pred, (c, c)), {
              maybe_ground_atom.arg_at(0): c
          }) for c in self._constants)
        else:
          return ((
              GroundAtom(pred, (c1, c2)), {
                  maybe_ground_atom.arg_at(0): c1,
                  maybe_ground_atom.arg_at(1): c2
              }) for c1,
                  c2 in itertools.product(self._constants, repeat=arity))
    else:
      raise Exception("TODO: something better")

  def all_ground_atom_generator(self) -> Iterable[GroundAtom]:
    for pred in self._preds:
      arity = u.arity(pred)
      if arity == 0:
        yield GroundAtom(pred, ())
      elif arity == 1:
        for c in self._constants:
          yield GroundAtom(pred, (c,))
      elif arity == 2:
        for c1, c2 in itertools.product(self._constants, repeat=2):
          yield GroundAtom(pred, (c1, c2))

  def get_ground_atom_index(self, ground_atom: GroundAtom) -> int:
    pred, consts = ground_atom
    if u.arity(pred) == 0:
      return self._ground_atom_base_index[pred]
    elif u.arity(pred) == 1:
      return self._ground_atom_base_index[pred] + self._constants.map[consts[0]]
    elif u.arity(pred) == 2:
      return (
          self._ground_atom_base_index[pred]
          + self._constants.map[consts[0]] * self._number_of_constants
          + self._constants.map[consts[1]])
    raise Exception("TODO: something better")

