from collections import defaultdict
from typing import Dict, List, OrderedDict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ordered_set import OrderedSet
from tqdm import trange

from lernd.classes import (
  Clause,
  GroundAtoms,
  LanguageModel,
  MaybeGroundAtom,
  ProgramTemplate,
)
from lernd.lernd_types import Constant, GroundAtom, Predicate, RuleTemplate

ClauseAndRule = Tuple[OrderedSet[Clause], RuleTemplate]
ClausesDict = Dict[Predicate, Tuple[ClauseAndRule, ClauseAndRule]]


class Inferrer:

  def __init__(
      self,
      ground_atoms: GroundAtoms,
      language_model: LanguageModel,
      clauses: ClausesDict,
      program_template: ProgramTemplate):
    self.xc_tensors = {}
    self.ground_atoms = ground_atoms
    self.lm = language_model
    self.clauses = clauses
    self.forward_chaining_steps = program_template.forward_chaining_steps
    self.xc_tensors = self._init_tensors()

  def _init_tensors(self) -> Dict[Predicate, List[List[jax.Array]]]:
    print('Inferrer initializing xc tensors...')
    tensors = defaultdict(list)
    for pred, clause_pair in self.clauses.items():
      for clauses_and_rule in clause_pair:
        if clauses_and_rule.rule is not None:
          tensors[pred].append([
              make_xc_tensor(xc, self.lm.constants, clauses_and_rule.rule, self.ground_atoms)
              for xc in [make_xc(c, self.ground_atoms) for c in clauses_and_rule.clauses]])
    return tensors

  def f_infer(
      self,
      a: jax.Array,
      weights: OrderedDict[Predicate, jax.Array]) -> jax.Array:
    softmaxes = {}
    for pred in self.xc_tensors.keys():
      pred_weights = jnp.reshape(weights[pred], [-1])
      softmaxes[pred] = jax.nn.softmax(pred_weights)[:, np.newaxis]

    for t in trange(self.forward_chaining_steps):
      bt = jnp.zeros_like(a)
      for pred, tensors in self.xc_tensors.items():
        c_p = []
        f1jps = [fc(a, tensor) for tensor in tensors[0]]
        f2kps = [fc(a, tensor) for tensor in tensors[1]
                ] if len(tensors) == 2 else None
        # Check if they're the same
        # assert all(map(lambda x: np.array_equal(x[0].numpy(), x[1].numpy()), zip(f1jps, f2kps)))
        if f2kps is not None:
          for f1jp in f1jps:
            for f2kp in f2kps:
              c_p.append(jnp.max(f1jp, f2kp))
        else:
          c_p = f1jps

        bt += jnp.sum(
          jnp.multiply(jnp.stack(c_p), softmaxes[pred]),
          axis=0)
      # f_amalgamate - probabilistic sum (t-conorm)
      a = a + bt - jnp.multiply(a, bt)
    return a


def make_xc(
    c: Clause, ground_atoms: GroundAtoms
) -> List[Tuple[GroundAtom, List[Tuple[int, int]]]]:
  xc = []
  head_pred, head_vars = c.head
  atom1, atom2 = c.body

  atom_matches_head = MaybeGroundAtom.from_atom(c.head)
  for ground_head, _ in ground_atoms.ground_atom_generator(atom_matches_head):
    pairs = []
    ground_head_consts = ground_head[1]
    substitutions = dict(zip(head_vars, ground_head_consts))
    a1 = MaybeGroundAtom.from_atom(atom1)
    a1.apply_substitutions(substitutions)
    a2 = MaybeGroundAtom.from_atom(atom2)
    a2.apply_substitutions(substitutions)
    for ground_atom1, new_subst1 in ground_atoms.ground_atom_generator(a1):
      a2_ = a2.copy()
      a2_.apply_substitutions(new_subst1)
      for ground_atom2, new_subst2 in ground_atoms.ground_atom_generator(a2_):
        i1 = ground_atoms.get_ground_atom_index(ground_atom1)
        i2 = ground_atoms.get_ground_atom_index(ground_atom2)
        pairs.append((i1, i2))
    xc.append((ground_head, pairs))
  return xc


def make_xc_tensor(
    xc: List[Tuple[GroundAtom, List[Tuple[int, int]]]],
    constants: List[Constant],
    tau: RuleTemplate,
    ground_atoms: GroundAtoms) -> jax.Array:
  """Returns a tensor of indices."""
  n = ground_atoms.len
  v = tau[0]
  w = len(constants) ** v
  xc_tensor = jnp.zeros((n, w, 2), dtype=jnp.int32)
  for ground_atom, xk_indices in xc:
    index = ground_atoms.get_ground_atom_index(ground_atom)
    if xk_indices:
      xc_tensor = xc_tensor.at[index].set(jnp.array(pad_indices(xk_indices, w)))
  return xc_tensor


def pad_indices(
    indices: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
  nidxs, rest = len(indices), max(0, n - len(indices))
  if nidxs == n:
    return indices
  return [indices[i] for i in range(nidxs)] + [(0, 0) for _ in range(rest)]


def fc(a: jax.Array, xc_tensor: jax.Array) -> jax.Array:
  x1 = xc_tensor[:, :, 0]
  x2 = xc_tensor[:, :, 1]
  y1 = a[x1]
  y2 = a[x2]
  # fuzzy_and - product t-norm, element-wise multiplication
  return jnp.max(jnp.multiply(y1, y2), axis=1)

