import itertools as it
from typing import Dict, List, Optional, OrderedDict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ordered_set import OrderedSet

import lernd.util as u
from lernd.classes import ILP, GroundAtoms, LanguageModel, ProgramTemplate
from lernd.generator import f_generate
from lernd.inferrer import Inferrer
from lernd.lernd_types import GroundAtom, Predicate, RuleTemplate


def all_variables(weights):
  return [weights_ for weights_ in weights.values()]


def f_convert(
    background_axioms: List[GroundAtom],
    ground_atoms: GroundAtoms) -> jax.Array:
  # non-differentiable operation
  # order must be the same as in ground_atoms
  all_ground_atoms = ground_atoms.all_ground_atom_generator()
  return jnp.array(
      [0] + [int(gamma in background_axioms) for gamma in all_ground_atoms],
      dtype=jnp.float32)


def get_ground_atoms(
    language_model: LanguageModel,
    program_template: ProgramTemplate) -> List[GroundAtom]:
  preds_ext = language_model.preds_ext
  preds_aux = program_template.preds_aux
  preds = preds_ext + preds_aux + [language_model.target]
  ground_atoms = []
  for pred in preds:
    const_combs = it.product(language_model.constants, repeat=u.arity(pred))
    ground_atoms.extend(GroundAtom(pred, cc) for cc in const_combs)
  return ground_atoms


def make_lambda(
    positive_examples: List[GroundAtom],
    negative_examples: List[GroundAtom],
    ground_atoms: GroundAtoms) -> Tuple[jax.Array, jax.Array]:
  example_indices = []
  example_values = []
  for ground_atom in positive_examples:
    example_indices.append(ground_atoms.get_ground_atom_index(ground_atom))
    example_values.append(1)
  for ground_atom in negative_examples:
    example_indices.append(ground_atoms.get_ground_atom_index(ground_atom))
    example_values.append(0)
  return (
      jnp.array(example_indices, dtype=jnp.int32),
      jnp.array(example_values, dtype=jnp.float32))


class Lernd:  # TODO convert to jax-style nn

  def __init__(
      self,
      ilp_problem: ILP,
      program_template: ProgramTemplate,
      mini_batch: float = 1.0,
      full_loss: bool = True):
    self._ilp_problem = ilp_problem
    self._language_model = ilp_problem.language_model
    self._program_template = program_template
    self._mini_batch = mini_batch
    self._full_loss = full_loss

    # TODO use jax?
    self._rng = np.random.default_rng()

    print('Generating clauses...')
    self._clauses = f_generate(self._program_template, self._language_model)

    print('Generating ground atoms...')
    self._ground_atoms = GroundAtoms(
        self._language_model, self._program_template)

    print('Making big lambda...')
    self._big_lambda = make_lambda(
        ilp_problem.positive_examples,
        ilp_problem.negative_examples,
        self._ground_atoms)

    print('Generating initial valuation...')
    self._initial_valuation = f_convert(
        self._ilp_problem.background_axioms, self._ground_atoms)

    print('Initializing Inferrer')
    self._inferrer = Inferrer(
        self._ground_atoms,
        self._language_model,
        self._clauses,
        self._program_template)

  @property
  def ilp_problem(self) -> ILP:
    return self._ilp_problem

  @property
  def forward_chaining_steps(self) -> int:
    return self._program_template.forward_chaining_steps

  @property
  def language_model(self) -> LanguageModel:
    return self._language_model

  @property
  def program_template(self) -> ProgramTemplate:
    return self._program_template

  SetAndRule = Tuple[OrderedSet, RuleTemplate]

  @property
  def clauses(self) -> Dict[Predicate, Tuple[SetAndRule, SetAndRule]]:
    return self._clauses

  @property
  def ground_atoms(self) -> GroundAtoms:
    return self._ground_atoms

  @property
  def big_lambda(self) -> Tuple[jax.Array, jax.Array]:
    return self._big_lambda

  def __call__(self, weights: OrderedDict[Predicate, jax.Array]):
    return self._inferrer.f_infer(
        self._initial_valuation, weights)

  def loss(
      self, valuation, weights: OrderedDict[Predicate, jax.Array]
      ) -> float:
    alphas, small_lambdas = self._big_lambda
    # Extracting predictions for given (positive and negative) examples (f_extract)
    preds = valuation[alphas]
    full_loss = None  # not used for training if mini batching
    if self._mini_batch >= 1 or self._full_loss:
      full_loss = -jnp.mean(
        small_lambdas * jnp.log(preds + 1e-12) +
        (1 - small_lambdas) * jnp.log(1 - preds + 1e-12)
      )

    training_loss = full_loss
    if self._mini_batch < 1:
      num_examples = len(alphas)
      batch_size = int(self._mini_batch * num_examples)
      # Pick batch_size random samples
      indices = self._rng.choice(num_examples, batch_size, replace=False)
      small_lambdas = small_lambdas[indices]
      preds = preds[indices]
      training_loss = -jnp.mean(
          small_lambdas * jnp.log(preds + 1e-12) +
          (1 - small_lambdas) * jnp.log(1 - preds + 1e-12))

    #return training_loss, valuation, full_loss
    return full_loss

