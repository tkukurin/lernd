#!/usr/bin/env python3
import datetime as dt
import json
import os
import pickle
import typing
from collections import OrderedDict
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from matplotlib import pyplot as plt
from ordered_set import OrderedSet

from lernd.classes import ILP, Clause, ProgramTemplate
from lernd.generator import ClausePair
from lernd.lernd_loss import Lernd
from lernd.lernd_types import GroundAtom, Predicate, RuleTemplate
from lernd.util import get_ground_atom_probs, ground_atom2str


def output_to_files(
    task_id: str,
    definitions: str,
    ground_atoms: str,
    losses: typing.List[float],
    weights: typing.OrderedDict[Predicate, jnp.array],
    folder: str = ''):
  with open(os.path.join(folder, f'{task_id}_definitions.json'), 'w') as f:
    f.write(definitions)
  with open(os.path.join(folder, f'{task_id}_ground_atoms.txt'), 'w') as f:
    f.write(ground_atoms)
  with open(os.path.join(folder, f'{task_id}_losses.txt'), 'w') as f:
    [f.write(str(loss) + '\n') for loss in losses]
  with open(os.path.join(folder, f'{task_id}_weights.pickle'), 'wb') as f:
    pickle.dump(weights, f)


def generate_weight_matrices(
    clauses: Dict[Predicate, ClausePair],
    stddev: float = 0.05) -> typing.OrderedDict[Predicate,
        jax.Array]:
  rule_weights = OrderedDict()
  key = jax.random.PRNGKey(seed=42)
  for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
    rule_weights[pred] = jax.random.normal(
        key, (len(clauses_1), len(clauses_2) or 1)) * stddev
  return rule_weights


def extract_definitions(
    clauses: Dict[Predicate, ClausePair],
    weights: typing.OrderedDict[Predicate, jax.Array],
    clause_prob_threshold: float = 0.1) -> str:
  """JSON of target and aux predicates from weights."""
  output = []
  for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
    shape = weights[pred].shape
    pred_weights = jnp.reshape(weights[pred], [-1])
    pred_probs_flat = jax.nn.softmax(pred_weights)
    max_value = np.max(pred_probs_flat)
    clause_prob_threshold = min(max_value, clause_prob_threshold)
    pred_probs = jnp.reshape(pred_probs_flat[:, np.newaxis], shape)
    item = {'clause_prob_threshold': clause_prob_threshold}
    indices = np.nonzero(pred_probs >= clause_prob_threshold)
    if tau2 is not None:
      for index_tuple in zip(indices[0], indices[1]):
        item['confidence'] = pred_probs[index_tuple].numpy().astype(float)
        item['definition'] = [
            str(clauses_1[index_tuple[0]]), str(clauses_2[index_tuple[1]])
        ]
    else:
      for index in indices[0]:
        item['confidence'] = pred_probs[index][0].numpy().astype(float)
        item['definition'] = [str(clauses_1[index])]
    output.append(item)
  return json.dumps(output, indent=2)


def get_valuations(
    ground_atom_probs: typing.OrderedDict[GroundAtom, float],
    threshold: float = 0.01) -> str:
  return '\n'.join([
    f'ground atom valuations >{threshold} (readability):',
    *(f'{ground_atom2str(ground_atom)} - {p}\n'
      for ground_atom, p in ground_atom_probs.items()
      if p > threshold)
  ])


def main_loop(
    ilp_problem: ILP,
    program_template: ProgramTemplate,
    learning_rate: float = 0.5,
    steps: int = 6000,
    mini_batch: float = 1.0,
    weight_stddev: float = 0.05,
    clause_prob_threshold: float = 0.1,
    plot_loss: bool = False,
    save_output: bool = False):
  ts = dt.datetime.now().__format__('%y-%m-%d_%H-%M')
  task_id = f'{ilp_problem.name}_{ts}'
  mb = mini_batch < 1.0
  lernd_model = Lernd(ilp_problem, program_template, mini_batch=mini_batch)

  print('Generating weight matrices...')
  weights = generate_weight_matrices(lernd_model.clauses, stddev=weight_stddev)

  losses = []
  opt = optax.rmsprop(learning_rate)
  opt_state = opt.init(weights)
  for i in range(1, steps + 1):
    valuation = lernd_model(weights)
    loss, grad = jax.value_and_grad(lernd_model.loss)(
        valuation, weights)
    updates, opt_state = opt.update(grad, opt_state)
    weights = optax.apply_updates(weights, updates)

    losses.append(loss)
    if i % 10 == 0:
      print(loss)
    if i == steps:
      definitions = extract_definitions(
          lernd_model.clauses,
          weights,
          clause_prob_threshold=clause_prob_threshold)
      print('Definitions:', definitions)
      ground_atom_probs = get_ground_atom_probs(
          valuation, lernd_model.ground_atoms)
      ground_atom_probs_str = get_valuations(ground_atom_probs)
      print(ground_atom_probs_str)
      if save_output:
        output_to_files(
            task_id, definitions, ground_atom_probs_str, losses, weights)

  if (plot_loss or save_output) and len(losses) > 0:
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title('Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    if save_output:
      plt.savefig(task_id + '.png')
    if plot_loss:
      plt.show()

