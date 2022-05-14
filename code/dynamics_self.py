# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Continuous-time population dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def replicator(state, fitness):
  """Continuous-time replicator dynamics.

  This is the standard form of the continuous-time replicator dynamics also
  known as selection dynamics.

  For more details, see equation (5) page 9 in
  https://jair.org/index.php/jair/article/view/10952

  Args:
    state: Probability distribution as an `np.array(shape=num_strategies)`.
    fitness: Fitness vector as an `np.array(shape=num_strategies)`.

  Returns:
    Time derivative of the population state.
  """
  avg_fitness = state.dot(fitness)
  return state * (fitness - avg_fitness)


def boltzmannq_self(state, fitness, other_state, temperature=0.01, alpha=0.1, kappa=5):
  """Selection-mutation dynamics modeling Q-learning with Boltzmann exploration.

  For more details, see equation (10) page 15 in
  https://jair.org/index.php/jair/article/view/10952

  Args:
    state: Probability distribution as an `np.array(shape=num_strategies)`.
    fitness: Fitness vector as an `np.array(shape=num_strategies)`.
    temperature: A scalar parameter determining the rate of exploration.

  Returns:
    Time derivative of the population state.
  """
  u = _boltzmann_utility_vector(other_state, fitness, kappa)
  exploitation = (1. / temperature) * replicator(state, u)
  #exploitation = np.divide(state, temperature)*(u - state.dot(u))
  exploration = (np.log(state) - state.dot(np.log(state).transpose()))
  res = alpha * (exploitation-state*exploration)
  return res

def _boltzmann_utility_vector(y, a, kappa):
  u = []
  for i, _ in enumerate(y):
    u_i = 0
    for j, y_j in enumerate(y):
      k_lt, k_eq = _distribute_in_buckets(a[i], j)
      y_k_lt = y[k_lt]
      y_k_eq = y[k_eq]
      term1 = (np.sum(y_k_lt)+np.sum(y_k_eq))**kappa
      term2 = np.sum(y_k_lt)**kappa
      denom = np.sum(y_k_eq)
      u_i += a[i][j]*y_j*(term1-term2)/denom
    u.append(u_i)
  return u

def _distribute_in_buckets(ai, j):
  lt = [aik < ai[j] for aik in ai]
  eq = [aik == ai[j] for aik in ai]
  return lt, eq



def qpg(state, fitness):
  """Q-based policy gradient dynamics (QPG).

  For more details, see equation (12) on page 18 in
  https://arxiv.org/pdf/1810.09026.pdf

  Args:
    state: Probability distribution as an `np.array(shape=num_strategies)`.
    fitness: Fitness vector as an `np.array(shape=num_strategies)`.

  Returns:
    Time derivative of the population state.
  """
  regret = fitness - state.dot(fitness)
  return state * (state * regret - np.sum(state**2 * regret))


class SinglePopulationDynamics(object):
  """Continuous-time single population dynamics.

  Attributes:
    payoff_matrix: The payoff matrix as an `numpy.ndarray` of shape `[2, k_1,
      k_2]`, where `k_1` is the number of strategies of the first player and
      `k_2` for the second player. The game is assumed to be symmetric.
    dynamics: A callback function that returns the time-derivative of the
      population state.
  """

  def __init__(self, payoff_matrix, dynamics):
    """Initializes the single-population dynamics."""
    assert payoff_matrix.ndim == 3
    assert payoff_matrix.shape[0] == 2
    assert np.allclose(payoff_matrix[0], payoff_matrix[1].T)
    self.payoff_matrix = payoff_matrix[0]
    self.dynamics = dynamics

  def __call__(self, state=None, time=None):
    """Time derivative of the population state.

    Args:
      state: Probability distribution as list or
        `numpy.ndarray(shape=num_strategies)`.
      time: Time is ignored (time-invariant dynamics). Including the argument in
        the function signature supports numerical integration via e.g.
        `scipy.integrate.odeint` which requires that the callback function has
        at least two arguments (state and time).

    Returns:
      Time derivative of the population state as
      `numpy.ndarray(shape=num_strategies)`.
    """
    state = np.array(state)
    assert state.ndim == 1
    assert state.shape[0] == self.payoff_matrix.shape[0]
    # (Ax')' = xA'
    fitness = np.matmul(state, self.payoff_matrix.T)
    return self.dynamics(state, fitness, self.payoff_matrix)


class MultiPopulationDynamics(object):
  """Continuous-time multi-population dynamics.

  Attributes:
    payoff_tensor: The payoff tensor as an numpy.ndarray of size `[n, k0, k1,
      k2, ...]`, where n is the number of players and `k0` is the number of
      strategies of the first player, `k1` of the second player and so forth.
    dynamics: List of callback functions for the time-derivative of the
      population states, where `dynamics[i]` computes the time-derivative of the
      i-th player's population state. If at construction, only a single callback
      function is provided, the same function is used for all populations.
  """

  def __init__(self, payoff_tensor, dynamics):
    """Initializes the multi-population dynamics."""
    if isinstance(dynamics, list) or isinstance(dynamics, tuple):
      assert payoff_tensor.shape[0] == len(dynamics)
    else:
      dynamics = [dynamics] * payoff_tensor.shape[0]
    self.payoff_tensor = payoff_tensor
    self.dynamics = dynamics

  def __call__(self, state, time=None):
    """Time derivative of the population states.

    Args:
      state: Combined population state for all populations as a list or flat
        `numpy.ndarray` (ndim=1). Probability distributions are concatenated in
        order of the players.
      time: Time is ignored (time-invariant dynamics). Including the argument in
        the function signature supports numerical integration via e.g.
        `scipy.integrate.odeint` which requires that the callback function has
        at least two arguments (state and time).

    Returns:
      Time derivative of the combined population state as `numpy.ndarray`.
    """
    state = np.array(state)
    n = self.payoff_tensor.shape[0]  # number of players
    ks = self.payoff_tensor.shape[1:]  # number of strategies for each player
    assert state.shape[0] == sum(ks)

    states = np.split(state, np.cumsum(ks)[:-1])
    dstates = [None] * n
    for i in range(n):
      # move i-th population to front
      fitness = np.moveaxis(self.payoff_tensor[i], i, 0)
      # marginalize out all other populations
      #for i_ in set(range(n)) - {i}:
        #fitness = np.tensordot(states[i_], fitness, axes=[0, 1])
      dstates[i] = self.dynamics[i](states[i], fitness, states[1-i])

    return np.concatenate(dstates)


def time_average(traj):
  """Time-averaged population state trajectory.

  Args:
    traj: Trajectory as `numpy.ndarray`. Time is along the first dimension,
      types/strategies along the second.

  Returns:
    Time-averaged trajectory.
  """
  n = traj.shape[0]
  sum_traj = np.cumsum(traj, axis=0)
  norm = 1. / np.arange(1, n + 1)
  return sum_traj * norm[:, np.newaxis]
