from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
import numpy as np

class AlwaysFoldBot(pyspiel.Bot):
  """Chooses uniformly at random from the available legal actions."""

  def __init__(self, player_id, rng):
    """Initializes a uniform-random bot.
    Args:
      player_id: The integer id of the player for this bot, e.g. `0` if acting
        as the first player.
      rng: A random number generator supporting a `choice` method, e.g.
        `np.random`
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng

  def restart_at(self, state):
    pass

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state.
    Args:
      state: The current state of the game.
    Returns:
      A `(policy, action)` pair, where policy is a `list` of
      `(action, probability)` pairs for each legal action, with
      `probability = 1/num_actions`
      The `action` is selected uniformly at random from the legal actions,
      or `pyspiel.INVALID_ACTION` if there are no legal actions available.
    """
    legal_actions = state.legal_actions(self._player_id)
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    if 0 in legal_actions:
      p = np.zeros(len(legal_actions))
      p[0] = 1.0
      return p, legal_actions[0]
    p = 1 / len(legal_actions)
    policy = [(action, p) for action in legal_actions]
    action = self._rng.choice(legal_actions)
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]

class FiftyFiftyBot(pyspiel.Bot):
  """Chooses uniformly at random from the available legal actions."""

  def __init__(self, player_id, rng):
    """Initializes a uniform-random bot.
    Args:
      player_id: The integer id of the player for this bot, e.g. `0` if acting
        as the first player.
      rng: A random number generator supporting a `choice` method, e.g.
        `np.random`
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng

  def restart_at(self, state):
    pass

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state.
    Args:
      state: The current state of the game.
    Returns:
      A `(policy, action)` pair, where policy is a `list` of
      `(action, probability)` pairs for each legal action, with
      `probability = 1/num_actions`
      The `action` is selected uniformly at random from the legal actions,
      or `pyspiel.INVALID_ACTION` if there are no legal actions available.
    """
    legal_actions = state.legal_actions(self._player_id)
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    if (0 in legal_actions) and (1 in legal_actions):
      p = np.zeros(len(legal_actions))
      p[0] = 0.5
      p[1] = 0.5
      action = self._rng.choice([0, 1])
      return p, action
    if (0 in legal_actions):
      p = np.zeros(len(legal_actions))
      p[0] = 1.0
      return p, legal_actions[0]
    if (1 in legal_actions):
      p = np.zeros(len(legal_actions))
      p[1] = 1.0
      return p, legal_actions[1]
    p = 1 / len(legal_actions)
    policy = [(action, p) for action in legal_actions]
    action = self._rng.choice(legal_actions)
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]



class ProbsBot(pyspiel.Bot):
  """Chooses uniformly at random from the available legal actions."""

  def __init__(self, player_id, rng):
    """Initializes a uniform-random bot.
    Args:
      player_id: The integer id of the player for this bot, e.g. `0` if acting
        as the first player.
      rng: A random number generator supporting a `choice` method, e.g.
        `np.random`
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng
    self._probs = np.array([.10, .50, .35, .5])

  def restart_at(self, state):
    pass

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state.
    Args:
      state: The current state of the game.
    Returns:
      A `(policy, action)` pair, where policy is a `list` of
      `(action, probability)` pairs for each legal action, with
      `probability = 1/num_actions`
      The `action` is selected uniformly at random from the legal actions,
      or `pyspiel.INVALID_ACTION` if there are no legal actions available.
    """
    legal_actions = state.legal_actions(self._player_id)
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    probs = self._probs[np.array(legal_actions)]
    probs /= sum(probs)
    policy = [(action, p) for action, p in zip(legal_actions, probs)]
    action = self._rng.choice(legal_actions)
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]