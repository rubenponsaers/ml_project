# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.algorithms.eva."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import eva

import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

def get_game(name):
  if name == "fcpa":
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)

  return game

def get_environment(name):
  if name == "fcpa":
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    env = rl_environment.Environment(fcpa_game_string)
  return env

def train_eva(game):
  env = rl_environment.Environment(game)
  num_players = env.num_players
  eva_agents = []
  num_actions = env.action_spec()["num_actions"]
  state_size = env.observation_spec()["info_state"][0]
  with tf.Session() as sess:
    for player in range(num_players):
      eva_agents.append(
          eva.EVAAgent(
              sess,
              env,
              player,
              state_size,
              num_actions,
              embedding_network_layers=(64, 32),
              embedding_size=12,
              learning_rate=1e-4,
              mixing_parameter=0.5,
              memory_capacity=int(1e6),
              discount_factor=1.0,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_duration=int(1e6)))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print(type(eva_agents[0]))
    print(dir(eva_agents[0]))
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      current_agent = eva_agents[current_player]
      # 1.  Step the agent.
      # 2.  Step the Environment.
      agent_output = current_agent.step(time_step)
      time_step = env.step([agent_output.action])
    for agent in eva_agents:
      agent.step(time_step)
    saver.save(sess, "eva_model")
    print("SAVED")
    validate(game, env)

def validate(game, env):
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('eva_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    print("RESTORED")
    print(dir(new_saver))
    sess.run(tf.global_variables_initializer())
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      current_agent = eva_agents[current_player]
      # 1.  Step the agent.
      # 2.  Step the Environment.
      agent_output = current_agent.step(time_step)
      time_step = env.step([agent_output.action])
    for agent in eva_agents:
      agent.step(time_step)





if __name__ == "__main__":
  game = get_game("fcpa")
  train_eva(game)