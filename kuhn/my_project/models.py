from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml

from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import deep_cfr_tf2, exploitability, dqn, random_agent
import tensorflow.compat.v1 as tf

import pyspiel


import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def load_config(model):
    with open('config.yaml') as file:
        config = yaml.full_load(file)
    return config[model]

'''
def deep_cfr_model_tf1():
    import tensorflow.compat.v1 as tf1
    config = load_config('deep_cfr_tf1')
    game = pyspiel.load_game(config['game'])
    with tf1.Session() as sess:
        print("Training a Deep CFR Model with following configuration: {}".format(config))
        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers = config['policy_network_layers'],
            advantage_network_layers = config['advantage_network_layers'],
            num_iterations = config['num_iterations'],
            num_traversals = config['num_traversals'],
            learning_rate = config['learning_rate'],
            batch_size_advantage=config['batch_size_advantage'],
            batch_size_strategy=config['batch_size_strategy']
        )
        sess.run(tf1.global_variables_initializer())
        for step in range(config['num_steps']):
            deep_cfr_solver.solve()
            policy_model = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
            conv = exploitability.nash_conv(
                game,
                policy_model
                )
            if step%(config['save_every']) == 0:
                _policy_to_csv(game, policy_model, config['checkpoint_dir'])
            if step%(config['eval_every']) == 0:
                print('Deep CFR in Kuhn Poker 2p. NashConv: {} after {} steps'.format(conv, step))
    return deep_cfr_solver
'''

def deep_cfr_model_tf2():
    config = load_config('deep_cfr_tf2')
    game = pyspiel.load_game(config['game'])
    print(game)
    print("Training a Deep CFR Model with following configuration: {}".format(config))
    deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
        game,
        policy_network_layers = config['policy_network_layers'],
        advantage_network_layers = config['advantage_network_layers'],
        num_iterations = config['num_iterations'],
        num_traversals = config['num_traversals'],
        learning_rate = config['learning_rate'],
        batch_size_advantage=config['batch_size_advantage'],
        batch_size_strategy=config['batch_size_strategy']
    )
    for step in range(config['num_steps']):
        deep_cfr_solver.solve()
        policy_model = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(
                game, deep_cfr_solver.action_probabilities))
        if step%(config['save_every']) == 0:
            _policy_to_csv(game, policy_model, config['checkpoint_dir'])
        if step%(config['eval_every']) == 0:
            print('Deep CFR in Kuhn Poker 2p. NashConv: {} after {} steps'.format(conv, step))
    return deep_cfr_solver

def dqn_model():
    config = load_config('dqn')
    game = pyspiel.load_game(config['game'])
    print(game)
    print("Training a DQN model with following configuration: {}".format(config))
    env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(config["num_players"])
    ]

    with tf.Session() as sess:
        hidden_layers_sizes = [int(l) for l in config["hidden_layers_sizes"]]
        # pylint: disable=g-complex-comprehension
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                replay_buffer_capacity=config["replay_buffer_capacity"],
                learning_rate=config["learning_rate"],
                epsilon_start=config["epsilon_start"],
                epsilon_end=config["epsilon_end"],
                batch_size=config["batch_size"]) for idx in range(config["num_players"])
        ]
        sess.run(tf.global_variables_initializer())
        for step in range(config["num_steps"]):
            if (step + 1) % config["eval_every"] == 0:
                r_mean = _eval_against_random_bots(env, agents, random_agents, 100)
                print("[{}] Mean episode rewards {}".format(step + 1, r_mean))
            if (step + 1) % config["save_every"] == 0:
                for agent in agents:
                    agent.save(config["checkpoint_dir"])

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
            if env.is_turn_based:
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
            else:
                agents_output = [agent.step(time_step) for agent in agents]
                action_list = [agent_output.action for agent_output in agents_output]
            time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)


def _policy_to_csv(game, policy, filename):
    df = pd.DataFrame(
            data=policy.action_probability_array,
            index=[s.history_str() for s in policy.states])
    df.to_csv(filename)

def _eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes