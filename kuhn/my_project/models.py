from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr_tf2
from open_spiel.python.algorithms import exploitability

import pyspiel


import pandas as pd


import tensorflow as tf

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


def _policy_to_csv(game, policy, filename):
    df = pd.DataFrame(
            data=policy.action_probability_array,
            index=[s.history_str() for s in policy.states])
    df.to_csv(filename)