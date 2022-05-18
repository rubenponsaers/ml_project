#import yaml
import pyspiel
from open_spiel.python.algorithms import cfr, deep_cfr_tf2
#import deep_cfr_tf2, deep_cfr
#from deep_cfr_tf2 import PolicyNetwork
from open_spiel.python.bots import uniform_random
from open_spiel.python.bots import policy as policy_bot
from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import evaluate_bots, dqn, random_agent
import custom_agents

import numpy as np
#import pandas as pd
import fcpa_agent
#import tensorflow.compat.v1 as tf1
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


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

#def load_config(model):
#  with open("config.yaml") as file:
#    config = yaml.full_load(file)
#  return config[model]

def cfr_model():
  #config = load_config("cfr")
  game = get_game(config["game"])
  cfr_solver = cfr.CFRSolver(game)
  for step in range(config["num_steps"]):
    print("{}/{}".format(step, config["num_steps"]))
    cfr_solver.evaluate_and_update_policy()
  average_policy = cfr_solver.average_policy()
  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  print(average_policy_values)

'''
def deep_cfr_model():
  config = load_config("deep_cfr")
  game = get_game(config["game"])
  env = get_environment(config["game"])
  print(game)
  with tf.compat.v1.Session() as sess:
    print("Training a Deep CFR Model with following configuration: {}".format(config))
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
      sess,
      game,
      policy_network_layers = config['policy_network_layers'],
      advantage_network_layers = config['advantage_network_layers'],
      num_iterations = config['num_iterations'],
      num_traversals = config['num_traversals'],
      learning_rate = config['learning_rate'],
      epsilon_start = config['epsilon_start'],
      epsilon_end = config['epsilon_end'],
      memory_capacity=config['memory_capacity']
    )
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(config['num_steps']):
      policy_model, advantage_losses, policy_loss = deep_cfr_solver.solve()
      
      # for player, losses in advantage_losses.items():
      #   print("Advantage for player %d: %s", player,
      #            losses[:2] + ["..."] + losses[-2:])
      #   print("Advantage Buffer Size for player %s: '%s'", player,
      #               len(deep_cfr_solver.advantage_buffers[player]))
      # print("Strategy Buffer Size: '%s'",
      #             len(deep_cfr_solver.strategy_buffer))
      # print("Final policy loss: '%s'", policy_loss)
      # print(step)
      

      #if step%(config['save_every']) == 0:
        #policy_model.save_weights("tmp/policy_model_it{}.h5".format(step))
        #policy_table = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
        #_policy_to_csv(game, policy_table, config['checkpoint_dir'])
      if step%(config['eval_every']) == 0:
        utility_against_random = eval_against_random(
          game,
          deep_cfr_solver,
          0,
          100
        )
        print('Deep CFR in Kuhn FCPA 2p. Utility against random agent: {} after {} steps'.format(utility_against_random, step))
  return deep_cfr_solver
'''
def deep_cfr_model_tf2():
  #config = load_config("deep_cfr_tf2")
  print("DEEP CFR TF2")
  game = get_game("fcpa")
  env = get_environment("fcpa")
  num_actions = env.action_spec()["num_actions"]
  print("Training a Deep CFR Model with following configuration:")

  random_agents = [
    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
    for idx in range(2)
  ]

  fold_agents = [
    custom_agents.FoldAgent(player_id=idx, num_actions=num_actions) 
    for idx in range(2)
  ]

  fiftyfifty_agents = [
    custom_agents.FiftyFiftyAgent(player_id=idx, num_actions=num_actions) 
    for idx in range(2)
  ]
  '''
  deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
    game,
    policy_network_layers = config['policy_network_layers'],
    advantage_network_layers = config['advantage_network_layers'],
    num_iterations = config['num_iterations'],
    num_traversals = config['num_traversals'],
    batch_size_advantage=config['batch_size_advantage'],
    batch_size_strategy=config['batch_size_strategy'],
    memory_capacity=config['memory_capacity'],
    learning_rate = config['learning_rate'],
    policy_network_train_steps=config['policy_network_train_steps'],
    advantage_network_train_steps=config['advantage_network_train_steps'],
    reinitialize_advantage_networks=config['reinitialize_advantage_networks']
  )
  '''
  deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
    game,
    policy_network_layers = (32,16),
    advantage_network_layers = (16,8),
    num_iterations = 2,
    num_traversals = 2,
    batch_size_advantage=2048,
    batch_size_strategy=2048,
    memory_capacity=1000000,
    learning_rate = 0.001,
    policy_network_train_steps=500,
    advantage_network_train_steps=50,
    reinitialize_advantage_networks=True
  )
  for step in range(100):
    policy_model, _, _ = deep_cfr_solver.solve()
    if step%1 == 0:
      policy_model.save_weights("./checkpoints/fcpa_deep_cfr_tf2.h5")
      #deep_cfr_solver.save_policy_network("/tmp/deep_cfr_test/")
    if step%1 == 0:
      agents = [
        fcpa_agent.Agent(player_id=idx) 
        for idx in range(2)
      ]
      r_mean_random = eval_against_bots(env, [deep_cfr_solver], random_agents, 1000)
      r_mean_fold = eval_against_bots(env, agents, fold_agents, 1000)
      r_mean_fiftyfifty = eval_against_bots(env, agents, fiftyfifty_agents, 1000)
      print("[{}] Mean episode rewards against random {}".format(step + 1, r_mean_random))
      print("[{}] Mean episode rewards against always fold {}".format(step + 1, r_mean_fold))
      print("[{}] Mean episode rewards against 50/50 fold call {}".format(step + 1, r_mean_fiftyfifty))
      #print('Deep CFR in FCPA Poker 2p. Utility against random agent: {} after {} steps'.format(utility_against_random, step))
  return deep_cfr_solver
'''
def dqn_model():
  """Source: open_spiel.python.examples.breakthrough_dqn.py"""
  config = load_config("dqn")
  game = get_game(config["game"])
  env = get_environment(config["game"])
  print("Training a DQN model with following configuration: {}".format(config))
  #env = rl_environment.Environment(game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
    for idx in range(config["num_players"])
  ]

  fold_agents = [
    custom_agents.FoldAgent(player_id=idx, num_actions=num_actions) 
    for idx in range(config["num_players"])
  ]

  fiftyfifty_agents = [
    custom_agents.FiftyFiftyAgent(player_id=idx, num_actions=num_actions) 
    for idx in range(config["num_players"])
  ]

  with tf.compat.v1.Session() as sess:
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
        #learning_rate=config["learning_rate"],
        batch_size=config["batch_size"]) for idx in range(config["num_players"])
    ]
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(config["num_steps"]):
      if (step + 1) % config["eval_every"] == 0:
        r_mean_random = eval_against_bots(env, agents, random_agents, 1000)
        r_mean_fold = eval_against_bots(env, agents, fold_agents, 1000)
        r_mean_fiftyfifty = eval_against_bots(env, agents, fiftyfifty_agents, 1000)
        print("[{}] Mean episode rewards against random {}".format(step + 1, r_mean_random))
        print("[{}] Mean episode rewards against always fold {}".format(step + 1, r_mean_fold))
        print("[{}] Mean episode rewards against 50/50 fold call {}".format(step + 1, r_mean_fiftyfifty))
      if (step + 1) % config["save_every"] == 0:
        for agent in agents:
          agent.save(config["checkpoint_dir"])
        print("SAVED")

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
'''

def eval_against_bots(env, trained_agents, bots, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    # set current agents to random agents
    cur_agents = bots[:]
    # set one trained agent in list, the other remains the random agent
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


def evaluate_policy_against_random(game, policy, num_evals):
  """Source: open_spiel.python.algorithms.evaluate_bots.py"""
  u_list = []
  rng = np.random.RandomState()
  random_agent = uniform_random.UniformRandomBot(1, rng)
  fcpa_agent = fcpa_agent.get_agent_for_tournament(0)
  
  """
  pid = 0
  for i in range(num_evals):
    state = game.new_initial_state()
    if i == int(num_evals/2):
      pid = 1
      random_agent = uniform_random.UniformRandomBot(0, rng)
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        action = rng.choice(outcomes, p=probs)
        state.apply_action(action)
      else:
        if state.current_player() == pid:
          actions = policy.action_probabilities(state)
          best_action = max(actions, key=actions.get)
          state.apply_action(best_action)
        else:
          action =random_agent.step(state)
          state.apply_action(action)
    u_list.append(state.returns()[pid])

  return sum(u_list)/num_evals
  """



def eval_against_random(game, solver, pid, num_episodes):
  print("Evaluating against random agents")
  """ Evaluates agent for num_episodes against an opponent that chooses a random action. """
  random_pid = 1-pid
  rng = np.random.RandomState()
  random_agent = uniform_random.UniformRandomBot(random_pid, rng)
  if pid == 0:
      agents = [solver, random_agent]
  else:
      agents = [random_agent, solver]

  avg_u = 0
  for step in range(num_episodes):
    state = game.new_initial_state()

    while not state.is_terminal():
      # The state can be three different types: chance node,
      # simultaneous node, or decision node
      current_player = state.current_player()
      if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = rng.choice(action_list, p=prob_list)
        state.apply_action(action)			
      else:
        # Decision node: sample action for the single current player
        if current_player == pid:
          actions = solver.action_probabilities(state)
          best_action = max(actions, key=actions.get)
          state.apply_action(best_action)
        else:
          action = agents[current_player].step(state)
          state.apply_action(action)



    # Game is now done. Print utilities for each player
    returns = state.returns()
    avg_u += returns[0]


  # We wish to return the average, so divide the sum by num_episodes
  avg_u = avg_u/num_episodes
  return avg_u

def _policy_to_csv(game, policy, filename):
  df = pd.DataFrame(
      data=policy.action_probability_array,
      index=[s.history_str() for s in policy.states])
  df.to_csv(filename)
  

if __name__ == "__main__":
  deep_cfr_model_tf2()
  #dqn_model()
