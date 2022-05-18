import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner, action_value, random_agent
from open_spiel.python.egt import dynamics, visualization, utils

import matplotlib.pyplot as plt
#from matplotlib import Figure
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet

import dynamics_self

import numpy as np

import seaborn as sns
sns.set() # Setting seaborn as default style even if use only matplotlib


import pandas as pd

import dynamics_self

import numpy as np

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

def train(game, reset_every, num_steps, eval_every, labels):
    #initialization of the environment 
    env = rl_environment.Environment(game)
    game_inv = get_game("dg_inverted")
    env_inv = rl_environment.Environment(game_inv)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    #initialization of the agents
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    #agents = [
    #    tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
    #    random_agent.RandomAgent(player_id=1, num_actions=num_actions)
    #]

    #random_agents = [
    #    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
    #    for idx in range(num_players)
    #]

    agent1_probs = []
    agent2_probs = []

    agent1_probs_avg = []
    agent2_probs_avg = []

    agent1_s1_probs = []

    steps = []
    resets = []

    s1_chosen = 0
    s2_chosen = 0

    #training of the agents via Q-learning algorithm (and self-play)
    for step in range(num_steps):
        if True:
        #if step < int(num_steps/2):
            time_step = env.reset()
            while not time_step.last():
                agent1_output = agents[0].step(time_step)
                agent2_output = agents[1].step(time_step)
                time_step = env.step([agent1_output.action, agent2_output.action])
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
            if (step+1)%reset_every == 0:
                if sum(agent1_s1_probs)/len(agent1_s1_probs) > .5:
                    s1_chosen += 1
                else:
                    s2_chosen += 1
                resets.append(step+1)
                agent1_s1_probs.clear()
                agents = [
                    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
                    for idx in range(num_players)
                ]
            if (step+1)%eval_every == 0:
                agent1_probs_avg.append(sum(agent1_probs)/len(agent1_probs))
                agent1_probs.clear()
                agent2_probs_avg.append(sum(agent2_probs)/len(agent2_probs))
                agent2_probs.clear()
                steps.append(step+1)
            
        else:
            time_step = env_inv.reset()
            while not time_step.last():
                agent1_output = agents[0].step(time_step)
                agent2_output = agents[1].step(time_step)
                time_step = env_inv.step([agent1_output.action, agent2_output.action])
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
            if (step+1)%eval_every == 0:
                agent1_probs_avg.append(sum(agent1_probs)/len(agent1_probs))
                agent1_probs.clear()
                agent2_probs_avg.append(sum(agent2_probs)/len(agent2_probs))
                agent2_probs.clear()
                steps.append(step+1)

        agent1_probs.append(agent1_output.probs)
        agent1_s1_probs.append(agent1_output.probs[0])
        agent2_probs.append(agent2_output.probs)

    #print(sum(agent1_probs_avg)/len(agent1_probs_avg))

    df_probs1 = pd.DataFrame(agent1_probs_avg, columns=labels)
    df_probs2 = pd.DataFrame(agent2_probs_avg, columns=labels)
    df_probs1['step'] = steps
    df_probs2['step'] = steps
    df_probs1.set_index('step', inplace=True)
    df_probs2.set_index('step', inplace=True)
    #print(df_probs1)

    print("Done!")
    print(s1_chosen)
    print(s2_chosen)
    fig, axes = plt.subplots(2, 1)


    axes[0].set_title("Player 1")
    axes[1].set_title("Player 2")
    #sns.lineplot(steps, agent1_probs_avg)
    plt1 = sns.lineplot(data=df_probs1, ax=axes[0])
    axes[0].legend(bbox_to_anchor=(1., 1.05))
    plt2 = sns.lineplot(data=df_probs2, ax=axes[1], legend = False)
    axes[0].set_xticks([])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Probability")
    axes[0].set_yticks([0.25, 0.5, 0.75, 1.])
    axes[1].set_ylabel("Probability")
    axes[1].set_yticks([0.25, 0.5, 0.75, 1.])
    for r in resets:
        axes[0].axvline(r, color='gray', label='reset')
        axes[1].axvline(r, color='gray', label='reset')
    #plt.plot(steps, agent2_probs_avg)
    #plt.legend(["P1-Rock", "P1-Paper", "P1-Scissors", "P2-Rock", "P2-Paper", "P2-Scissors"])
    #plt.show()


    return agents[0], agents[1]

def play_game(game, agent1, agent2):
    for _ in range(10):
        env = rl_environment.Environment(game)
        time_step = env.reset()
        agent1_output = agent1.step(time_step, is_evaluation=True)
        agent2_output = agent2.step(time_step, is_evaluation=True)
        time_step = env.step([agent1_output.action, agent2_output.action])

        print(time_step.rewards)
    

def plot_dynamics(game, actions, alpha = 0.01, temperature = 1.5, singlePopulation = False):
    fig = plt.figure(figsize=[25, 10])
    payoff_tensor = utils.game_payoffs_array(game)

    if singlePopulation:
        labels = ["Probability of choosing " + action for action in actions]
        boltzmann_dyn = dynamics_self.SinglePopulationDynamics(payoff_tensor, dynamics_self.boltzmannq_self)
        ax1 = fig.add_subplot(121, projection='3x3')
        ax1.quiver(boltzmann_dyn)
        ax1.streamplot(boltzmann_dyn)
        ax1.set_labels(labels)
        replicator_dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
        ax2 = fig.add_subplot(122, projection='3x3')
        ax2.quiver(replicator_dyn)
        ax2.streamplot(replicator_dyn)
        ax2.set_labels(labels)
    else:
        boltzmann_dyn = dynamics_self.MultiPopulationDynamics(payoff_tensor, dynamics_self.boltzmannq_self)
        ax1 = fig.add_subplot(121, projection='2x2')
        ax1.quiver(boltzmann_dyn)
        ax1.streamplot(boltzmann_dyn)
        ax1.set( 
            xlabel="Player1, probability of choosing " + actions[0],
            ylabel="Player2, probability of choosing " + actions[0],
            
            )

        replicator_dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
        ax2 = fig.add_subplot(122, projection='2x2')
        ax2.quiver(replicator_dyn)
        ax2.streamplot(replicator_dyn)
        ax2.set( 
            xlabel="Player1, probability of choosing " + actions[0],
            ylabel="Player2, probability of choosing " + actions[0]
            )
    #plt.savefig(str(game)[:-2]+'.jpg')
    plt.show()


def get_game(name):
    if name == "biased_rps":
        row_player_utils = [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]]
        col_player_utils = [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]]
        short_name = 'biased_rps'
        long_name = 'Biased Rock-Paper-Scissors'
        row_names, col_names = ['P1_Rock', 'P1_Paper', 'P1_Scissors'], ['P2_Rock', 'P2_Paper', 'P2_Scissors']

        return pyspiel.create_matrix_game(short_name, long_name, row_names, col_names, row_player_utils, col_player_utils)
    if name == "dg":
        row_player_utils = [[-1, 1], [1, -1]]
        col_player_utils = [[-1, 1], [1, -1]]
        short_name = 'dg'
        long_name = 'Dispersion Game'
        row_names, col_names = ['P1_A', 'P1_B'], ['P2_A', 'P2_B']

        return pyspiel.create_matrix_game(short_name, long_name, row_names, col_names, row_player_utils, col_player_utils)
    if name == "dg_inverted":
        row_player_utils = [[1, -1], [-1, 1]]
        col_player_utils = [[1, -1], [-1, 1]]
        short_name = 'dg'
        long_name = 'Dispersion Game'
        row_names, col_names = ['P1_A', 'P1_B'], ['P2_A', 'P2_B']

        return pyspiel.create_matrix_game(short_name, long_name, row_names, col_names, row_player_utils, col_player_utils)
    if name == "bots":
        row_player_utils = [[3, 0], [0, 2]]
        col_player_utils = [[2, 0], [0, 3]]
        short_name = 'bots'
        long_name = 'Battle of the Sexes'
        row_names, col_names = ['P1_O', 'P1_M'], ['P2_O', 'P2_M']

        return pyspiel.create_matrix_game(short_name, long_name, row_names, col_names, row_player_utils, col_player_utils)
    if name == "sg":
        row_player_utils = [[10, 0], [11, 12]]
        col_player_utils = [[10, 11], [0, 12]]
        short_name = 'sg'
        long_name = 'Subsidy Game'
        row_names, col_names = ['P1_S1', 'P1_S2'], ['P2_S1', 'P2_S2']

        return pyspiel.create_matrix_game(short_name, long_name, row_names, col_names, row_player_utils, col_player_utils)


