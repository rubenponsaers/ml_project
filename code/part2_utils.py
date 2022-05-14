import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner, action_value
from open_spiel.python.egt import dynamics, visualization, utils

import matplotlib.pyplot as plt
#from matplotlib import Figure
#from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet

import dynamics_self

import numpy as np

def train(game):
    #initialization of the environment 
    env = rl_environment.Environment(game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    #initialization of the agents
    agent1 = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions)
    agent2 = action_value.TreeWalkCalculator(game)
    agents = [agent1, agent2]
    #agents = [
    #    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    #    for idx in range(num_players)
    #]

    agent1_losses_avg = []
    agent2_losses_avg = []

    agent1_losses = []
    agent2_losses = []

    #training of the agents via Q-learning algorithm (and self-play)
    for cur_episode in range(25000):
        if cur_episode % 1000 == 0:
            print(f"Episodes: {cur_episode}")
            if cur_episode != 0:
                agent1_losses_avg.append(sum(agent1_losses)/len(agent1_losses))
                agent1_losses.clear()

                agent2_losses_avg.append(sum(agent2_losses)/len(agent2_losses))
                agent2_losses.clear()
        time_step = env.reset()
        while not time_step.last():
            agent1_output = agents[0].step(time_step)
            agent2_output = agents[1].step(time_step)
            time_step = env.step([agent1_output.action, agent2_output.action])
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)
        
        agent1_losses.append(agents[0].loss)
        agent2_losses.append(agents[1].loss)

    print("Done!")

    return agents[0], agents[1]

def play_game(game, agent1, agent2):
    for _ in range(10):
        env = rl_environment.Environment(game)
        time_step = env.reset()
        agent1_output = agent1.step(time_step, is_evaluation=True)
        agent2_output = agent2.step(time_step, is_evaluation=True)
        time_step = env.step([agent1_output.action, agent2_output.action])

        print(time_step.rewards)

def plot_dynamics_regular(game, actions, singlePopulation = False):
    plot_dynamics(game, actions, dynamics.replicator, singlePopulation)

def own_leniet_boltzmann_dyn(state, fitness, temperature=1.5, alpha=0.01):
    #exploitation = np.divide(1., temperature) * dynamics.replicator(state, fitness)

    state = np.array(state)
    n = 2 # number of players
    ks = (len(state), len(state))  # number of strategies for each player
    states = np.split(state, np.cumsum(ks)[:-1]) # Split state into states
    print(states)
    u = boltzmann_utility_vector(state, fitness)
    exploitation = np.divide(state, temperature)*(u - state.dot(u))
    exploration = state*(np.log(state)-np.sum(state*np.log(state)))
    res = alpha * (exploitation-exploration)
    return res
 

def boltzmann_utility_vector(state, fitness, kappa=5):
    print(fitness)
    print(state)
    u = []
    for i in enumerate(state):
        u_i = 0
        for j, y_j in enumerate(state):
            a_ij = fitness
    return np.array()
    

def plot_dynamics_boltzmann(game, actions, alpha = 0.01, temperature = 1.5, singlePopulation = False):
    plot_dynamics(game, actions, dynamics_self.boltzmannq_self, singlePopulation)


def plot_dynamics(game, actions, dyn, singlePopulation = False):
    fig = plt.figure(figsize=[25, 10])
    payoff_tensor = utils.game_payoffs_array(game)

    if singlePopulation:
        labels = ["Probability of choosing " + action for action in actions]
        dyn = dynamics_self.SinglePopulationDynamics(payoff_tensor, dyn)
        ax1 = fig.add_subplot(121, projection='3x3')
        ax1.quiver(dyn)
        ax1.set_labels(labels)
        ax2 = fig.add_subplot(122, projection='3x3')
        ax2.streamplot(dyn, dt=0.2)
        ax2.set_labels(labels)
    else:
        dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dyn)
        ax1 = fig.add_subplot(121, projection='2x2')
        ax1.quiver(dyn)
        ax1.set( 
            xlabel="Player1, probability of choosing " + actions[0],
            ylabel="Player2, probability of choosing " + actions[0]
            )
        ax2 = fig.add_subplot(122, projection='2x2')
        ax2.streamplot(dyn)
        ax2.set( 
            xlabel="Player1, probability of choosing " + actions[0],
            ylabel="Player2, probability of choosing " + actions[0]
            )
    plt.savefig(str(game)[:-2]+'.jpg')
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


