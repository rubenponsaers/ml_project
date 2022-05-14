import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.egt import dynamics, visualization, utils

import matplotlib.pyplot as plt
#from matplotlib import Figure
#from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet

def train(game):
    #initialization of the environment 
    env = rl_environment.Environment(game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    #initialization of the agents
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

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

def plot_dynamics(game, singlePopulation = False):
    fig = plt.figure()
    payoff_tensor = utils.game_payoffs_array(game)

    if singlePopulation:
        dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)
        ax1 = fig.add_subplot(121, projection='3x3')
        ax1.set(labels=["Rock", "Paper", "Scissors"])
        ax1.quiver(dyn)
        ax2 = fig.add_subplot(122, projection='3x3')
        ax2.set(labels=["Rock", "Paper", "Scissors"])
        ax2.streamplot(dyn)
        plt.savefig('biased_rps.png')
    else:
        dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
        ax1 = fig.add_subplot(121, projection='2x2')
        ax1.quiver(dyn)
        ax2 = fig.add_subplot(122, projection='2x2')
        ax2.streamplot(dyn)
    
    plt.show()


