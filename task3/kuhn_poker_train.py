from absl import app

import pyspiel
import tensorflow.compat.v1 as tf
import my_project.eval_tools as eval_tools

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent


def train_kuhn(env, alg, agents):
	num_actions = env.action_spec()["num_actions"]
	game = pyspiel.load_game("kuhn_poker")
	num_episodes = 15000 
	num_eval = 100	# how many evaluations are done against a specific agent
	step_size = 1000 # after how many steps does an evaluation happen
	
	# evaluation structures
	exploitability_evolution = [[] for _ in range(len(agents))]
	nashconv_evolution = [[]]
	utility_evolution_against_random = [[] for _ in range(len(agents))]
	
	# start training agents
	pol = train_agents(env, alg, agents, num_actions, num_episodes, step_size, num_eval,
		exploitability_evolution, nashconv_evolution, utility_evolution_against_random)
	
	# save evaluation structures

	# save final policy

	# draw figures

# Train agents based on a specific algorithm
def train_agents(env, alg, agents, num_actions, num_episodes, step_size, num_eval, 
	exploitability_evolution, nashconv_evolution, utility_evolution_against_random):
	
	# Start training loop
	for i in range(num_episodes):
		if i % step_size == 0:
			print("Training progress: "+str(i)+"/"+str(num_episodes))
		time_step = env.reset()
		while not time_step.last():
			player_id = time_step.observations["current_player"]
			agent_output = agents[player_id].step(time_step)
			time_step = env.step([agent_output.action])
		for agent in agents:
			agent.step(time_step)
		
		# Get current policy with available 


	print("Training agents of " + alg + " complete.")
	return


# Get agents for Q-learning
def get_qlearning_agents(env,sess):
	num_actions = env.action_spec()["num_actions"]
	agent1 = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions)
	agent2 = tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)
	return [agent1, agent2]

def main(_):
	env = rl_environment.Environment("kuhn_poker")
	sess = tf.Session()

	train_kuhn(env, "Q-learning", get_qlearning_agents(env,sess))
	return

if __name__ == "__main__":
    app.run(main)