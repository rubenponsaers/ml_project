from absl import app

import pyspiel
import tensorflow.compat.v1 as tf
import exploit_nash as exploit_nash
import extract_policy as extract_policy
import evaluation_opponents as evaluation_opponents

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner


def train_kuhn(env, alg, agents):
	num_actions = env.action_spec()["num_actions"]
	game = pyspiel.load_game("kuhn_poker")
	num_episodes = 1000 # number of iterations
	num_eval = 100	# how many evaluations are done against a specific agent
	step_size = 100 # after how many steps does an evaluation happen
	
	# evaluation structures
	exploitability_evolution = [[],[]]
	nashconv_evolution = [[]]
	utility_against_random_evolution = [[],[]]
	
	# Training information
	print("---------- "+alg+" ----------")
	print("Number of actions: "+str(num_actions))
	print("Game: "+str(game))
	print("Number of episodes: "+str(num_episodes))
	print("Number of evaluations: "+str(num_eval))
	print("---------- "+alg+" (START) ----------")

	# start training agents
	pol = train_agents(env, game, alg, agents, num_actions, num_episodes, step_size, num_eval,
		exploitability_evolution, nashconv_evolution, utility_against_random_evolution)

	# save evaluation structures


	# save final policy

	# draw figures

# Train agents based on a specific algorithm
def train_agents(env, game, alg, agents, num_actions, num_episodes, step_size, num_eval, 
	exploitability_evolution, nashconv_evolution, utility_evolution_against_random):
	
	# Start training loop
	for i in range(num_episodes):
		
		# Evaluate current agents
		if i % step_size == 0:
			print("Training progress: "+str(i)+"/"+str(num_episodes))
			# Get current policy
			pol = extract_policy.ExtractPolicy(env,agents)
			#print(pol.to_tabular())

			# Exploitability
			expl = exploit_nash.exploitability(game, pol)
			print(expl)

			# NashConv
			nash = exploit_nash.nash_conv(game,pol)

			# Evaluation against random
			(avg_u, avg_w) = evaluation_opponents.evaluate_against_random(env,agents[0],0,alg,num_actions,num_eval)
			print((avg_u, avg_w))
	
		# Play a game
		time_step = env.reset()
		while not time_step.last():
			player_id = time_step.observations["current_player"]
			agent_output = agents[player_id].step(time_step)
			time_step = env.step([agent_output.action])
		for agent in agents:
			agent.step(time_step)

	print("---------- "+alg+" (COMPLETE) ----------")
	return pol


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