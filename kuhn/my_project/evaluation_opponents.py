# Play automatic game for an agent trained on a a specific algorithm
from open_spiel.python.algorithms import random_agent


def evaluate_against_random(env,agent,agent_id,alg,num_actions,num_eval):
	print("---------- EVALUATING AGAINST RANDOM (AGENT "+str(agent_id)+" : "+alg+") ----------") 
	random_agent_id = (agent_id + 1) % 2
	eval_agent = random_agent.RandomAgent(random_agent_id, num_actions, "Entropy Master 2000")
	if agent_id==0:
		eval_agents = [agent, eval_agent]
	else :
		eval_agents = [eval_agent, agent]
	# Calculate average utility and wins
	avg_u = 0
	avg_w = 0
	for i in range(num_eval):
		time_step = env.reset()
		while not time_step.last():	
			player_id = time_step.observations["current_player"]
			agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
			time_step = env.step([agent_output.action])
		avg_u += time_step.rewards[agent_id]
		if time_step.rewards[agent_id] > time_step.rewards[random_agent_id]: avg_w +=1
	
	avg_u = avg_u/num_eval
	return (avg_u, avg_w)