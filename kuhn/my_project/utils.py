import pandas as pd
from open_spiel.python.policy import TabularPolicy

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
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

def policy_to_csv(game, policy, filename):
    #tabular_policy = tabular_policy_from_policy(game, policy)
    df = pd.DataFrame(
            data=policy.action_probability_array,
            index=[s.history_str() for s in policy.states])
    df.to_csv(filename)
  
def tabular_policy_from_csv(game, filename):
    csv = pd.read_csv(filename, index_col=0)

    empty_tabular_policy = TabularPolicy(game)
    for state_index, state in enumerate(empty_tabular_policy.states):
        action_probabilities = {
                action: probability
                for action, probability in enumerate(csv.loc[state.history_str()])
                if probability > 0
            }
        infostate_policy = [
            action_probabilities.get(action, 0.)
            for action in range(game.num_distinct_actions())
        ]
        empty_tabular_policy.action_probability_array[
            state_index, :] = infostate_policy
    return empty_tabular_policy