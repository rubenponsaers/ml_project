from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment

from typing import Dict

# Based on the rl_agent_policy class provided by OpenSpiel
class ExtractPolicy(policy.Policy):
    def __init__(self, game, agents):
    
        player_ids = [0,1]
        super().__init__(game, player_ids)
        self._agents = agents
        self._obs = {"info_state": [None,None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        if state.is_simultaneous_node():
            assert player_id is not None, "Player ID should be specified."
        else:
            if player_id is None:
                player_id = state.current_player()
            else:
                assert player_id == state.current_player()

        # Make sure that player_id is an integer and not an enum as it is used to
        # index lists.
        player_id = int(player_id)

        legal_actions = state.legal_actions(player_id)

        self._obs["current_player"] = player_id
        self._obs["info_state"][player_id] = state.information_state_tensor(player_id)
        self._obs["legal_actions"][player_id] = legal_actions

        info_state = rl_environment.TimeStep(observations=self._obs, rewards=None, discounts=None, step_type=None)

        p = self._agents[player_id].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict
