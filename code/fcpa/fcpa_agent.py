#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py
Extend this class to provide an agent that can participate in a tournament.
Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots, deep_cfr_tf2
from open_spiel.python.bots import human
import tensorflow as tf



logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.
    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    #my_player = Agent(human.HumanBot(), None)
    
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.
        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        
        fcpa_game_string = (
                "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
                "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
                "stack=20000 20000,bettingAbstraction=fcpa)")
        game = pyspiel.load_game(fcpa_game_string)
        input_size = len(game.new_initial_state().information_state_tensor(0))
        policy_network_layers = (32, 16)
        num_actions = game.num_distinct_actions()
        model = deep_cfr_tf2.PolicyNetwork(input_size, policy_network_layers, num_actions)
        model((tf.random.uniform(shape=(1, input_size)), tf.random.uniform(shape=(num_actions,))), training=False)
        model.load_weights("./checkpoints/fcpa_deep_cfr_tf299.h5")
        print(type(model))
        self.state = game.new_initial_state()
        
        
        
        #model = tf.keras.models.load_model("/tmp/deep_cfr_test/", compile=False)
        #print(type(model))

        
        self.player_id = player_id
        #self.policy = deep_cfr_tf2.PolicyNetwork(model)
        self.policy = model
        print("loaded model")

    def restart_at(self, state):
        """Starting a new game in the given state.
        :param state: The initial state of the game.
        """
        print("restart")
        self.state = state
        

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.
        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        print("inform")
        self.state = state
        
    
    def step_with_policy():
        pass
        '''
        policy = self._policy.action_probabilities(state, self._player_id)
        action_list = list(policy.keys())
        if not any(action_list):
            return [], pyspiel.INVALID_ACTION

        action = self._rng.choice(action_list, p=list(policy.values()))
        return list(policy.items()), action
        '''


    def step(self, state, is_evaluation=True):
        """Returns the selected action in the given state.
        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        action_probs = self._action_probabilities(state)
        print(action_probs)
        probs = list(action_probs.values())
        probs /= sum(probs)
        actions = list(action_probs.keys())
        print(actions)
        best_action = np.random.choice(actions, p=probs)
        print(best_action)
        return best_action
        
        '''
        print("step")
        #print(dir(self.policy))
        print(self.policy.predict(state))
        return self.step_with_policy(state)[1]
        '''

    def _action_probabilities(self, state):
        """Returns action probabilities dict for a single batch."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(cur_player), dtype=tf.float32)
        info_state_vector = tf.constant(
            state.information_state_tensor(), dtype=tf.float32)
        if len(info_state_vector.shape) == 1:
            info_state_vector = tf.expand_dims(info_state_vector, axis=0)
        probs = self.policy((info_state_vector, legal_actions_mask),
                                    training=False)
        probs = probs.numpy()
        return {action: probs[0][action] for action in legal_actions}



def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = (
        "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=150 100,"
        "firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,bettingAbstraction=fcpa)")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())