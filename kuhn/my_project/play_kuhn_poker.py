from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner


def get_env(poker):
    num_players = 2
    env_configs = {"players": num_players}
    env = rl_environment.Environment(poker, **env_configs)

    return env

def convert_to_jqk(n):
    if n == '0':
        return "J"
    if n == '1':
        return "Q"
    if n == '2':
        return "K"
    else:
        raise Exception

def convert_to_int(play):
    if play == "pass" or play == "Pass":
        return 0
    if play == "bet" or play == "Bet":
        return 1
    else:
        raise Exception

def convert_to_play(i):
    if i == 0:
        return "Pass"
    if i == 1:
        return "Bet"
    else:
        raise Exception

def show_card(i):
    card = convert_to_jqk(i)
    return "This is the card you were dealt: " + card

def main(_):
    poker = "kuhn_poker"

    env = get_env(poker)
    num_players = 2

    player_id = random.randint(0, 1)
    if player_id == 0:
        pos = "first"
    else:
        pos = "second"
    print()
    print("Welcome to Kuhn Poker. You play " + pos + ".")
    print()

    time_step = env.reset()

    # State is structured as follows for Kuhn Poker:
    # <card dealt to player 0> <card dealt to player 1> <action taken by player 0>
    #           <action taken by player 1> <action taken by player 0 if applicable>
    # For example if state == 2 1 p b b:
    #       Player 0 was dealt 2, player 1 was dealt 1.
    #       Player 0 chose to pass, player 1 chose to bet, player 0 chose to bet.
    # For example if state == 0 1 b p:
    #       Player 0 was dealt 0, player 1 was dealt 1.
    #       Player 0 chose to bet, player 1 chose to pass.
    state = env.get_state

    card = str(state).split()[player_id]
    print(show_card(card))

    if player_id == 0:
        # The human plays
        print("You have the option of playing either Pass or Bet.")
        play = input("What do you choose to do? ")
        play_int = convert_to_int(play)
        actions = [play_int]
        time_step = env.step(actions)

        # The AI plays
        play_int = random.randint(0, 1)
        play = convert_to_play(play_int)
        print("The AI chose to play: " + play)
        actions = [play_int]
        time_step = env.step(actions)

        if not time_step.last():
            # The human plays
            show_card(card)
            print("You have the option of playing either Pass or Bet.")
            play = input("What do you choose to do? ")
            play_int = convert_to_int(play)
            actions = [play_int]
            time_step = env.step(actions)

    else:
        # The AI plays
        play_int = random.randint(0, 1)
        play = convert_to_play(play_int)
        print("The AI chose to play: " + play)
        actions = [play_int]
        time_step = env.step(actions)

        # The human plays
        print("You have the option of playing either Pass or Bet.")
        play = input("What do you choose to do? ")
        play_int = convert_to_int(play)
        actions = [play_int]
        time_step = env.step(actions)

        if not time_step.last():
            # The AI plays
            play_int = random.randint(0, 1)
            play = convert_to_play(play_int)
            print("The AI chose to play: " + play)
            actions = [play_int]
            time_step = env.step(actions)

    # The game is over now
    print()
    rewards = time_step.rewards
    your_rewards = rewards[player_id]
    ai_rewards = rewards[(player_id + 1) % 2]
    if your_rewards > ai_rewards:
        print("You won!")
    elif your_rewards == ai_rewards:
        print("It's a tie!")
    else:
        print("You lost!")
    print("Your rewards: " + str(your_rewards))
    print("The AI's rewards: " + str(ai_rewards))


if __name__ == "__main__":
    app.run(main)
