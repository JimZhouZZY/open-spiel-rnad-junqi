# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

import logging
import sys
from absl import app
from absl import flags
import numpy as np

import pyspiel
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e1), "Number of train episodes.")
flags.DEFINE_boolean(
    "interactive_play",
    True,
    "Whether to run an interactive play with the agent after training.",
)

"""
_DICT_CHESS_CELL = {6: "雷", 5: "师", 4: "旅", 3: "团", 2: "炸", 1: "棋", 0: "空"}
def pretty_board(time_step):
    info_state = time_step.observations["info_state"]
    info_board = np.reshape(info_state[0], (13,6,3))
    board = [[f"\033[;;m空\033[0m"]*3]*6
    for chess_axis in range(1, len(info_board)):
        for i in range(6):
            for j in range(3):
                if info_board[chess_axis][i][j]:
                    if chess_axis >= 6:
                        axis = chess_axis - 6
                        board[i][j] = f"\033[;30;42m{_DICT_CHESS_CELL[axis]}\033[0m"
                    else:
                        axis = chess_axis + 0
                        board[i][j] = f"\033[;30;43m{_DICT_CHESS_CELL[axis]}\033[0m"
    string_board = "\n".join("".join([str(chess) for chess in row]) for row in board)
    return string_board
"""


def command_line_action(time_step):
    """Gets a valid action from the user on the command line."""
    current_player = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][current_player]
    action = -1
    while action not in legal_actions:
        print("Choose an action from {}:".format(legal_actions))
        sys.stdout.flush()
        action_str = input()
        try:
            action = int(action_str)
        except ValueError:
            continue
    return action


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


class JunQiEnv(rl_environment.Environment):
    def get_stringed_board(self):
        return self._state.serialize()

    def get_stringed_action(self):
        actions = self._state.legal_actions()
        return "\n".join(self._state.serialize_action(action) for action in actions)


def main(_):
    game = "junqi1"
    num_players = 2
    env = JunQiEnv(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [32, 32]
    replay_buffer_capacity = int(1e4)
    train_episodes = FLAGS.num_episodes
    loss_report_interval = 1000

    with tf.Session() as sess:
        dqn_agent = dqn.DQN(
            sess,
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity)
        tabular_q_agent = tabular_qlearner.QLearner(
            player_id=1, num_actions=num_actions)
        agents = [dqn_agent, tabular_q_agent]

        sess.run(tf.global_variables_initializer())

        # Train agent

        for ep in range(train_episodes):
            if ep and ep % loss_report_interval == 0:
                logging.info("[%s/%s] DQN loss: %s", ep, train_episodes, agents[0].loss)
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

        # Evaluate against random agent
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]
        r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
        logging.info("Mean episode rewards: %s", r_mean)

        if not FLAGS.interactive_play:
            return

        # 2. Play from the command line against the trained agent.
        human_player = 0
        while True:
            print("You are playing as %s", "O" if human_player else "X")
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == human_player:
                    agent_out = agents[human_player].step(time_step, is_evaluation=True)
                    # print("\n%s", agent_out)
                    print(env.get_stringed_board(), end="\n\n")
                    print(env.get_stringed_action())
                    action = command_line_action(time_step)
                else:
                    agent_out = agents[1 - human_player].step(time_step, is_evaluation=True)
                    action = agent_out.action
                time_step = env.step([action])

            print(env.get_stringed_board(), end="\n\n")

            print("End of game!")
            if time_step.rewards[human_player] > 0:
                print("You win")
            elif time_step.rewards[human_player] < 0:
                print("You lose")
            else:
                print("Draw")
            # Switch order of players
            human_player = 1 - human_player


if __name__ == "__main__":
    app.run(main)
