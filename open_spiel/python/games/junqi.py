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

# Lint as python3
"""Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""
import random
import copy

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 6
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="junqi",
    long_name="Python Simplized Junqi",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)  #
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS * _NUM_CELLS + 1,
    max_chance_outcomes=0,  #
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=100 # _NUM_CELLS * _NUM_CELLS
)


class JunQiGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return JunQiState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            return BoardObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)


class JunQiState(pyspiel.State):
    """A python version of the Tic-Tac-Toe state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._cur_player = 0
        self._player0_score = 0.0
        self._is_terminal = False
        self.board = []
        chesslist = [9, 9, 8, 7, 7, 6, 6, 2, 1]
        random.shuffle(chesslist)
        #[self.board.append([Chess(chesslist[i * _NUM_COLS + j], 1) for j in range(_NUM_COLS)]) for i in
         #range(_NUM_ROWS // 2)]
        #random.shuffle(chesslist)
        #[self.board.append([Chess(chesslist[i * _NUM_COLS + j], 0) for j in range(_NUM_COLS)]) for i in
         #range(_NUM_ROWS // 2)]
        num_board = [[9, 1, 7],
                     [2, 9, 7],
                     [6, 6, 8],
                     [8, 7, 6],
                     [7, 9, 2],
                     [6, 1, 9]]
        [self.board.append([Chess(num_board[i][j], 1) for j in range(_NUM_COLS)]) for i in range(_NUM_ROWS // 2)]
        [self.board.append([Chess(num_board[i+3][j], 0) for j in range(_NUM_COLS)]) for i in range(_NUM_ROWS // 2)]

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        actions = []
        idx = 0
        for i in range(_NUM_ROWS):
            for j in range(_NUM_COLS):
                for k in range(_NUM_ROWS):
                    for l in range(_NUM_COLS):
                        if self.is_legal_action([i, j], [k, l]):
                            actions.append(idx)
                            flatten_actions[idx] = [[i, j], [k, l]]
                        idx += 1
        if actions == []:
            actions.append(324)
        #print(actions)
        return actions

    def is_legal_action(self, from_pos, to_pos):
        # print(self.board[from_pos[0]][from_pos[1]].country,self.current_player())
        if (self.board[from_pos[0]][from_pos[1]].country == self.current_player()
                and to_pos in [[from_pos[0] + 1, from_pos[1]],
                               [from_pos[0] - 1, from_pos[1]],
                               [from_pos[0], from_pos[1] + 1],
                               [from_pos[0], from_pos[1] - 1]]
                and self.board[to_pos[0]][to_pos[1]].country != self.board[from_pos[0]][from_pos[1]].country
                and self.board[from_pos[0]][from_pos[1]] != 9 and self.board[from_pos[0]][from_pos[1]] != 1):
            return True
        else:
            return False

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        if action == 324:
            self._is_terminal = True
            self._player0_score = -1.0
            return
        # print(f"Action:{flatten_actions[action]} AKA {action}")
        from_pos, to_pos = flatten_actions[action][0], flatten_actions[action][1]
        attacker = self.board[from_pos[0]][from_pos[1]]
        if self.board[to_pos[0]][to_pos[1]].country == -1:
            self.board[to_pos[0]][to_pos[1]] = copy.deepcopy(attacker)
        else:
            defender = self.board[to_pos[0]][to_pos[1]]
            if attacker == 2 or defender == 2 or attacker == defender:
                self.board[to_pos[0]][to_pos[1]] = Chess(0, -1)
            elif attacker > defender:
                self.board[to_pos[0]][to_pos[1]] = copy.deepcopy(attacker)
                if defender == 1:
                    self._is_terminal = True
                    self._player0_score = 1.0 if self._cur_player == 0 else -1.0
            elif attacker < defender:
                pass
            if self.no_chess(0):
                self._is_terminal = True
                self._player0_score -= 1.0
            elif self.no_chess(1):
                self._is_terminal = True
                self._player0_score += 1.0
        self.board[from_pos[0]][from_pos[1]] = Chess(0, -1)
        self._cur_player = 1 - self._cur_player
        return

    def _action_to_string(self, player, action):
        """Action -> string."""
        from_pos, to_pos = flatten_actions[action][0], flatten_actions[action][1]
        return "{}({},{})".format("0" if player == 0 else "1", from_pos, to_pos)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    def no_chess(self, country):
        for row in self.board:
            for chess in row:
                if chess.country == country and chess != 9 and chess != 1:
                    return False
        return True

    def serialize(self):
        return _board_to_string(self.board)

    def serialize_action(self, action):
        return(f"Action:{flatten_actions[action]} AKA {action}")

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return _board_to_string(self.board)


class BoardObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = (_NUM_CHESS_TYPE * _NUM_PLAYERS + 1, _NUM_ROWS, _NUM_COLS)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs.fill(0)
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                if state.board[row][col].country != -1:
                    cell_state = _DICT_CHESS_CELL[int(repr(state.board[row][col]))] + _NUM_CHESS_TYPE * \
                                 state.board[row][col].country
                else:
                    cell_state = 0
                obs[cell_state, row, col] = 1
        pass

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        del player
        return _board_to_string(state.board)


# Helper functions for game details.

_NUM_CHESS_TYPE = 6
_DICT_CHESS_CELL = {9: 6, 8: 5, 7: 4, 6: 3, 2: 2, 1: 1, 0: 0}
flatten_actions = [[0, 0], [0, 0]] * (_NUM_COLS * _NUM_ROWS) ** 2


class Chess():
    def __init__(self, num, country=-1):
        self.num = num
        self.country = country if num != 0 else -1

    def __str__(self):
        if self.country == -1:
            return f"\033[;;m{self.num}\033[0m"
        elif self.country == 0:
            return f"\033[;30;43m{self.num}\033[0m"
        elif self.country == 1:
            return f"\033[;30;42m{self.num}\033[0m"

    def __repr__(self):
        return repr(self.num)

    def __eq__(self, other):
        return self.num == other

    def __lt__(self, other):
        return self.num < other

    def __gt__(self, other):
        return self.num > other


def _board_to_string(board):
    """Returns a string representation of the board."""
    return "\n".join("".join([str(chess) for chess in row]) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, JunQiGame)
