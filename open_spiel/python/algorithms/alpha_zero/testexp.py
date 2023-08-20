import pyspiel


def alpha_zero(config):
  """Start all the worker processes for a full alphazero setup."""

  print(pyspiel.registered_games())
  game = pyspiel.load_game(config.game)