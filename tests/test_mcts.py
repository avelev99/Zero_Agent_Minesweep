import numpy as np
from agent import EzAgent, run_mcts
from minesweeper_env import MinesweeperEnv


def test_mcts_selects_valid_action():
    env = MinesweeperEnv(size=(4, 4), n_mines=0, seed=0)
    agent = EzAgent(env.size)
    obs, _ = env.reset()
    mask = env.action_mask()
    action, policy = run_mcts(agent, obs, mask, num_simulations=2)
    assert mask[action]
    assert np.isclose(policy.sum(), 1.0)
