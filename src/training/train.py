import argparse
import torch
import numpy as np
from minesweeper_env import MinesweeperEnv
from agent import EzAgent, run_mcts


def train(args):
    env = MinesweeperEnv(size=(8, 8), n_mines=10, seed=0)
    agent = EzAgent(env.size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            mask = env.action_mask()
            action, policy_target = run_mcts(agent, obs, mask, num_simulations=5)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            logits = agent.initial_inference(obs_tensor)["policy_logits"]
            target = torch.tensor([policy_target], dtype=torch.float32)
            loss = torch.sum(-target * torch.log_softmax(logits, dim=1))
            obs, reward, done, _, _ = env.step(action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_reward += reward
        print(f"Episode {episode}: reward {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    train(args)
