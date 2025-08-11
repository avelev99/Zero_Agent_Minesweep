from minesweeper_env import MinesweeperEnv
from agent import EzAgent
import torch


def evaluate():
    env = MinesweeperEnv()
    agent = EzAgent(env.size)
    obs, _ = env.reset()
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        action = agent.initial_inference(obs_tensor)["policy_logits"].argmax().item()
        obs, reward, done, _, _ = env.step(action)
        env.render()
    print("Reward", reward)


if __name__ == "__main__":
    evaluate()
