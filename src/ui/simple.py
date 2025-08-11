from minesweeper_env import MinesweeperEnv


def main():
    env = MinesweeperEnv()
    obs, _ = env.reset()
    env.render()
    done = False
    while not done:
        a = int(input("action: "))
        obs, reward, done, _, _ = env.step(a)
        env.render()
        print("reward", reward)

if __name__ == "__main__":
    main()
