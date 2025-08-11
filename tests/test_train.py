from training.train import train

class Args:
    episodes = 1

def test_train_runs():
    train(Args())
