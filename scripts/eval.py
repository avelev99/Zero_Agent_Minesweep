import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from training.eval import evaluate

if __name__ == "__main__":
    evaluate()
