import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class Node:
    visit_count: int
    value_sum: float
    prior: float
    state: torch.Tensor | None = None
    reward: float = 0.0
    children: Dict[int, "Node"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


def ucb_score(parent: Node, child: Node, c: float = 1.25) -> float:
    prior = child.prior
    if child.visit_count == 0:
        q = 0.0
    else:
        q = child.value()
    ucb = q + c * prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return ucb


def select_child(node: Node) -> Tuple[int, Node]:
    best_score = -float("inf")
    best_action = -1
    best_child = None
    for action, child in node.children.items():
        score = ucb_score(node, child)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def expand(root: Node, policy_logits: torch.Tensor, action_mask: np.ndarray | None):
    if action_mask is not None:
        mask = torch.tensor(action_mask, dtype=torch.bool, device=policy_logits.device)
        policy_logits = policy_logits.masked_fill(~mask, -1e9)
    policy = torch.softmax(policy_logits, dim=1)[0]
    for a, p in enumerate(policy.tolist()):
        root.children[a] = Node(visit_count=0, value_sum=0.0, prior=p)


def run_mcts(agent, obs: np.ndarray, action_mask: np.ndarray, num_simulations: int = 10):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    network_output = agent.initial_inference(obs_tensor)
    root = Node(0, 0.0, 1.0, state=network_output["state"])
    expand(root, network_output["policy_logits"], action_mask)
    root_value = network_output["value"].item()

    for _ in range(num_simulations):
        node = root
        search_path = [node]
        # Selection
        while node.children:
            action, node = select_child(node)
            search_path.append(node)
        parent = search_path[-2]
        action_tensor = torch.tensor([action])
        network_output = agent.recurrent_inference(parent.state, action_tensor)
        node.state = network_output["state"]
        node.reward = network_output["reward"].item()
        expand(node, network_output["policy_logits"], None)
        value = network_output["value"].item()
        # Backpropagation
        for n in reversed(search_path):
            n.value_sum += value
            n.visit_count += 1
            value = n.reward + value

    # Choose action with highest visit count
    visits = np.array([child.visit_count for child in root.children.values()])
    best_action = int(np.argmax(visits))
    policy_target = visits / visits.sum()
    return best_action, policy_target
