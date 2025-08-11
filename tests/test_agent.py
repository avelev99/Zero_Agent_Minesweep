from agent import EzAgent
import torch


def test_inference_shapes():
    agent = EzAgent((8, 8))
    obs = torch.zeros(1, 1, 8, 8)
    init = agent.initial_inference(obs)
    assert init["policy_logits"].shape[-1] == 192
    support_dim = agent.support.numel()
    assert init["value_logits"].shape[-1] == support_dim
    assert init["reward_logits"].shape[-1] == support_dim

    action = torch.zeros(1, dtype=torch.long)
    recur = agent.recurrent_inference(init["state"], action)
    assert recur["policy_logits"].shape[-1] == 192
    assert recur["reward_logits"].shape[-1] == support_dim
