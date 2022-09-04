from stable_baselines3 import PPO
import torch


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)


if __name__ == "__main__":
    model = PPO.load("../logs/ppo/JavaBot-v1_44/rl_model_999999_steps")
    model.policy.to("cpu")
    onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)
    obs = [0, 0, 0, 0, -2, 0, 0, 0, 0, -2, 0, 0, 0, 0, -2, 0, 0, 0, 0, -2, -2, -2, -2, -2, -2, 9, 9, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    dummy_input = torch.Tensor(obs)

    torch.onnx.export(onnxable_model, dummy_input, "my_ppo_model.onnx", opset_version=9)
