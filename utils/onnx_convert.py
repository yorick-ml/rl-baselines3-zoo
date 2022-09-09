import numpy as np
from stable_baselines3 import PPO
import torch
from stable_baselines3.common.preprocessing import preprocess_obs


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net
        self.training = False

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)


if __name__ == "__main__":
    model = PPO.load("../logs/ppo/JavaBot-v1_64/rl_model_10000_steps")
    model.policy.to("cpu")
    onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)
    obs = torch.tensor((0.3, 0.3, 0.3, 0.3))
    # dummy_input = torch.randn(4)
    torch.onnx.export(onnxable_model, obs, "my_ppo_model.onnx", opset_version=9, input_names=['zzz'])

    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_model = onnx.load("my_ppo_model.onnx")
    onnx.checker.check_model(onnx_model)

    # observation = np.zeros((1, 4)).astype(np.float32)
    obs = (0.3, 0.3, 0.3, 0.3)
    ort_sess = ort.InferenceSession("my_ppo_model.onnx")

    action, value = ort_sess.run(None, {'zzz': np.array(obs, dtype=np.float32)})
    print(action)
