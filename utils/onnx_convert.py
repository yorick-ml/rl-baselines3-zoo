import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import torch
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import BaseModel


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, extractor, action_net, value_net, obser_space, feature_extractor, policy):
        super(OnnxablePolicy, self).__init__()
        self.policy = policy
        self.features_extractor = feature_extractor
        self.obser_space = obser_space
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net
        self.training = False

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        # _obs = self.feature_extractor(preprocessed_obs)

        # preprocessed_obs = preprocess_obs(observation, observation_space=self.obser_space, normalize_images=False)
        # observation, _ = self.policy.obs_to_tensor(observation)
        _obs = self.policy.extract_features(observation)
        action_hidden, value_hidden = self.extractor(_obs)
        return self.action_net(action_hidden), self.value_net(value_hidden)


if __name__ == "__main__":
    model = PPO.load("../rl_model_24_2M_steps.zip")
    model.policy.to("cpu")
    feature_extractor = model.policy.features_extractor
    obser_space = model.observation_space
    onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net,
                                    obser_space, feature_extractor, model.policy)
    obs = {
        'grid': torch.randint(-2, 2, (1, 25)),
        'modificators': torch.tensor([[0, 0, 0, 0, 0]]),
        'my_mods': torch.tensor([[0, 0, 0, 0]]),
        'opp_mods': torch.tensor([[0, 0, 0, 0]]),
        'positions': torch.randint(0, 100, (1, 4)),
    }

    torch.onnx.export(onnxable_model, args=(obs, {}), f="my_ppo_model.onnx", opset_version=9, training=False,
                      input_names=['grid', 'modificators', 'my_mods', 'opp_mods', 'positions'])

    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_model = onnx.load("my_ppo_model.onnx")
    onnx.checker.check_model(onnx_model)

    obs = {'grid': np.array([[-2, 0, -2, -1, 0, 0, -2, 1, 1, -1, -1, 1, 0, -2, 0, -1, -1, -1, 1, 1, -1, 0, -1, -1, -2]], dtype=np.int64),
           'modificators': np.array((0, 0, 0, 0, 0), dtype=np.int64, ndmin=2),
           'my_mods': np.array((0, 0, 0, 0), dtype=np.int64, ndmin=2),
           'opp_mods': np.array((0, 0, 0, 0), dtype=np.int64, ndmin=2),
           'positions': np.array([[10, 0, 0, 0]], dtype=np.int64),
           }

    ort_sess = ort.InferenceSession("my_ppo_model.onnx")
    action, value = ort_sess.run(None, obs)
    print(action)
