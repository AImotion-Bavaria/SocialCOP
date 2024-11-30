import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

#model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.locals["rewards"][-1]
        self.logger.record("reward", value)
        return True


#model.learn(50000, callback=TensorboardCallback())