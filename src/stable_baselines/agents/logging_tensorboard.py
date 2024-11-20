import tensorflow as tf
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Create SAC model
model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][-1]
        writer = self.logger.get_dir()
        with tf.summary.create_file_writer(writer).as_default():
            tf.summary.scalar("reward", reward, step=self.num_timesteps)
        return True

#Test
model.learn(total_timesteps=1000, callback=TensorboardCallback())
