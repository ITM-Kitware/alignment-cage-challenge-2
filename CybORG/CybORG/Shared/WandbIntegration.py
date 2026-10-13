from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np

class CAGEWandbCallback(BaseCallback):
    def __init__(self, log_freq=500, window=500): #log per 5 games or so
        super().__init__()
        self.log_freq = log_freq
        self.window = window
        self.episode_rewards = deque(maxlen=window)  # store rewards for smoothing
        self.infos = {}

    def _on_step(self) -> bool:

        # Track recent rewards
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]

            if len(rewards) > 0:
                self.episode_rewards.append(rewards[-1])

        custominfo = self.locals.get("infos", [{'info': {}}])[-1] #other custom env info

        for key in [
            "CAGE Reward",
            "Alignment Reward"
        ]:
            self.infos.setdefault(key, deque(maxlen=self.window)).append(custominfo['info'].get(key, 0))
            
        # Log every N calls
        if self.n_calls % self.log_freq == 0:
            log_data = {"timesteps": self.num_timesteps}
            # Pull SB3 logger metrics
            for key in [
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/approx_kl",
                "train/explained_variance",
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "time/fps",
            ]:
                value = self.model.logger.name_to_value.get(key)
                if value is not None:
                    log_data[key] = float(value)

            # Add recent reward metrics (deques are already capped at self.window)
            if len(self.episode_rewards) > 0:
                log_data["reward/mean_recent"] = np.mean(self.episode_rewards)
                log_data["reward/last"] = self.episode_rewards[-1]
                log_data["reward/CAGE_reward"] = np.mean(self.infos["CAGE Reward"])
                log_data["reward/Alignment_reward"] = np.mean(self.infos["Alignment Reward"])

                    
            wandb.log(log_data)

        return True