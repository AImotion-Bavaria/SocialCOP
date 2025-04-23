
import optuna





def ppo_hyper_params(trial: optuna.Trial) -> dict:
        """
        Sample PPO hyperparameters for Optuna trial.

        Parameters:
        trial (optuna.Trial): Optuna trial object.

        Returns:
        dict: Dictionary of sampled hyperparameters.
        """
        batch_size = 64
        possible_n_steps = [i for i in range(5, 2049) if i % batch_size == 0]
        n_steps = trial.suggest_categorical("n_steps", possible_n_steps)
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "n_steps": n_steps,
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2),
            "vf_coef": 0.5, #trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": 0.5,#trial.suggest_float("max_grad_norm", 0.3, 10),
            "batch_size": batch_size,  # Include the batch size in kwargs
        }

    