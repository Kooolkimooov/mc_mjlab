"""PPO settings for the residual balance task -- docs/ppo.md for why each value."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def residual_balance_ppo_cfg(
  max_iterations: int = 500, experiment_name: str = "mc_rtc_residual_balance"
) -> RslRlOnPolicyRunnerCfg:
  """PPO settings, following mjlab's locomotion configs."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # rsl_rl leaves the mean rows at nn.Linear's default: untrained RMS 0.094.
      class_name="mc_mjlab.tasks.zero_init_actor:ZeroInitMLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.1,
        "std_type": "scalar",
        "std_range": (0.05, 0.30),
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0005,
      # Their product is the adaptive schedule's step count: 20 events allow a
      # 1.5^20 = 3325x rate collapse in one iteration, 4 events only 5x.
      num_learning_epochs=2,
      num_mini_batches=2,
      learning_rate=1.0e-3,
      schedule="adaptive",
      # The 6.7 s discount horizon and 4.6 s 95% GAE trace cover delayed falls.
      gamma=0.997,
      lam=0.99,
      desired_kl=0.02,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=20,
    num_steps_per_env=256,
    max_iterations=max_iterations,
    logger="wandb",
  )
