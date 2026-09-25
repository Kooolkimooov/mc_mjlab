"""PPO settings for the locomanip residual task -- docs/ppo.md for why each value."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

NUM_STEPS_PER_ENV = 64

MAX_ITERATIONS = 2000


def locomanip_ppo_cfg(
  experiment_name: str = "mc_rtc_locomanip_residual",
  max_iterations: int = MAX_ITERATIONS,
  num_steps_per_env: int = NUM_STEPS_PER_ENV,
) -> RslRlOnPolicyRunnerCfg:
  """Build the shared zero-init, squashed-Gaussian residual policy settings."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # rsl_rl leaves the mean rows at nn.Linear's default: untrained RMS 0.094.
      class_name="mc_mjlab.rl.zero_init_actor:ZeroInitMLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "mc_mjlab.rl.squashed_gaussian:SquashedGaussianDistribution",
        "init_std": 0.1,
        "std_range": (0.05, 0.15),
        "learn_std": True,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.00005,
      num_learning_epochs=2,
      num_mini_batches=2,
      learning_rate=1.0e-3,
      schedule="adaptive",
      # A 52 s manipulation cycle at 50 Hz: the discount has to reach the end of it.
      gamma=0.997,
      lam=0.99,
      desired_kl=0.02,
      max_grad_norm=1.0,
      # One rate decision per rollout from the KL it actually produced, rather
      # than four per-minibatch ones. docs/ppo.md#RolloutAdaptivePPO
      class_name="mc_mjlab.rl.rollout_adaptive_ppo:RolloutAdaptivePPO",
    ),
    experiment_name=experiment_name,
    save_interval=20,
    num_steps_per_env=num_steps_per_env,
    max_iterations=max_iterations,
    logger="wandb",
  )
