"""PPO settings for the residual balance task -- docs/ppo.md for why each value."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

NUM_STEPS_PER_ENV = 256

#: Iterations are derived from this, not typed. docs/ppo.md#training-budget
POLICY_STEPS_PER_ENV = 128_000


def iterations_for_budget(policy_steps_per_env: int, num_steps_per_env: int) -> int:
  """Iterations that spend a per-env policy-step budget at this rollout length."""
  return max(1, round(policy_steps_per_env / num_steps_per_env))


def residual_balance_ppo_cfg(
  max_iterations: int | None = None,
  experiment_name: str = "mc_rtc_residual_balance",
  num_steps_per_env: int = NUM_STEPS_PER_ENV,
  policy_steps_per_env: int = POLICY_STEPS_PER_ENV,
) -> RslRlOnPolicyRunnerCfg:
  """PPO settings, following mjlab's locomotion configs."""
  if max_iterations is None:
    max_iterations = iterations_for_budget(policy_steps_per_env, num_steps_per_env)
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
    num_steps_per_env=num_steps_per_env,
    max_iterations=max_iterations,
    logger="wandb",
  )
