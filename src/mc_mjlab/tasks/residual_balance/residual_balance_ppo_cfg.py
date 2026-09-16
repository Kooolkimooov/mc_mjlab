"""PPO settings for the residual balance task -- docs/ppo.md for why each value."""

from __future__ import annotations

from typing import Literal

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
  schedule: Literal["rollout_adaptive", "adaptive", "fixed"] = "rollout_adaptive",
  learning_rate: float = 1.0e-3,
  num_learning_epochs: int = 2,
  num_mini_batches: int = 2,
  objective_clipping: bool = True,
) -> RslRlOnPolicyRunnerCfg:
  """PPO settings, following mjlab's locomotion configs."""
  if max_iterations is None:
    max_iterations = iterations_for_budget(policy_steps_per_env, num_steps_per_env)
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # rsl_rl leaves the mean rows at nn.Linear's default: untrained RMS 0.094.
      class_name="mc_mjlab.rl.zero_init_actor:ZeroInitMLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      rnn_type=None,
      rnn_hidden_dim=256,
      distribution_cfg={
        "class_name": ("mc_mjlab.rl.squashed_gaussian:SquashedGaussianDistribution"),
        "init_std": 0.1,
        # Match the powered ResidualMPC setting. docs/ppo.md#std_range
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
      clip_param=0.2 if objective_clipping else 1.0e6,
      # Tenfold reduction, paired with the tighter ceiling. docs/ppo.md#entropy_coef
      entropy_coef=0.00005,
      # Their product is the adaptive schedule's step count: 20 events allow a
      # 1.5^20 = 3325x rate collapse in one iteration, 4 events only 5x.
      num_learning_epochs=num_learning_epochs,
      num_mini_batches=num_mini_batches,
      learning_rate=learning_rate,
      schedule="fixed" if schedule == "fixed" else "adaptive",
      # The 6.7 s discount horizon and 4.6 s 95% GAE trace cover delayed falls.
      gamma=0.997,
      lam=0.99,
      desired_kl=0.02,
      max_grad_norm=1.0,
      class_name=(
        "mc_mjlab.rl.rollout_adaptive_ppo:RolloutAdaptivePPO"
        if schedule == "rollout_adaptive"
        else "PPO"
      ),
    ),
    experiment_name=experiment_name,
    save_interval=20,
    num_steps_per_env=num_steps_per_env,
    max_iterations=max_iterations,
    logger="wandb",
  )
