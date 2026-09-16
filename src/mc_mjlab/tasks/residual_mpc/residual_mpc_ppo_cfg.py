"""PPO configuration for the ResidualMPC reproduction."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def residual_mpc_ppo_cfg(
  experiment_name: str = "mc_rtc_residual_mpc",
  max_iterations: int = 1000,
) -> RslRlOnPolicyRunnerCfg:
  """Build the cited PPO settings with the repository's bounded distribution."""
  model = RslRlModelCfg(
    class_name="mc_mjlab.rl.zero_init_actor:ZeroInitMLPModel",
    hidden_dims=(256, 256, 256),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      # Safety choice: bounded actions and 0.1 initial std. docs/residual-mpc.md
      "class_name": "mc_mjlab.rl.squashed_gaussian:SquashedGaussianDistribution",
      "init_std": 0.1,
      # A bound, not a pressure: any entropy bonus walks std to this ceiling, and
      # reward per step falls 29% doing it. docs/residual-mpc.md#std_range
      "std_range": (0.05, 0.15),
      "learn_std": True,
    },
  )
  return RslRlOnPolicyRunnerCfg(
    actor=model,
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 256), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # At 0.1 the entropy term outweighed the surrogate 8.5x and pinned std to
      # its ceiling for a whole run. docs/residual-mpc.md#entropy_coef
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    logger="wandb",
  )
