"""Calibrate recovery authority from zero-residual nominal and post-push traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab import mdp
from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.residuals.recovery_authority import (
  FEATURE_NAMES,
  RecoveryCalibration,
  RecoveryFeatureExtractor,
  RecoveryFilter,
  detector_target,
)
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg


def collect(
  args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Collect feature, push-age, and episode-age tensors with zero residual."""
  cfg = _make_env_cfg(
    "position",
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    push_velocity=args.push_velocity,
    recovery_detector_path=None,
    disturbance="velocity",
  )
  cfg.seed = args.seed
  cfg.events["push_robot"].params["planar_speed"] = args.push_velocity
  env = ManagerBasedRlEnv(cfg, device=args.device)
  term = env.action_manager.get_term("mc_rtc_residual")
  assert isinstance(term, McRtcResidualActionBase)
  extractor = RecoveryFeatureExtractor(env, term)
  zmp_sensors = mdp.sensors._ZmpSensors(
    env, mdp.sensors.GROUND_CONTACT_SENSORS, "robot"
  )
  env_ids = torch.arange(env.num_envs, device=env.device)
  mdp.disturbances._push_term(env, "push_robot").disable(env_ids[env_ids % 4 < 2])
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  feature_rows: list[torch.Tensor] = []
  age_rows: list[torch.Tensor] = []
  episode_rows: list[torch.Tensor] = []
  env.reset()
  for step in range(args.steps):
    env.step(action)
    features = extractor()
    expected_dcm, _ = zmp_sensors.dcm_offset(env)
    if not torch.allclose(features[:, 0], expected_dcm, atol=1.0e-4):
      error = float((features[:, 0] - expected_dcm).abs().max())
      raise AssertionError(f"detector DCM differs from reward DCM by {error:.3g}")
    feature_rows.append(features.cpu())
    age_rows.append(mdp.observations.steps_since_push(env).cpu())
    episode_rows.append(env.episode_length_buf.cpu().clone())
    if (step + 1) % 250 == 0:
      print(f"[calibrate] collected {step + 1}/{args.steps} steps", flush=True)
  env.close()
  return (
    torch.stack(feature_rows),
    torch.stack(age_rows),
    torch.stack(episode_rows),
  )


def masks(
  ages: torch.Tensor, episode_ages: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return nominal, recovery, and late-recovery sample masks."""
  recovery_steps = round(2.0 / dt)
  warmup_steps = round(10.0 / dt)
  nominal = (episode_ages >= warmup_steps) & (ages >= mdp.disturbances.NEVER_AGE)
  recovery = (ages >= 1) & (ages <= recovery_steps)
  late = (ages > recovery_steps) & (ages <= round(5.0 / dt))
  return nominal, recovery, late


def replay(
  features: torch.Tensor,
  calibration: RecoveryCalibration,
  dt: float,
  episode_ages: torch.Tensor | None = None,
) -> torch.Tensor:
  """Replay the detector filter across a recorded time-by-env trace."""
  recovery_filter = RecoveryFilter(features.shape[1], features.device, calibration)
  output = []
  previous_age = None
  for index, row in enumerate(features):
    if episode_ages is not None:
      current_age = episode_ages[index]
      if previous_age is not None:
        recovery_filter.reset(current_age < previous_age)
      previous_age = current_age
    score, target = detector_target(row, calibration)
    onset_index = FEATURE_NAMES.index(calibration.onset_feature)
    output.append(
      recovery_filter.update(score, row[:, onset_index], target, dt).clone()
    )
  return torch.stack(output)


def fit(
  features: torch.Tensor,
  ages: torch.Tensor,
  episode_ages: torch.Tensor,
  dt: float,
) -> tuple[RecoveryCalibration, dict]:
  """Fit normalizers and a train-split threshold, then score the held-out envs."""
  nominal, recovery, late = masks(ages, episode_ages, dt)
  env_ids = torch.arange(features.shape[1])
  train_env = env_ids % 2 == 0
  test_env = ~train_env

  train_nominal = features[:, train_env][nominal[:, train_env]]
  train_recovery = features[:, train_env][recovery[:, train_env]]

  centers = torch.quantile(train_nominal, 0.95, dim=0)
  recovery_q80 = torch.quantile(train_recovery, 0.80, dim=0)
  nominal_q75 = torch.quantile(train_nominal, 0.75, dim=0)
  nominal_q25 = torch.quantile(train_nominal, 0.25, dim=0)
  scales = torch.maximum(recovery_q80 - centers, nominal_q75 - nominal_q25)
  scales = scales.clamp(min=torch.tensor((0.005, 0.02, 0.01, 0.05)))

  normalized = ((features - centers) / scales).amax(dim=2)
  train_scores = normalized[:, train_env][nominal[:, train_env]]
  recovery_scores = normalized[:, train_env][recovery[:, train_env]]

  angular_rise = features[1:, :, 1] - features[:-1, :, 1]
  valid_rise = episode_ages[1:] >= episode_ages[:-1]
  nominal_rise = nominal[1:] & valid_rise
  train_nominal_rise = angular_rise[:, train_env][nominal_rise[:, train_env]]

  best: tuple[float, float, RecoveryCalibration] | None = None
  closest: tuple[float, float, float, RecoveryCalibration] | None = None
  best_duty: tuple[float, float, float, RecoveryCalibration] | None = None

  for max_active_s in (1.80, 2.00):
    for rearm_s in (0.10, 0.25, 0.50):
      for rearm_quantile in (0.90, 0.95, 0.99):
        rearm_score = float(torch.quantile(train_scores, rearm_quantile))
        for onset_quantile in (0.990, 0.995, 0.999):
          onset_delta = float(torch.quantile(train_nominal_rise, onset_quantile))
          for quantile in torch.linspace(0.80, 0.999, 20):
            threshold = float(torch.quantile(train_scores, quantile))
            span = max(0.25, float(torch.quantile(recovery_scores, 0.80)) - threshold)
            candidate = RecoveryCalibration(
              centers=(
                float(centers[0]),
                float(centers[1]),
                float(centers[2]),
                float(centers[3]),
              ),
              scales=(
                float(scales[0]),
                float(scales[1]),
                float(scales[2]),
                float(scales[3]),
              ),
              threshold=threshold,
              activation_span=span,
              max_active_s=max_active_s,
              rearm_s=rearm_s,
              rearm_score=rearm_score,
              onset_delta=onset_delta,
            )
            authority = replay(
              features[:, train_env], candidate, dt, episode_ages[:, train_env]
            )

            duty = float(authority[nominal[:, train_env]].mean())
            recall = float((authority[recovery[:, train_env]] > 0.05).float().mean())
            late_rate = float((authority[late[:, train_env]] > 0.05).float().mean())
            if duty <= 0.05:
              key = (-recall, late_rate, duty)
              if best_duty is None or key < best_duty[:3]:
                best_duty = (-recall, late_rate, duty, candidate)

            if duty <= 0.05 and recall >= 0.80:
              key = (late_rate, duty, -recall)
              if closest is None or key < closest[:3]:
                closest = (late_rate, duty, -recall, candidate)

            if duty > 0.05 or recall < 0.80 or late_rate >= 0.01:
              continue
            key = (recall, max_active_s)
            if best is None or key > best[:2]:
              best = (recall, max_active_s, candidate)

  if best is None:
    detail = "no candidate met nominal-duty and recovery-recall gates"
    if closest is not None:
      detail = (
        f"best late rate {closest[0]:.4f} at duty {closest[1]:.4f}, "
        f"recall {-closest[2]:.4f}, calibration {closest[3]}"
      )
    elif best_duty is not None:
      detail = (
        f"best duty-passing recall {-best_duty[0]:.4f} at late rate "
        f"{best_duty[1]:.4f}, duty {best_duty[2]:.4f}, "
        f"calibration {best_duty[3]}"
      )
    raise RuntimeError(f"no detector candidate met all three train gates: {detail}")

  calibration = best[2]

  result = {
    "train": evaluate_split(
      features[:, train_env],
      nominal[:, train_env],
      recovery[:, train_env],
      late[:, train_env],
      calibration,
      dt,
      episode_ages[:, train_env],
    ),
    "heldout": evaluate_split(
      features[:, test_env],
      nominal[:, test_env],
      recovery[:, test_env],
      late[:, test_env],
      calibration,
      dt,
      episode_ages[:, test_env],
    ),
    "feature_quantiles": {
      name: {
        "nominal_q95": float(centers[index]),
        "recovery_q80": float(recovery_q80[index]),
        "scale": float(scales[index]),
      }
      for index, name in enumerate(FEATURE_NAMES)
    },
  }
  return calibration, result


def evaluate_split(
  features: torch.Tensor,
  nominal: torch.Tensor,
  recovery: torch.Tensor,
  late: torch.Tensor,
  calibration: RecoveryCalibration,
  dt: float,
  episode_ages: torch.Tensor,
) -> dict[str, float]:
  """Compute the three detector acceptance quantities on one env split."""
  authority = replay(features, calibration, dt, episode_ages)
  return {
    "nominal_duty": float(authority[nominal].mean()),
    "recovery_recall": float((authority[recovery] > 0.05).float().mean()),
    "late_activation": float((authority[late] > 0.05).float().mean()),
    "nominal_samples": int(nominal.sum()),
    "recovery_samples": int(recovery.sum()),
  }


def main() -> None:
  """Collect, fit, validate, and write one detector calibration JSON."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--steps", type=int, default=1500)
  parser.add_argument("--push-velocity", type=float, default=0.4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--trace-in", type=Path)
  parser.add_argument("--trace-out", type=Path)
  args = parser.parse_args()

  if args.trace_in is None:
    features, ages, episode_ages = collect(args)
    if args.trace_out is not None:
      args.trace_out.parent.mkdir(parents=True, exist_ok=True)
      torch.save((features, ages, episode_ages), args.trace_out)
  else:
    features, ages, episode_ages = torch.load(
      args.trace_in, map_location="cpu", weights_only=True
    )

  calibration, results = fit(features, ages, episode_ages, 0.02)
  heldout = results["heldout"]
  accepted = (
    heldout["nominal_duty"] <= 0.05
    and heldout["recovery_recall"] >= 0.80
    and heldout["late_activation"] < 0.01
  )

  payload = {
    "feature_names": FEATURE_NAMES,
    "centers": calibration.centers,
    "scales": calibration.scales,
    "threshold": calibration.threshold,
    "activation_span": calibration.activation_span,
    "attack_s": calibration.attack_s,
    "decay_tau_s": calibration.decay_tau_s,
    "cutoff": calibration.cutoff,
    "max_active_s": calibration.max_active_s,
    "rearm_s": calibration.rearm_s,
    "rearm_score": calibration.rearm_score,
    "onset_feature": calibration.onset_feature,
    "onset_delta": calibration.onset_delta,
    "calibration": {
      "seed": args.seed,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "push_velocity": args.push_velocity,
      "accepted": accepted,
      **results,
    },
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2) + "\n")
  print(json.dumps(payload, indent=2))
  print(f"[calibrate] {'PASS' if accepted else 'FAIL'} -> {args.output}")


if __name__ == "__main__":
  main()
