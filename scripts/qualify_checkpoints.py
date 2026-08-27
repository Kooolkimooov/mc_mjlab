"""Qualify every validation checkpoint on paired deterministic scenarios."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.tasks import mdp
from mc_mjlab.tasks.residual_balance.curriculum_stages import (
  ACHIEVEMENT_STAGES,
  MINIMUM_QUALIFICATION_SEEDS,
  REQUIRED_QUALIFICATION_SCENARIOS,
  achievement_contract,
  achievement_contract_sha256,
  achievement_stage,
)
from mc_mjlab.tasks.residual_balance.qualification_strata import (
  StratifiedDiagnostics,
  StratumRecord,
)
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)

SCENARIOS = ("nominal", "current_kick", "finite_impulse", "robust")
DISTURBANCE_WARMUP_S = 10.0
POLICY_STEP_S = 0.02


@dataclass
class Episode:
  """One fixed-schedule episode and its accumulated outputs."""

  checkpoint: str
  scenario: str
  seed: int
  env_id: int
  pair: int
  arm: str
  length: int
  terminations: dict[str, int]
  rewards: dict[str, float]
  metrics: dict[str, float]


def resolve_checkpoints(inputs: list[str]) -> list[Path]:
  """Expand checkpoint paths, directories, and globs in iteration order."""
  paths: set[Path] = set()
  for value in inputs:
    path = Path(value)
    if path.is_dir():
      paths.update(path.glob("model_*.pt"))
    elif any(char in value for char in "*?["):
      paths.update(Path(item) for item in glob.glob(value))
    elif path.is_file():
      paths.add(path)
    else:
      raise FileNotFoundError(f"checkpoint input matches nothing: {value}")

  def key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"model_(\d+)$", path.stem)
    return (str(path.parent), int(match.group(1)) if match else -1, path.name)

  return sorted((path.resolve() for path in paths), key=key)


def scenario_cfg(name: str, seed: int, args) -> ManagerBasedRlEnvCfg:
  """Build a deterministic qualification cfg for one scenario."""
  cfg = _make_env_cfg(
    control=args.control,
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    push_velocity=0.0,
    console_output="none",
    disturbance="none",
    authority_set=args.authority_set,
    controller_history=args.controller_history,
    proprio_history=args.proprio_history,
    randomization_stage=1 if name == "robust" else 0,
  )
  cfg.seed = seed
  cfg.auto_reset = False
  cfg.episode_length_s = args.episode_length_s
  cfg.events["reset_base"].params["pose_range"] = {}
  return cfg


class PairedDisturbances:
  """Apply a deterministic schedule shared by both arms of each episode pair."""

  def __init__(
    self, env, scenario: str, seed: int, achievement_stage_index: int | None = None
  ) -> None:
    self.env = env
    self.scenario = scenario
    self.seed = seed
    self.achievement_stage = (
      achievement_stage(achievement_stage_index)
      if achievement_stage_index is not None
      else None
    )
    self.asset = env.scene["robot"]
    self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.force = torch.zeros(env.num_envs, 1, 3, device=env.device)
    self.torque = torch.zeros_like(self.force)

  def before_step(
    self, active: torch.Tensor, pairs: list[int], episode_steps: torch.Tensor
  ) -> None:
    """Expire wrenches and trigger disturbances at fixed episode ages."""
    expired = (self.remaining == 0) & (self.force.square().sum(dim=(1, 2)) > 0.0)
    if bool(expired.any()):
      self._write_zeros(expired.nonzero(as_tuple=False).flatten())
    if self.scenario != "nominal":
      first = round(10.0 / self.env.step_dt)
      interval = round(6.0 / self.env.step_dt)
      due = (
        active & (episode_steps >= first) & ((episode_steps - first) % interval == 0)
      )
      ids = due.nonzero(as_tuple=False).flatten()
      if ids.numel():
        occurrences = ((episode_steps[ids] - first) // interval).tolist()
        self._trigger(ids, pairs, occurrences)

  def after_step(self) -> None:
    """Advance the finite-wrench duration by one policy step."""
    self.remaining.clamp_min_(0)
    self.remaining -= (self.remaining > 0).long()

  def clear(self, env_ids: torch.Tensor) -> None:
    """Clear any finite wrench before resetting an environment."""
    if env_ids.numel():
      self._write_zeros(env_ids)

  def _trigger(
    self, env_ids: torch.Tensor, pairs: list[int], occurrences: list[int]
  ) -> None:
    delta_b = torch.zeros(len(env_ids), 3, device=self.env.device)
    durations = torch.zeros(len(env_ids), device=self.env.device)
    heights = torch.zeros(len(env_ids), device=self.env.device)
    for row, (env_id, occurrence) in enumerate(
      zip(env_ids.tolist(), occurrences, strict=True)
    ):
      rng = random.Random(
        self.seed * 1_000_003
        + env_id * 10_007
        + pairs[env_id] * 101
        + occurrence * 17
        + sum(map(ord, self.scenario))
      )
      angle = rng.uniform(-math.pi, math.pi)
      if self.achievement_stage is None:
        dv = rng.uniform(0.50, 0.60) if self.scenario == "robust" else 0.40
      elif self.scenario == "robust":
        dv = rng.uniform(*self.achievement_stage.robust_velocity_range)
      else:
        dv = self.achievement_stage.qualification_velocity
      delta_b[row, :2] = torch.tensor(
        [dv * math.cos(angle), dv * math.sin(angle)], device=self.env.device
      )
      durations[row] = rng.uniform(0.08, 0.20)
      heights[row] = rng.uniform(0.0, 0.25)
    if self.scenario == "current_kick":
      self._velocity_kick(env_ids, delta_b)
    else:
      self._finite_impulse(env_ids, delta_b, durations, heights)
    mdp.record_disturbance(self.env, env_ids, delta_b)

  def _velocity_kick(self, env_ids: torch.Tensor, delta_b: torch.Tensor) -> None:
    quat = self.asset.data.root_link_quat_w[env_ids]
    delta_w = quat_apply(quat, delta_b)
    velocity = self.asset.data.root_link_vel_w[env_ids].clone()
    velocity[:, :3] += delta_w
    self.asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _finite_impulse(
    self,
    env_ids: torch.Tensor,
    delta_b: torch.Tensor,
    durations: torch.Tensor,
    heights: torch.Tensor,
  ) -> None:
    quat = self.asset.data.root_link_quat_w[env_ids]
    delta_w = quat_apply(quat, delta_b)
    body_ids = self.asset.indexing.body_ids
    mass = self.env.sim.model.body_mass[env_ids][:, body_ids].sum(dim=1)
    force = mass.unsqueeze(-1) * delta_w / durations.unsqueeze(-1)
    offset_b = torch.zeros_like(force)
    offset_b[:, 2] = heights
    offset_w = quat_apply(quat, offset_b)
    torque = torch.cross(offset_w, force, dim=1)
    self.force[env_ids, 0] = force
    self.torque[env_ids, 0] = torque
    self.remaining[env_ids] = torch.ceil(durations / self.env.step_dt).long()
    self.asset.write_external_wrench_to_sim(
      self.force[env_ids], self.torque[env_ids], env_ids=env_ids, body_ids=[0]
    )

  def _write_zeros(self, env_ids: torch.Tensor) -> None:
    zeros = torch.zeros(len(env_ids), 1, 3, device=self.env.device)
    self.asset.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids, body_ids=[0])
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.remaining[env_ids] = 0


def _metric_snapshot(env, done: torch.Tensor) -> dict[str, list[float]]:
  """Read true episode metric reductions before reset clears their buffers."""
  manager = env.metrics_manager
  counts = manager._step_count[done].float().clamp(min=1.0)
  output: dict[str, list[float]] = {}
  for index, name in enumerate(manager.active_terms):
    reduce = manager._term_cfgs[index].reduce
    if reduce == "max":
      values = manager._episode_max[name][done]
    elif reduce == "last":
      values = manager._step_values[done, index]
    else:
      values = manager._episode_sums[name][done] / counts
    output[name] = values.tolist()
  return output


def _reset_done(env, env_ids: torch.Tensor) -> None:
  """Recycle envs without appending an extra observation-history frame."""
  env._reset_idx(env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()


def run_checkpoint(
  checkpoint: Path, scenario: str, seed: int, args
) -> tuple[list[Episode], list[StratumRecord]]:
  """Run one checkpoint and scenario to a fixed paired episode count."""
  torch.manual_seed(seed)
  cfg = scenario_cfg(scenario, seed, args)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  runner = ResidualBalanceOnPolicyRunner(
    wrapped,
    asdict(residual_balance_ppo_cfg(recurrent=args.recurrent)),
    device=args.device,
  )
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)
  residual_term = env.action_manager.get_term("mc_rtc_residual")
  if not isinstance(residual_term, McRtcResidualActionBase):
    raise TypeError(f"unexpected residual action type: {type(residual_term).__name__}")
  recovery_s = float(env.reward_manager.get_term_cfg("recovery_dcm").params["window_s"])
  stratified = StratifiedDiagnostics(
    env, residual_term, DISTURBANCE_WARMUP_S, recovery_s
  )
  disturbances = PairedDisturbances(env, scenario, seed, args.achievement_stage)
  counts = [[0, 0] for _ in range(env.num_envs)]
  current_policy = torch.tensor(
    [bool(env_id % 2) for env_id in range(env.num_envs)],
    dtype=torch.bool,
    device=env.device,
  )
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  pairs = [0] * env.num_envs
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  episodes: list[Episode] = []
  strata: list[StratumRecord] = []
  term_names = env.termination_manager.active_terms
  reward_names = env.reward_manager.active_terms
  env.reset()
  while bool(active.any()):
    disturbances.before_step(active, pairs, env.episode_length_buf)
    action.zero_()
    if bool((current_policy & active).any()):
      with torch.inference_mode():
        proposed = policy(wrapped.get_observations())
      action[current_policy & active] = proposed[current_policy & active]
    _, _, terminated, time_outs, _ = env.step(action)
    policy.reset(terminated | time_outs)
    disturbances.after_step()
    stratified.capture(active)
    done = (terminated | time_outs).nonzero(as_tuple=False).flatten()
    if done.numel() == 0:
      continue
    metric_values = _metric_snapshot(env, done)
    reward_values = {
      name: env.reward_manager._episode_sums[name][done].tolist()
      for name in reward_names
    }
    termination_terms = {
      name: env.termination_manager.get_term(name) for name in term_names
    }
    termination_values = {
      name: values[done].tolist() for name, values in termination_terms.items()
    }
    lengths = env.episode_length_buf[done].tolist()
    strata.extend(
      stratified.finish(
        str(checkpoint),
        scenario,
        seed,
        done,
        pairs,
        current_policy,
        termination_terms,
      )
    )
    for row, env_id in enumerate(done.tolist()):
      if not bool(active[env_id]):
        continue
      arm_index = int(current_policy[env_id])
      episodes.append(
        Episode(
          checkpoint=str(checkpoint),
          scenario=scenario,
          seed=seed,
          env_id=env_id,
          pair=pairs[env_id],
          arm="policy" if arm_index else "baseline",
          length=int(lengths[row]),
          terminations={
            name: int(termination_values[name][row]) for name in term_names
          },
          rewards={name: float(reward_values[name][row]) for name in reward_names},
          metrics={name: float(metric_values[name][row]) for name in metric_values},
        )
      )
      counts[env_id][arm_index] += 1
      if min(counts[env_id]) >= args.episodes_per_env:
        active[env_id] = False
      else:
        current_policy[env_id] = not current_policy[env_id]
        pairs[env_id] = counts[env_id][int(current_policy[env_id])]
    disturbances.clear(done)
    _reset_done(env, done)
  residual_term.close()
  env.close()
  return episodes, strata


def _episode_value(episode: Episode, name: str) -> float:
  """Derive one qualification quantity from an episode record."""
  if name == "hazard":
    return float(
      any(
        episode.terminations.get(term, 0)
        for term in ("fell_over", "collapsed", "controller_failed")
      )
    )
  if name == "pre_disturbance_hazard":
    first_disturbance_step = round(DISTURBANCE_WARMUP_S / POLICY_STEP_S)
    return float(
      episode.scenario != "nominal"
      and episode.length <= first_disturbance_step
      and _episode_value(episode, "hazard")
    )
  if name == "worker_failure":
    return float(episode.terminations.get("controller_worker_failed", 0))
  if name == "recovery_dcm_error":
    active = episode.metrics.get("recovery_active", 0.0)
    return episode.metrics.get(name, 0.0) / active if active > 0.0 else float("nan")
  if name == "gate_duty":
    return episode.metrics["gate_mean"]
  if name == "residual_rms":
    return math.sqrt(max(0.0, episode.metrics["executed_residual_l2"]))
  return episode.metrics[name]


#: Two-sided 95% Student-t critical values for 1 to 30 degrees of freedom.
_T_CRITICAL_95 = (
  12.7062,
  4.3027,
  3.1824,
  2.7764,
  2.5706,
  2.4469,
  2.3646,
  2.3060,
  2.2622,
  2.2281,
  2.2010,
  2.1788,
  2.1604,
  2.1448,
  2.1314,
  2.1199,
  2.1098,
  2.1009,
  2.0930,
  2.0860,
  2.0796,
  2.0739,
  2.0687,
  2.0639,
  2.0595,
  2.0555,
  2.0518,
  2.0484,
  2.0452,
  2.0423,
)
_NORMAL_CRITICAL_95 = 1.959963985


def _t_critical(clusters: int) -> float:
  """Two-sided 95% critical value for a cluster count, not the normal limit."""
  df = clusters - 1
  if df < 1:
    return float("inf")
  if df <= len(_T_CRITICAL_95):
    return _T_CRITICAL_95[df - 1]
  z = _NORMAL_CRITICAL_95
  return z + (z**3 + z) / (4.0 * df)


def clusters_for_confidence(
  mean: float, sem: float, clusters: int, cap: int = 4096
) -> float:
  """Independent clusters at which the observed effect would exclude zero."""
  if not (mean < 0.0 and math.isfinite(sem) and sem > 0.0 and clusters >= 2):
    return float("nan")
  for count in range(clusters, cap + 1):
    if _t_critical(count) * sem * math.sqrt(clusters / count) < -mean:
      return float(count)
  return float("inf")


def _cluster_stats(values: list[float]) -> dict[str, float]:
  """Mean, cluster standard error, Student-t interval, and normal p-value."""
  values = [value for value in values if math.isfinite(value)]
  if not values:
    keys = ("mean", "sem", "ci_low", "ci_high", "p", "clusters")
    return {key: float("nan") for key in keys}
  mean = sum(values) / len(values)
  if len(values) == 1:
    # Infinite, not NaN: a NaN bound made every `>= 0` gate read False.
    sem = float("inf")
  else:
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sem = math.sqrt(variance / len(values))
  radius = _t_critical(len(values)) * sem
  p = math.erfc(abs(mean / sem) / math.sqrt(2.0)) if sem > 0.0 else float("nan")
  return {
    "mean": mean,
    "sem": sem,
    "ci_low": mean - radius,
    "ci_high": mean + radius,
    "p": p,
    "clusters": float(len(values)),
  }


def summarize(episodes: list[Episode]) -> dict:
  """Compute paired, environment-clustered scenario summaries."""
  names = (
    "hazard",
    "pre_disturbance_hazard",
    "worker_failure",
    "dcm_error",
    "recovery_dcm_error",
    "com_velocity_error",
    "zmp_error",
    "foot_slip",
    "projection_fraction",
    "near_bound_fraction",
    "max_effort_ratio",
    "gate_duty",
    "residual_rms",
  )
  grouped: dict[tuple[int, int, int, str], Episode] = {
    (episode.seed, episode.env_id, episode.pair, episode.arm): episode
    for episode in episodes
  }
  summary: dict[str, dict] = {}
  for name in names:
    arm_values = {"baseline": [], "policy": []}
    env_differences: dict[tuple[int, int], list[float]] = {}
    for seed, env_id, pair, arm in grouped:
      if arm != "baseline":
        continue
      baseline = grouped[(seed, env_id, pair, "baseline")]
      policy = grouped.get((seed, env_id, pair, "policy"))
      if policy is None:
        continue
      base_value = _episode_value(baseline, name)
      policy_value = _episode_value(policy, name)
      if not math.isfinite(base_value) or not math.isfinite(policy_value):
        continue
      arm_values["baseline"].append(base_value)
      arm_values["policy"].append(policy_value)
      env_differences.setdefault((seed, env_id), []).append(policy_value - base_value)
    env_clusters = {
      key: sum(values) / len(values) for key, values in env_differences.items()
    }
    # A seed-mean collapse made a second seed shrink 32 clusters to 2, which is
    # fewer than the interval needs. docs/evaluation.md#clusters_for_confidence
    clusters = list(env_clusters.values())
    cluster_level = "seed-environment"
    paired = _cluster_stats(clusters)
    base_mean = (
      sum(arm_values["baseline"]) / len(arm_values["baseline"])
      if arm_values["baseline"]
      else float("nan")
    )
    policy_mean = (
      sum(arm_values["policy"]) / len(arm_values["policy"])
      if arm_values["policy"]
      else float("nan")
    )
    summary[name] = {
      "baseline": base_mean,
      "policy": policy_mean,
      "paired": paired,
      "relative": paired["mean"] / base_mean if base_mean else float("nan"),
      "clusters": len(clusters),
      "cluster_level": cluster_level,
    }
  _holm_adjust(summary)
  return summary


def _paired_stratum_value(
  records: list[StratumRecord], value
) -> dict[str, float | int | str | dict]:
  """Summarize one stratum value with environment- or seed-clustered pairing."""
  grouped = {
    (record.seed, record.env_id, record.pair, record.arm): record for record in records
  }
  arm_values: dict[str, list[float]] = {"baseline": [], "policy": []}
  env_differences: dict[tuple[int, int], list[float]] = {}
  for seed, env_id, pair, arm in grouped:
    if arm != "baseline":
      continue
    baseline = grouped[(seed, env_id, pair, "baseline")]
    policy = grouped.get((seed, env_id, pair, "policy"))
    if policy is None:
      continue
    base_value = float(value(baseline))
    policy_value = float(value(policy))
    if not math.isfinite(base_value) or not math.isfinite(policy_value):
      continue
    arm_values["baseline"].append(base_value)
    arm_values["policy"].append(policy_value)
    env_differences.setdefault((seed, env_id), []).append(policy_value - base_value)
  env_clusters = {
    key: sum(values) / len(values) for key, values in env_differences.items()
  }
  seed_clusters: dict[int, list[float]] = {}
  for (seed, _), difference in env_clusters.items():
    seed_clusters.setdefault(seed, []).append(difference)
  if len(seed_clusters) > 1:
    clusters = [sum(values) / len(values) for values in seed_clusters.values()]
    cluster_level = "seed"
  else:
    clusters = list(env_clusters.values())
    cluster_level = "environment"
  paired = _cluster_stats(clusters)
  baseline = (
    sum(arm_values["baseline"]) / len(arm_values["baseline"])
    if arm_values["baseline"]
    else float("nan")
  )
  policy = (
    sum(arm_values["policy"]) / len(arm_values["policy"])
    if arm_values["policy"]
    else float("nan")
  )
  return {
    "baseline": baseline,
    "policy": policy,
    "paired": paired,
    "relative": paired["mean"] / baseline if baseline else float("nan"),
    "clusters": len(clusters),
    "cluster_level": cluster_level,
  }


def summarize_strata(records: list[StratumRecord]) -> dict[str, dict]:
  """Compute paired scalar, termination, and per-joint summaries by stratum."""
  grouped: dict[tuple[str, str, str], list[StratumRecord]] = {}
  for record in records:
    grouped.setdefault((record.regime, record.axis, record.direction), []).append(
      record
    )
  output = {}
  for (regime, axis, direction), group in grouped.items():
    metrics = {
      "duration_s": _paired_stratum_value(group, lambda record: record.duration_s),
      **{
        name: _paired_stratum_value(
          group, lambda record, metric=name: record.metrics[metric]
        )
        for name in sorted({name for record in group for name in record.metrics})
      },
    }
    terminations = {
      name: _paired_stratum_value(
        group, lambda record, term=name: record.terminations.get(term, 0)
      )
      for name in sorted({name for record in group for name in record.terminations})
    }
    _holm_adjust(metrics)
    _holm_adjust(terminations)
    joints = {}
    joint_names = sorted({name for record in group for name in record.joints})
    for joint_name in joint_names:
      fields = sorted(
        {name for record in group for name in record.joints.get(joint_name, {})}
      )
      joint = {
        name: _paired_stratum_value(
          group,
          lambda record, field=name, joint=joint_name: record.joints[joint][field],
        )
        for name in fields
      }
      _holm_adjust(joint)
      joints[joint_name] = joint
    output[f"{regime}/{direction}"] = {
      "regime": regime,
      "axis": axis,
      "direction": direction,
      "episodes": len({(r.seed, r.env_id, r.pair, r.arm) for r in group}),
      "metrics": metrics,
      "terminations": terminations,
      "joints": joints,
    }
  return output


def _holm_adjust(summary: dict[str, dict]) -> None:
  """Attach Holm-adjusted p-values across the scenario's reported comparisons."""
  finite = sorted(
    (
      (values["paired"]["p"], name)
      for name, values in summary.items()
      if math.isfinite(values["paired"]["p"])
    )
  )
  running = 0.0
  total = len(finite)
  for rank, (p_value, name) in enumerate(finite):
    running = max(running, min(1.0, (total - rank) * p_value))
    summary[name]["paired"]["p_holm"] = running


def promotion(checkpoint_summaries: dict[str, dict]) -> dict:
  """Apply safety/nominal gates before lexicographic recovery ranking."""
  reasons: list[str] = []
  invalid: set[str] = set()
  for scenario, summary in checkpoint_summaries.items():
    if summary["worker_failure"]["baseline"] or summary["worker_failure"]["policy"]:
      reasons.append(f"{scenario}: controller worker failure invalidated the run")
      invalid.add(scenario)
    if summary["max_effort_ratio"]["policy"] > 1.0001:
      reasons.append(f"{scenario}: hard effort ratio exceeds 1")
    if summary["projection_fraction"]["policy"] >= 0.001:
      reasons.append(f"{scenario}: projection is at least 0.1%")
    if summary["near_bound_fraction"]["policy"] >= 0.01:
      reasons.append(f"{scenario}: near-bound activity is at least 1%")
  nominal = checkpoint_summaries.get("nominal")
  if nominal is not None:
    if nominal["gate_duty"]["policy"] > 0.05:
      reasons.append("nominal: authority duty exceeds 5%")
    base = nominal["com_velocity_error"]["baseline"]
    upper = nominal["com_velocity_error"]["paired"]["ci_high"]
    if base and upper / base >= 0.05:
      reasons.append("nominal: CoM velocity upper-CI regression is at least 5%")
    for name in ("zmp_error", "foot_slip"):
      values = nominal[name]
      base = values["baseline"]
      paired = values["paired"]
      if not base:
        continue
      sampling_floor = 2.0 * paired["sem"] / base
      limit = max(0.05, sampling_floor) if math.isfinite(sampling_floor) else 0.05
      if paired["ci_high"] / base >= limit:
        reasons.append(f"nominal: {name} upper-CI regression exceeds its gate")
  recovery = checkpoint_summaries.get("finite_impulse")
  # A gate read off an invalidated scenario is not a verdict about the policy.
  if "finite_impulse" in invalid:
    recovery = None
    reasons.append("finite impulse: recovery DCM unreadable, scenario invalidated")
  if recovery is not None:
    base = recovery["recovery_dcm_error"]["baseline"]
    paired = recovery["recovery_dcm_error"]["paired"]
    if base and -paired["mean"] / base < 0.05:
      reasons.append("finite impulse: recovery DCM improvement is below 5%")
    elif base and not paired["ci_high"] < 0.0:
      needed = clusters_for_confidence(
        paired["mean"], paired["sem"], int(paired["clusters"])
      )
      reasons.append(
        "finite impulse: recovery DCM improvement is unresolved by "
        f"{paired['clusters']:.0f} clusters; {needed:.0f} would resolve it"
      )
  robust = checkpoint_summaries.get("robust")
  if robust is not None and robust["pre_disturbance_hazard"]["baseline"] > 0.05:
    reasons.append("robust: baseline pre-disturbance hazard exceeds 5%")
  scored = {
    scenario: summary
    for scenario, summary in checkpoint_summaries.items()
    if scenario not in invalid
  }
  hazards = [summary["hazard"] for summary in scored.values()]
  total_base = sum(item["baseline"] for item in hazards)
  total_policy = sum(item["policy"] for item in hazards)
  hazard_ratio = (
    total_policy / total_base if total_base else (1.0 if not total_policy else math.inf)
  )
  if scored and hazard_ratio > 0.90:
    suffix = f" over {len(scored)}/{len(checkpoint_summaries)} valid scenarios"
    reasons.append(f"overall hazard ratio exceeds 0.90{suffix if invalid else ''}")
  for scenario, summary in scored.items():
    base = summary["hazard"]["baseline"]
    policy = summary["hazard"]["policy"]
    ratio = policy / base if base else (1.0 if not policy else math.inf)
    if ratio > 1.10:
      reasons.append(f"{scenario}: hazard ratio exceeds 1.10")
  recovery_gain = (
    -recovery["recovery_dcm_error"]["relative"] if recovery is not None else -math.inf
  )
  residual = sum(
    summary["residual_rms"]["policy"] for summary in checkpoint_summaries.values()
  ) / max(1, len(checkpoint_summaries))
  return {
    "eligible": not reasons,
    "reasons": reasons,
    "rank": [recovery_gain, -hazard_ratio, -residual],
    "hazard_ratio": hazard_ratio,
    "invalidated_scenarios": sorted(invalid),
  }


def write_outputs(
  episodes: list[Episode],
  strata: list[StratumRecord],
  report: dict,
  out_dir: Path,
) -> None:
  """Write raw paired episodes to CSV and complete summaries to JSON."""
  out_dir.mkdir(parents=True, exist_ok=True)
  term_names = sorted({name for episode in episodes for name in episode.terminations})
  reward_names = sorted({name for episode in episodes for name in episode.rewards})
  metric_names = sorted({name for episode in episodes for name in episode.metrics})
  with (out_dir / "qualification.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      ["checkpoint", "scenario", "seed", "env", "pair", "arm", "length"]
      + [f"termination/{name}" for name in term_names]
      + [f"reward/{name}" for name in reward_names]
      + [f"metric/{name}" for name in metric_names]
    )
    for episode in episodes:
      writer.writerow(
        [
          episode.checkpoint,
          episode.scenario,
          episode.seed,
          episode.env_id,
          episode.pair,
          episode.arm,
          episode.length,
        ]
        + [episode.terminations.get(name, 0) for name in term_names]
        + [episode.rewards.get(name, float("nan")) for name in reward_names]
        + [episode.metrics.get(name, float("nan")) for name in metric_names]
      )
  stratum_term_names = sorted(
    {name for record in strata for name in record.terminations}
  )
  stratum_metric_names = sorted({name for record in strata for name in record.metrics})
  identity = [
    "checkpoint",
    "scenario",
    "seed",
    "env",
    "pair",
    "arm",
    "regime",
    "axis",
    "direction",
    "steps",
    "duration_s",
  ]
  with (out_dir / "qualification_strata.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      identity
      + [f"termination/{name}" for name in stratum_term_names]
      + [f"metric/{name}" for name in stratum_metric_names]
    )
    for record in strata:
      writer.writerow(
        [
          record.checkpoint,
          record.scenario,
          record.seed,
          record.env_id,
          record.pair,
          record.arm,
          record.regime,
          record.axis,
          record.direction,
          record.steps,
          record.duration_s,
        ]
        + [record.terminations.get(name, 0) for name in stratum_term_names]
        + [record.metrics.get(name, float("nan")) for name in stratum_metric_names]
      )
  joint_fields = sorted(
    {name for record in strata for joint in record.joints.values() for name in joint}
  )
  with (out_dir / "qualification_joints.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(identity + ["joint"] + joint_fields)
    for record in strata:
      prefix = [
        record.checkpoint,
        record.scenario,
        record.seed,
        record.env_id,
        record.pair,
        record.arm,
        record.regime,
        record.axis,
        record.direction,
        record.steps,
        record.duration_s,
      ]
      for joint_name, values in record.joints.items():
        writer.writerow(
          prefix
          + [joint_name]
          + [values.get(name, float("nan")) for name in joint_fields]
        )
  (out_dir / "qualification.json").write_text(
    json.dumps(report, indent=2, allow_nan=True) + "\n"
  )


def main() -> None:
  """Qualify checkpoint inputs and select only among candidates passing gates."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("checkpoints", nargs="+", help="checkpoint paths, dirs, or globs")
  parser.add_argument("--scenario", action="append", choices=SCENARIOS)
  parser.add_argument("--episodes-per-env", type=int, default=2)
  # Clusters are seeds x environments; 8 could not resolve the effect the
  # 2026-08-25 screen measured. docs/evaluation.md#clusters_for_confidence
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--num-workers", type=int, default=6)
  parser.add_argument("--episode-length-s", type=float, default=90.0)
  parser.add_argument("--seed", type=int, action="append")
  parser.add_argument(
    "--achievement-stage", type=int, choices=range(len(ACHIEVEMENT_STAGES))
  )
  parser.add_argument("--control", choices=("position", "torque"), default="position")
  parser.add_argument(
    "--authority-set",
    choices=("uniform", "ankle", "sagittal", "hardware"),
    default=None,
  )
  parser.add_argument(
    "--controller-history", type=int, choices=(1, 5, 10, 20), default=20
  )
  parser.add_argument("--proprio-history", type=int, choices=(1, 5), default=5)
  parser.add_argument("--recurrent", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--out-dir", type=Path, default=Path("logs/qualification"))
  args = parser.parse_args()
  args.authority_set = args.authority_set or (
    "ankle" if args.achievement_stage is not None else "uniform"
  )
  checkpoints = resolve_checkpoints(args.checkpoints)
  scenarios = args.scenario or list(SCENARIOS)
  seeds = args.seed or ([42, 43] if args.achievement_stage is not None else [42])
  if args.achievement_stage is not None:
    if len(checkpoints) != 1:
      parser.error("achievement qualification accepts exactly one checkpoint")
    if set(scenarios) != set(REQUIRED_QUALIFICATION_SCENARIOS):
      parser.error("achievement qualification requires every scenario")
    if len(set(seeds)) < MINIMUM_QUALIFICATION_SEEDS:
      parser.error(
        f"achievement qualification requires {MINIMUM_QUALIFICATION_SEEDS} seeds"
      )
  episodes: list[Episode] = []
  strata: list[StratumRecord] = []
  summaries: dict[str, dict] = {}
  for checkpoint in checkpoints:
    print(f"[qualify] {checkpoint}", flush=True)
    by_scenario: dict[str, dict] = {}
    by_stratum: dict[str, dict] = {}
    for scenario in scenarios:
      print(f"[qualify]   {scenario}", flush=True)
      result: list[Episode] = []
      scenario_strata: list[StratumRecord] = []
      for seed in seeds:
        print(f"[qualify]     seed {seed}", flush=True)
        seed_episodes, seed_strata = run_checkpoint(checkpoint, scenario, seed, args)
        result.extend(seed_episodes)
        scenario_strata.extend(seed_strata)
      episodes.extend(result)
      strata.extend(scenario_strata)
      by_scenario[scenario] = summarize(result)
      by_stratum[scenario] = summarize_strata(scenario_strata)
    gate = promotion(by_scenario)
    summaries[str(checkpoint)] = {
      "scenarios": by_scenario,
      "stratified": by_stratum,
      "promotion": gate,
    }
    print(
      f"[qualify]   {'PASS' if gate['eligible'] else 'FAIL'}: "
      + ("all gates" if gate["eligible"] else "; ".join(gate["reasons"])),
      flush=True,
    )
  eligible = [
    (values["promotion"]["rank"], checkpoint)
    for checkpoint, values in summaries.items()
    if values["promotion"]["eligible"]
  ]
  selected = max(eligible)[1] if eligible else None
  report = {
    "config": {
      "seeds": seeds,
      "scenarios": scenarios,
      "episodes_per_env": args.episodes_per_env,
      "num_envs": args.num_envs,
      "episode_length_s": args.episode_length_s,
      "authority_set": args.authority_set,
      "controller_history": args.controller_history,
      "proprio_history": args.proprio_history,
      "recurrent": args.recurrent,
      "achievement": (
        {
          "stage": args.achievement_stage,
          "contract": achievement_contract(args.achievement_stage),
          "contract_sha256": achievement_contract_sha256(args.achievement_stage),
        }
        if args.achievement_stage is not None
        else None
      ),
      "strata": {
        "startup": f"episode age <= {DISTURBANCE_WARMUP_S:g} s",
        "recovery": "reward-configured post-disturbance window",
        "sustained": "all remaining exposure",
        "direction_frame": "controller base frame",
      },
    },
    "checkpoints": summaries,
    "selected": selected,
    "selection_rule": "safety and nominal gates, then recovery/hazard/residual",
  }
  write_outputs(episodes, strata, report, args.out_dir)
  print(f"[qualify] selected: {selected or 'none'}")
  print(
    f"[qualify] outputs: {args.out_dir / 'qualification.csv'}, "
    "qualification_strata.csv, qualification_joints.csv, and qualification.json"
  )


if __name__ == "__main__":
  main()
