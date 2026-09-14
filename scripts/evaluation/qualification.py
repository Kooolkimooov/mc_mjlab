"""Paired-cluster inference and checkpoint qualification decisions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

from evaluation.qualification_strata import (
  StratumRecord,
)
from evaluation.records import QualificationEpisode
from evaluation.scenarios import DISTURBANCE_WARMUP_S, POLICY_STEP_S

#: PROVISIONAL, not yet measured. docs/evaluation.md#grounded_fraction_margin
GROUNDED_FRACTION_MARGIN = 0.01


def _episode_value(episode: QualificationEpisode, name: str) -> float:
  """Derive one qualification quantity from an episode record."""
  if name == "hazard":
    return float(
      any(
        episode.terminations[term]
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
    return float(episode.terminations["controller_worker_failed"])
  if name == "recovery_dcm_error":
    # Strict: a default here scored an absent metric as a perfect recovery.
    active = episode.metrics["recovery_active"]
    return episode.metrics[name] / active if active > 0.0 else float("nan")
  if name == "grounded_fraction":
    return episode.metrics["zmp_grounded"]
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


def t_critical(clusters: int) -> float:
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
    if t_critical(count) * sem * math.sqrt(clusters / count) < -mean:
      return float(count)
  return float("inf")


def cluster_stats(values: list[float]) -> dict[str, float]:
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
  radius = t_critical(len(values)) * sem
  p = math.erfc(abs(mean / sem) / math.sqrt(2.0)) if sem > 0.0 else float("nan")
  return {
    "mean": mean,
    "sem": sem,
    "ci_low": mean - radius,
    "ci_high": mean + radius,
    "p": p,
    "clusters": float(len(values)),
  }


def summarize(episodes: list[QualificationEpisode]) -> dict:
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
    "grounded_fraction",
    "residual_rms",
  )
  grouped: dict[tuple[int, int, int, str], QualificationEpisode] = {
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
    paired = cluster_stats(clusters)

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

  holm_adjust(summary)
  return summary


def _paired_stratum_value(
  records: list[StratumRecord], value: Callable[[StratumRecord], float]
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
  # Same unit as `summarize`: a second seed must add clusters, not collapse to 2.
  clusters = list(env_clusters.values())
  cluster_level = "seed-environment"
  paired = cluster_stats(clusters)
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

    holm_adjust(metrics)
    holm_adjust(terminations)

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
      holm_adjust(joint)
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


def holm_adjust(summary: dict[str, dict]) -> None:
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


@dataclass
class Criterion:
  """One promotion gate's verdict: passed, failed, or never measured."""

  name: str
  scenario: str
  verdict: str
  detail: str
  value: float = float("nan")
  limit: float = float("nan")


def promotion(checkpoint_summaries: dict[str, dict]) -> dict:
  """Apply safety/nominal gates before lexicographic recovery ranking."""
  criteria: list[Criterion] = []
  invalid: set[str] = set()
  for scenario, summary in checkpoint_summaries.items():
    failures = [summary["worker_failure"][arm] for arm in ("baseline", "policy")]
    if not _finite(*failures):
      criteria.append(
        Criterion(
          "worker_failure",
          scenario,
          "not_measured",
          f"{scenario}: controller worker failure was not measured",
        )
      )
      invalid.add(scenario)
    elif any(value > 0.0 for value in failures):
      criteria.append(
        Criterion(
          "worker_failure",
          scenario,
          "fail",
          f"{scenario}: controller worker failure invalidated the run",
        )
      )
      invalid.add(scenario)

    effort = summary["max_effort_ratio"]["policy"]
    criteria.append(
      _gate(
        "max_effort_ratio",
        scenario,
        effort,
        1.0001,
        effort > 1.0001,
        f"{scenario}: hard effort ratio exceeds 1",
      )
    )
    projection = summary["projection_fraction"]["policy"]
    criteria.append(
      _gate(
        "projection_fraction",
        scenario,
        projection,
        0.001,
        projection >= 0.001,
        f"{scenario}: projection is at least 0.1%",
      )
    )
    near_bound = summary["near_bound_fraction"]["policy"]
    criteria.append(
      _gate(
        "near_bound_fraction",
        scenario,
        near_bound,
        0.01,
        near_bound >= 0.01,
        f"{scenario}: near-bound activity is at least 1%",
      )
    )

    # The escape from every contact-gated term is to unload the feet, so the
    # policy may not spend less time grounded than the baseline arm does.
    grounded = summary["grounded_fraction"]
    shortfall = -grounded["paired"]["ci_high"]
    criteria.append(
      _gate(
        "grounded_fraction",
        scenario,
        shortfall,
        GROUNDED_FRACTION_MARGIN,
        shortfall > GROUNDED_FRACTION_MARGIN,
        f"{scenario}: grounded fraction falls below the baseline arm",
      )
    )

  nominal = checkpoint_summaries.get("nominal")
  if nominal is not None:
    duty = nominal["gate_duty"]["policy"]
    criteria.append(
      _gate(
        "gate_duty",
        "nominal",
        duty,
        0.05,
        duty > 0.05,
        "nominal: authority duty exceeds 5%",
      )
    )

    base = nominal["com_velocity_error"]["baseline"]
    upper = nominal["com_velocity_error"]["paired"]["ci_high"]
    ratio = upper / base if _finite(base, upper) and base else float("nan")
    criteria.append(
      _gate(
        "com_velocity_error",
        "nominal",
        ratio,
        0.05,
        ratio >= 0.05,
        "nominal: CoM velocity upper-CI regression is at least 5%",
      )
    )
    for name in ("zmp_error", "foot_slip"):
      values = nominal[name]
      base = values["baseline"]
      paired = values["paired"]
      ratio = (
        paired["ci_high"] / base
        if _finite(base, paired["ci_high"]) and base
        else float("nan")
      )
      sampling_floor = (
        2.0 * paired["sem"] / base if _finite(base) and base else math.nan
      )
      limit = max(0.05, sampling_floor) if math.isfinite(sampling_floor) else 0.05
      criteria.append(
        _gate(
          name,
          "nominal",
          ratio,
          limit,
          ratio >= limit,
          f"nominal: {name} upper-CI regression exceeds its gate",
        )
      )

  recovery = checkpoint_summaries.get("finite_impulse")
  # A gate read off an invalidated scenario is not a verdict about the policy.
  if "finite_impulse" in invalid:
    recovery = None
    criteria.append(
      Criterion(
        "recovery_dcm_error",
        "finite_impulse",
        "not_measured",
        "finite impulse: recovery DCM unreadable, scenario invalidated",
      )
    )
  if recovery is not None:
    base = recovery["recovery_dcm_error"]["baseline"]
    paired = recovery["recovery_dcm_error"]["paired"]
    gain = (
      -paired["mean"] / base if _finite(base, paired["mean"]) and base else math.nan
    )
    if not _finite(gain):
      criteria.append(
        Criterion(
          "recovery_dcm_error",
          "finite_impulse",
          "not_measured",
          "finite impulse: recovery DCM improvement was not measured",
        )
      )
    elif gain < 0.05:
      criteria.append(
        Criterion(
          "recovery_dcm_error",
          "finite_impulse",
          "fail",
          "finite impulse: recovery DCM improvement is below 5%",
          gain,
          0.05,
        )
      )
    elif not paired["ci_high"] < 0.0:
      needed = clusters_for_confidence(
        paired["mean"], paired["sem"], int(paired["clusters"])
      )
      criteria.append(
        Criterion(
          "recovery_dcm_error",
          "finite_impulse",
          "fail",
          "finite impulse: recovery DCM improvement is unresolved by "
          f"{paired['clusters']:.0f} clusters; {needed:.0f} would resolve it",
          gain,
          0.05,
        )
      )
    else:
      criteria.append(
        Criterion(
          "recovery_dcm_error",
          "finite_impulse",
          "pass",
          "finite impulse: recovery DCM improvement resolved",
          gain,
          0.05,
        )
      )

  robust = checkpoint_summaries.get("robust")
  if robust is not None:
    hazard = robust["pre_disturbance_hazard"]["baseline"]
    criteria.append(
      _gate(
        "pre_disturbance_hazard",
        "robust",
        hazard,
        0.05,
        hazard > 0.05,
        "robust: baseline pre-disturbance hazard exceeds 5%",
      )
    )

  scored = {
    scenario: summary
    for scenario, summary in checkpoint_summaries.items()
    if scenario not in invalid
  }

  hazards = [summary["hazard"] for summary in scored.values()]
  total_base = sum(item["baseline"] for item in hazards)
  total_policy = sum(item["policy"] for item in hazards)
  if not _finite(total_base, total_policy):
    hazard_ratio = float("nan")
  elif total_base:
    hazard_ratio = total_policy / total_base
  else:
    hazard_ratio = 1.0 if total_policy == 0.0 else math.inf
  if scored:
    suffix = f" over {len(scored)}/{len(checkpoint_summaries)} valid scenarios"
    criteria.append(
      _gate(
        "hazard_ratio",
        "overall",
        hazard_ratio,
        0.90,
        hazard_ratio > 0.90,
        f"overall hazard ratio exceeds 0.90{suffix if invalid else ''}",
      )
    )
  for scenario, summary in scored.items():
    base = summary["hazard"]["baseline"]
    policy = summary["hazard"]["policy"]
    if not _finite(base, policy):
      ratio = float("nan")
    elif base:
      ratio = policy / base
    else:
      ratio = 1.0 if policy == 0.0 else math.inf
    criteria.append(
      _gate(
        "hazard_ratio",
        scenario,
        ratio,
        1.10,
        ratio > 1.10,
        f"{scenario}: hazard ratio exceeds 1.10",
      )
    )

  recovery_gain = (
    -recovery["recovery_dcm_error"]["relative"] if recovery is not None else -math.inf
  )
  residual = sum(
    summary["residual_rms"]["policy"] for summary in checkpoint_summaries.values()
  ) / max(1, len(checkpoint_summaries))

  not_measured = [c for c in criteria if c.verdict == "not_measured"]
  failed = [c for c in criteria if c.verdict == "fail"]
  return {
    # Unmeasured blocks promotion but is never reported as a failure.
    "eligible": not failed and not not_measured,
    "reasons": [c.detail for c in failed + not_measured],
    "criteria": [asdict(c) for c in criteria],
    "not_measured": [c.name for c in not_measured],
    "rank": [recovery_gain, -hazard_ratio, -residual],
    "hazard_ratio": hazard_ratio,
    "invalidated_scenarios": sorted(invalid),
  }


def _finite(*values: float) -> bool:
  """True when every input is a real measurement rather than a missing sample."""
  return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _gate(
  name: str,
  scenario: str,
  value: float,
  limit: float,
  failed: bool,
  detail: str,
) -> Criterion:
  """Record a numeric gate, reporting an unmeasurable input as ``not_measured``."""
  if not _finite(value, limit):
    return Criterion(
      name, scenario, "not_measured", f"{scenario}: {name} was not measured"
    )
  verdict = "fail" if failed else "pass"
  return Criterion(name, scenario, verdict, detail, float(value), float(limit))
