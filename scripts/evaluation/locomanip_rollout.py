"""Fixed-condition locomanip rollouts for tuning, authority and paired evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from evaluation.locomanip_authority import action_matrix
from evaluation.locomanip_inputs import (
  assert_payload_masses,
  fixed_payloads,
  matched_encoder_bias,
)
from evaluation.locomanip_schedules import ForceSchedule
from evaluation.locomanip_trace import DispatchTrace, SafetyTrace
from evaluation.rollout import managed_env, reset_done
from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointPositionActionCfg,
)
from mc_mjlab.rl.controller_provenance import collect_controller_provenance
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.mdp import accessors, metrics


def run_conditions(
  task: str,
  rows: list[dict[str, Any]],
  *,
  config: Path | None = None,
  checkpoint: Path | None = None,
  encoder_bias: bool = True,
  workers: int = 4,
  device: str = "cuda:0",
  trace_path: Path | None = None,
) -> dict[str, Any]:
  """Run exactly one first episode per row and retain incomplete observations."""
  if not rows or workers < 1:
    raise ValueError("nonempty conditions and positive workers are required")
  cfg = load_env_cfg(task)
  cfg.seed = rows[0]["seed"]
  cfg.scene.num_envs = len(rows)
  cfg.auto_reset = False
  action_cfg = cast(
    ResidualFeedbackJointPositionActionCfg, cfg.actions[accessors.ACTION_NAME]
  )
  action_cfg.num_workers = min(workers, len(rows))
  action_cfg.console_output = "none"
  if config is not None:
    action_cfg.mc_rtc_config_path = str(config.resolve())
  payload = cfg.events["cart_payload"]
  cfg.events["cart_payload"] = EventTermCfg(
    func=fixed_payloads,
    mode="reset",
    params={
      "rows": rows,
      "nominal_mass": cart_nominal_mass_kg(),
      "asset_cfg": payload.params["asset_cfg"],
    },
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    func=matched_encoder_bias,
    mode="startup",
    params={"rows": rows, "enabled": encoder_bias},
  )
  stream = None
  if trace_path is not None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    stream = trace_path.open("w")
  try:
    with managed_env(cfg, device) as env:
      missing = set(LOCOMANIP_EVALUATION.metrics) - set(
        env.metrics_manager.active_terms
      )
      if missing:
        raise ValueError(f"required diagnostic metrics missing: {sorted(missing)}")
      wrapped = RslRlVecEnvWrapper(env)
      policy = None
      if checkpoint is not None:
        runner_cls = load_runner_cls(task)
        if runner_cls is None:
          raise ValueError("evaluation requires the provenance-checking task runner")
        agent_cfg = asdict(load_rl_cfg(task))
        agent_cfg["seed"] = cfg.seed
        runner = runner_cls(wrapped, agent_cfg, device=device)
        runner.load(
          str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
        )
        policy = runner.get_inference_policy(device=device)
      env.reset()
      provenance = collect_controller_provenance(wrapped)
      biases = env.scene["robot"].data.encoder_bias.cpu().tolist()
      actual_masses = accessors.object_mass_kg(env)
      assert_payload_masses(actual_masses, rows)
      masses = actual_masses.cpu().tolist()
      base_actions = action_matrix(env, rows)
      schedules = ForceSchedule(
        [row.get("schedule", "constant") for row in rows], device
      )
      trace = SafetyTrace(env, rows)
      dispatch = DispatchTrace(env)
      try:
        _step_conditions(
          env, wrapped, policy, base_actions, schedules, trace, dispatch, stream
        )
      finally:
        dispatch.close()
      for index, row in enumerate(trace.results):
        if row is None:
          raise RuntimeError("a requested condition has no terminal record")
        triggered = row.get("trigger_time_s", float(schedules.trigger_time[index]))
        row.update(
          encoder_bias_rad=biases[index],
          actual_mass_kg=masses[index],
          trigger_time_s=triggered if triggered >= 0 else None,
          requested_duration_s=row.get(
            "requested_duration_s", float(schedules.duration[index])
          ),
          probe_status=(
            "control"
            if row["level"] == 0
            else "not_measured"
            if triggered < 0 or row["force_duration_s"] == 0
            else "measured"
          ),
          input_delivery_verified=row["force_duration_s"] > 0
          and row["delivery_error_n"] < 1e-5,
        )
      return {
        "task": task,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "encoder_bias": encoder_bias,
        "episode_length_s": cfg.episode_length_s,
        "controller_provenance": provenance,
        "rows": trace.results,
      }
  finally:
    if stream is not None:
      stream.close()


def _step_conditions(
  env: Any,
  wrapped: Any,
  policy: Any,
  base_actions: torch.Tensor,
  schedules: ForceSchedule,
  trace: SafetyTrace,
  dispatch: DispatchTrace,
  stream: Any,
) -> None:
  """Apply causal requests, capture the terminal step, then recycle finished rows."""
  policy_rows = torch.tensor(
    [row.get("arm") == "policy" for row in trace.rows], device=env.device
  )
  for _ in range(math.ceil(env.cfg.episode_length_s / env.step_dt) + 1):
    live = ~trace.finished
    active = schedules.update(
      env.episode_length_buf * env.step_dt,
      metrics.forward_cart_lag(env),
      accessors.holding(env).bool() & live,
      env.step_dt,
    )
    action = base_actions * active.unsqueeze(-1)
    if policy is not None:
      with torch.inference_mode():
        action[policy_rows] = policy(wrapped.get_observations())[policy_rows]
    action[trace.finished] = 0
    dispatch.clear()
    _, _, terminated, truncated, _ = env.step(action)
    frame = trace.update(dispatch)
    if stream is not None:
      for index in live.nonzero().flatten().tolist():
        stream.write(
          json.dumps(
            {
              "row": index,
              **trace.rows[index],
              **{key: values[index] for key, values in frame.items()},
            }
          )
          + "\n"
        )
    done = terminated | truncated
    new_done = (done & live).nonzero().flatten()
    trace.finish(new_done, frame)
    for index in new_done.tolist():
      result = trace.results[index]
      assert result is not None
      result["trigger_time_s"] = float(schedules.trigger_time[index])
      result["requested_duration_s"] = float(schedules.duration[index])
    if trace.finished.all():
      break
    if done.any():
      if policy is not None:
        policy.reset(done)
      reset_ids = done.nonzero().flatten()
      reset_done(env, reset_ids)
      schedules.reset(reset_ids)
  remaining = (~trace.finished).nonzero().flatten()
  if remaining.numel():
    trace.finish(remaining, frame, observed=False)
