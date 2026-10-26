"""Separate-policy matched rollouts with completed warm-up and terminal capture."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from mjlab.managers.event_manager import (
  EventTermCfg,
  RecomputeLevel,
  requires_model_fields,
)
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from evaluation.locomanip_inputs import (
  assert_payload_masses,
  draw_encoder_bias,
  fixed_payloads,
)
from evaluation.rollout import managed_env, reset_done
from evaluation.study_trace import StudyTrace
from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionCfg
from mc_mjlab.rl.controller_provenance import collect_controller_provenance
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.mdp import accessors


def run_study_conditions(
  task: str,
  rows: list[dict],
  checkpoint: Path | None = None,
  workers: int = 4,
  device: str = "cuda:0",
  trace_path: Path | None = None,
) -> dict:
  """Evaluate one arm, applying no policy during a recycled row's warm-up."""
  if not rows or workers < 1:
    raise ValueError("nonempty rows and positive workers required")
  cfg = load_env_cfg(task)
  cfg.seed = rows[0]["seed"]
  cfg.scene.num_envs = len(rows)
  cfg.auto_reset = False
  cfg.observations["actor"].enable_corruption = False
  action_cfg = cast(McRtcResidualActionCfg, cfg.actions[accessors.ACTION_NAME])
  action_cfg.num_workers = min(workers, len(rows))
  action_cfg.console_output = "none"
  payload = cfg.events["cart_payload"]
  cfg.events.pop("encoder_bias", None)
  cfg.events["cart_payload"] = EventTermCfg(
    func=study_inputs,
    mode="reset",
    params={"rows": rows, "asset_cfg": payload.params["asset_cfg"]},
  )
  with managed_env(cfg, device) as env:
    wrapped = RslRlVecEnvWrapper(env)
    policy = None
    manifest = None
    if checkpoint is not None:
      runner_cls = load_runner_cls(task)
      if runner_cls is None:
        raise ValueError("study requires the provenance-checking runner")
      agent = asdict(load_rl_cfg(task))
      agent["seed"] = cfg.seed
      runner = runner_cls(wrapped, agent, device=device)
      runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
      )
      policy = runner.get_inference_policy(device=device)
      manifest = runner._effective_manifest
    env.reset()
    trace = StudyTrace(env, rows, trace_path)
    try:
      _rollout(env, wrapped, policy, trace, rows)
    finally:
      trace.close()
    return {
      "task": task,
      "checkpoint": str(checkpoint) if checkpoint else None,
      "controller_provenance": collect_controller_provenance(wrapped),
      "effective_manifest": manifest,
      "rows": trace.results,
    }


@requires_model_fields(
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def study_inputs(
  env: Any, env_ids: torch.Tensor | None, rows: list[dict], asset_cfg: Any
) -> None:
  """Apply exact payload and encoder draws only to resetting rows."""
  if not hasattr(env, "_study_active"):
    env._study_active = torch.tensor(
      [r["lifecycle"] == "fresh" for r in rows], device=env.device
    )
  active = env._study_active.cpu().tolist()
  payload_rows = [
    {**r, "mass_kg": r["mass_kg"] if active[i] else 600.0} for i, r in enumerate(rows)
  ]
  fixed_payloads(env, env_ids, payload_rows, cart_nominal_mass_kg(), asset_cfg)
  ids = list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
  bias = env.scene["robot"].data.encoder_bias
  for i in ids:
    bias[i] = draw_encoder_bias(
      rows[i]["input_id"], bias.shape[1], active[i] and not rows[i]["nominal"]
    ).to(env.device)


def _rollout(
  env: Any, wrapped: Any, policy: Any, trace: StudyTrace, rows: list[dict]
) -> None:
  zeros = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  for _ in range(2 * math.ceil(env.cfg.episode_length_s / env.step_dt) + 2):
    action = zeros.clone()
    live = ~trace.finished
    active = env._study_active & live
    if policy is not None:
      with torch.inference_mode():
        proposed = policy(wrapped.get_observations())
      action[active] = proposed[active]
    _, _, terminated, truncated, _ = env.step(action)
    frame = trace.update(active)
    done = terminated | truncated
    trace.finish((done & active).nonzero().flatten(), frame)
    warming = done & ~env._study_active & live
    ready = []
    for i in warming.nonzero().flatten().tolist():
      flags = frame["terminations"][i]
      if flags.get("task_complete", False) and not any(
        flags.get(n, False)
        for n in (
          *accessors.FALL_TERMINATIONS,
          "controller_failed",
          "controller_worker_failed",
        )
      ):
        ready.append(i)
      else:
        trace.finish(torch.tensor([i], device=env.device), frame, warmup_failed=True)
    if trace.finished.all():
      return
    if done.any():
      env._study_active[ready] = True
      if policy is not None:
        policy.reset(done)
      reset_done(env, done.nonzero().flatten())
      if ready:
        ids = torch.tensor(ready, device=env.device)
        trace.begin(ids)
        assert_payload_masses(
          accessors.object_mass_kg(env)[ids], [rows[i] for i in ready]
        )
  trace.finish((~trace.finished).nonzero().flatten(), frame, observed=False)
