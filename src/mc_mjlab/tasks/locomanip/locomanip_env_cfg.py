"""Play-only HRP5P reach, grasp, push and release scene."""

from __future__ import annotations

import math
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.termination_manager import TerminationTermCfg

from mc_mjlab import MC_RTC_CONFIG_PATH, mdp
from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionCfg
from mc_mjlab.tasks.locomanip.cart import (
  cart_cfg,
  cart_floor_contact,
  hand_cart_contact_sensors,
)
from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import _make_env_cfg

MC_RTC_YAML = MC_RTC_CONFIG_PATH / "mc_rtc_hrp5_locomanip_patched.yaml"


def locomanip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build one zero-residual position-controlled cart-pushing environment."""
  cfg = _make_env_cfg(
    "position",
    num_envs=1,
    num_workers=1,
    mc_rtc_yaml=MC_RTC_YAML,
    console_output="single" if play else "none",
  )
  cfg.scene.entities["cart"] = cart_cfg()
  cfg.scene.spec_fn = cart_floor_contact
  cfg.scene.sensors = hand_cart_contact_sensors()
  cfg.scene.env_spacing = 5.0
  cfg.episode_length_s = 60.0
  cfg.sim.mujoco.jacobian = "sparse"
  action = cast(McRtcResidualActionCfg, cfg.actions["robot_joints"])
  action.controller_objects = {"obj": "cart"}
  action.datastore_vectors_outputs = (
    "Locomanip::objectReferencePosition",
    "Locomanip::objectReferenceRpy",
  )
  action.datastore_scalar_outputs = (
    "Locomanip::leftPhase",
    "Locomanip::rightPhase",
    "Locomanip::complete",
  )
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": math.pi / 3},
    ),
    "collapsed": TerminationTermCfg(
      func=env_mdp.root_height_below_minimum,
      params={"minimum_height": 0.55},
    ),
    "controller_failed": TerminationTermCfg(
      func=mdp.terminations.controller_failed,
      params={"action_name": "robot_joints"},
    ),
    "controller_worker_failed": TerminationTermCfg(
      func=mdp.terminations.controller_worker_failed,
      params={"action_name": "robot_joints"},
      time_out=True,
    ),
  }
  return cfg
