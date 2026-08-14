"""Residual balance task: keep the mc_rtc controller walking under pushes.

The mc_rtc controller is the base policy and is *not* commanded by the RL side
-- this coupling feeds it state, not velocity references -- so the learnable
part is a residual that keeps reality close to the controller's plan when the
world disagrees with the controller's model: pushes, friction and CoM
randomization, reset noise.

Two things this task has to get right, both learned the hard way from a run
whose robot never walked at all:

*The env must hand mc_rtc a robot it recognizes.* Declaring an ``events`` dict
replaces mjlab's default, which is where ``reset_scene_to_default`` -- the only
term that resets joints -- lives. Without it every episode starts from the
model's qpos0 (legs straight) instead of the half-sitting stance, mc_rtc
initializes against that posture and its FSM never reaches the walking state.
The robot then stands still for the whole episode no matter what the residual
does, which reads exactly like a policy that has learned to freeze the gait.

*The reward must pay for tracking the controller's plan, not for surviving.*
With ``gamma=0.99`` the discounted horizon is ~100 steps (2 s), so the -200
termination penalty shapes only the last couple of seconds before a fall
(0.99^400 ~ 0.02) and the dense terms do all the work. Those are
``zmp_tracking`` and ``com_velocity_tracking``: the CoM-to-ZMP offset and the
CoM velocity, the two halves of the planar LIPM state the plan is written on.
There is deliberately no separate DCM term -- the divergent mode
``com + comVel/omega`` is a linear combination of those two, so any DCM
weighting is already reachable by choosing their weights.

Both were sized against the zero-residual baseline, and both collapse in the
run-up to a fall (to 0.12 and 0.10 of a possible 1.0), which is what makes them
the early warning the sparse penalty cannot be.

*Two shaping ideas that measurement rejected*, recorded so they are not
reinvented. Paying for the controller still generating a gait
(``mdp.controller_reference_motion``, tanh of the joint-velocity reference) is
anti-correlated with what we want: over 96 s x 16 envs the reference norm runs
1.78 rad/s in the second before a fall against 0.64 overall, because a falling
robot's controller thrashes -- the term would pay *more* for the run-up to a
fall. And a support-region margin on the *measured* ZMP is close to a
tautology, since a centre of pressure lies inside the contact hull by
construction; only the *commanded* ZMP can leave it.

Both tracking terms do score a standing robot slightly above a walking one
(0.75 vs 0.66 for the ZMP term), which is the wrong sign for this task. It
stays theoretical only because the residual is hard-clipped to an authority
that cannot cancel a swing trajectory -- revisit it if ``residual_scale`` grows.

Rates: the sim runs at 1 kHz and the controller at 500 Hz (``frameskip=2``, the
mc_mujoco pairing), while the policy acts at 50 Hz (``decimation=20``); the
residual is therefore held across 10 controller periods.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_torque_actions import (
  McRtcResidualJointTorqueActionCfg,
)
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc
from mc_mjlab.robots.robots_registry import (
  get_main_robot_spec,
  prepare_cfg_for_mc_rtc,
)
from mc_mjlab.tasks import mdp

# How hard the robot is shoved: the task's difficulty dial. Both directions
# ruin training -- too gentle and mc_rtc never falls, so the best residual is
# no residual; too hard and the robot falls whatever the residual does, which
# plateaus the policy at a fraction of an episode.
#
# 0.1 m/s already lost the zero-residual baseline about half its episodes
# (fell_over + collapsed vs time_out, over 65 s x 16 envs) -- a *walking* robot
# is far easier to topple than a standing one, which is why that reads timid
# next to mjlab's velocity task (+/-0.5). 0.4 is deliberately past that: the
# point of the dial is a baseline that *almost always* fails, so that the
# residual's contribution to the overall control is measurable as survival
# rather than hidden inside a controller that would have coped anyway.
#
# One caveat on the number itself. It was calibrated during the 2026-07-31 runs,
# whose working tree had ``"x": (push_velocity, push_velocity)`` -- a degenerate
# range, so every push was exactly +0.4 in x *and* +0.4 in y: a constant
# 0.566 m/s shove, same direction every time. Sampled symmetrically as below the
# same 0.4 is much gentler (mean |v| ~ 0.31 m/s, random direction, sometimes
# ~0), so the baseline failure rate that 0.4 was chosen for does not carry over.
# Re-measure it -- the number to keep an eye on is the baseline's survival rate,
# not the push magnitude, and the ceiling is where the robot falls whatever the
# residual does, which plateaus the policy at a fraction of an episode.
#
# They live here rather than as CLI flags because they sit inside an event
# term's ``velocity_range`` dict, which tyro does not flatten.
PUSH_VELOCITY = 0.4
PUSH_ANGULAR_VELOCITY = 0.0

# How long the base controller actually walks, and therefore how long an
# episode is worth running. This tracks the *installed* controller's FSM and
# has to be revisited whenever that changes, because an episode running past
# the walk trains the residual on a stationary robot -- the opposite of this
# task, and doubly so since a standing robot outscores a walking one on both
# tracking terms (0.75 vs 0.66 for ZMP).
#
# The FSM currently walks indefinitely, so the ceiling is ours to pick:
# `Logistic::FSMMoveBoxTableToLeftShelf` now begins with
# `Walking::WalkCmdVelImpl` (`targetCmdVel: [0.1, 0, 0]`, `timeout: 1000.0`)
# rather than `Logistic::GoToTable`, which is commented out. 60 s is ~4 s of
# posture settling then ~6 m of walking. Note the top-level `transitions:` map
# alone does not show this -- it ends at `Logistic::Demo`, and the walk is
# inside that Meta state's own transitions. It was not always so: the stock
# config walked 1 m to the table and then stood for good, which is what the
# earlier `WALK_WINDOW_S = 16.0` was sized for.
#
# 60 -> 90 when `PUSH_WARMUP_S` arrived, and the two have to move together. The
# warm-up removes the first-push massacre, which on its own would have lifted
# survival from ~20% to ~39% and given away the headroom `PUSH_VELOCITY = 0.4`
# was deliberately calibrated for. At the measured post-warm-up hazard of
# ~0.019/s, exp(-0.019 * (T - 10)) puts 90 s back at ~22%: the same difficulty as
# before, with the mortality spread across steady walking instead of piled onto
# one startup event. It also buys 13.3 pushes per episode against 10, and a third
# fewer resets per hour -- worth real throughput, since an env reset destroys and
# rebuilds its mc_rtc controller.
#
# Survival numbers from before this change are not comparable to ones after it.
WALK_WINDOW_S = 90.0

# ZMP tracking payment, sized off the zero-residual baseline rather than picked:
# measured over 64 s x 16 envs, the CoM-to-ZMP offset error runs median 2.1 cm,
# p75 4.4 cm, p90 9.0 cm. `std` is where the exponential has fallen to 1/e, so
# 5 cm scores that baseline 0.68 on average -- ordinary walking is well paid,
# a push landing (p90) drops payment to 0.02, and there is a third of the term
# left for the residual to earn. Tighten it and the signal is mostly noise
# (0.02 scores 0.41); loosen it and it saturates (0.10 scores 0.84).
#
# The weight is per second like every other one here, so a perfectly tracking
# episode earns 16 over the walk window and the baseline ~11, against the -40 a
# fall costs (the -2000 termination penalty lands on one 20 ms step). Dense
# shaping that ranks good balance without ever out-paying survival.
ZMP_TRACKING_STD = 0.05
ZMP_TRACKING_WEIGHT = 1.0

# CoM-velocity payment, the velocity half of the same LIPM state, sized the same
# way. Over 96 s x 16 envs (131 episodes, 58 of them falls) the error runs median
# 1.2 cm/s and mean 3.3 cm/s. It is kept equal in weight to the ZMP term because
# the two are complementary halves of one state, not two views of the same thing.
#
# The reason it is worth having is the second column of that measurement: in the
# last second before a fall the error is 0.347 m/s against 0.022 m/s before a
# time-out -- a 16x separation, against 5x for the ZMP error. It is the sharpest
# early warning of the two.
#
# Two scales, not one, because the axes are not comparable: measured per axis the
# horizontal error runs median 1.17 cm/s and the vertical 0.10 cm/s. Under one
# shared kernel the vertical channel scores 0.94 and is effectively free -- which
# silently cost the term its whole reason for existing, since crouch-collapse is
# a *vertical* failure. At 5 cm/s horizontal and 0.5 cm/s vertical the two
# channels score 0.79 and 0.81 on the baseline, so both keep comparable headroom
# and neither saturates the other. The total, 0.80, lands where the old single
# kernel was (0.785), so the weight carries over unchanged.
COM_VELOCITY_TRACKING_STD = 0.05
COM_VELOCITY_TRACKING_STD_VERTICAL = 0.005
COM_VELOCITY_TRACKING_WEIGHT = 1.0

# A viewer default, not a training one: every env is its own mc_rtc controller
# (~70 MB, ~570 ms to construct, built serially), so replaying at the training
# env count would spend minutes and gigabytes before the first frame.
PLAY_NUM_ENVS = 1


def _make_env_cfg(
  control: str,
  num_envs: int = 128,
  num_workers: int | None = None,
  residual_scale: float | dict[str, float] | None = None,
  episode_length_s: float = WALK_WINDOW_S,
  push_velocity: float = PUSH_VELOCITY,
  push_angular_velocity: float = PUSH_ANGULAR_VELOCITY,
  console_output: Literal["none", "single", "all"] = "none",
  print_residual_every: int = 0,
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  """Build the residual balance env cfg for the config's ``MainRobot``."""
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )
  # The stance height the reward aims at, taken from the controller's own
  # default attitude rather than a constant here: it is the height mc_rtc holds,
  # so the reward target and the base policy cannot drift apart.
  nominal_height = mc_rtc.get_default_root_position(robot_name)[2]

  # The residual acts on the legs only: balancing is what this task rewards, and
  # the upper body does not hold the robot up. This is the task's opinion, so it
  # lives here -- a locomotion or manipulation task would want the arms, and the
  # robot's own `get_residual_joints` stays the place for exclusions that hold
  # whatever the task is (a joint mc_rtc models as fixed can carry no residual
  # anywhere, and is dropped there). Filtering that set rather than taking the
  # legs directly keeps the robot's carve-outs and refJointOrder ordering.
  upper_body = set(mc_rtc.get_upper_body_joints(robot_name))
  residual_joints = tuple(j for j in robot.get_residual_joints() if j not in upper_body)

  ##
  # Actions: the residual on top of mc_rtc.
  ##

  # Residual authority, in the control channel's own unit (rad for position,
  # Nm for torque -- one number cannot serve both). ``scale`` maps the policy's
  # ~unit output into that authority and ``clip`` makes it a hard bound: a
  # residual able to outvote the controller is how the policy learns to freeze
  # the gait instead of stabilizing it (a swing trajectory is ~0.5 rad;
  # rejecting a push needs far less).
  #
  # That bound is set in position units, but the actuators are unlimited on
  # purpose (mc_mujoco parity, see `pd_actuator_configuration`), so what it
  # really buys is torque: 0.01 rad through the real PDgains_sim.dat gains is
  # 22-27% of every leg joint's *hardware* limit. Measured over 64 s x 16 envs
  # that is affordable -- mc_rtc alone asks ~5% of the budget, and a saturated
  # residual takes the worst joint (ankle pitch) to 0.64 of its limit without
  # adding a single over-limit step. Re-measure before raising this: nothing in
  # the sim clamps, so a residual that outgrows the hardware is invisible here
  # and divergent on the robot. That warning was then ignored: a run at 0.1
  # (20899 iterations, 2026-07-31) put a saturated residual at 220-270% of the
  # hardware limit, and -- because ``scale`` multiplies the *exploration* noise
  # too -- left the policy's own dither at 115-140% of it once `mean_std` had
  # grown to 0.52. Both tracking terms read ~0.008 of a possible 1.0 for the
  # whole run, against 0.68/0.80 for the zero-residual baseline, from the first
  # iteration onward and with a near-zero mean residual: the environment was
  # broken before the policy did anything. Hence back to 0.01.
  if residual_scale is None:
    residual_scale = 0.01 if control == "position" else 10.0

  # Per-joint authority, expressed once and used for both the scale and the clip
  # so the two cannot disagree. `processed = raw * scale` is clipped afterwards,
  # so a clip left at the old scalar would silently cap any joint given a larger
  # scale -- the reason these are derived from one dict rather than set apart.
  #
  # The patterns must **partition every actuator**, and only half of that is
  # enforced. mjlab's `resolve_matching_names_values` raises if a joint matches
  # two keys and if a key matches nothing, so a specific entry cannot sit
  # alongside a `".*"` catch-all -- write it as
  # ``{"[LR]_ANKLE_.*": 0.03, "^(?![LR]_ANKLE_).*": 0.005}``. What it does *not*
  # raise on is a joint matching no key at all: `BaseAction.__init__` seeds
  # `scale` to ones and `clip` to +/-inf, so a residual joint left out silently
  # gets scale 1.0 -- 100x the intended authority -- with no clip, which is the
  # failure the paragraph above measured at 220-270% of the hardware limit.
  # Note the resolution is against every actuator matched by `actuator_names`,
  # not just `residual_joints`; the residual subset is sliced out afterwards.
  #
  # Uniform for now: this is exactly the previous scalar behaviour, written in
  # the form that lets the ankles (which is what actually moves the centre of
  # pressure) be raised without also loosening the hips, once the authority probe
  # says which joints are worth it.
  residual_scales: dict[str, float] = (
    residual_scale if isinstance(residual_scale, dict) else {".*": residual_scale}
  )
  residual_clip = {pattern: (-v, v) for pattern, v in residual_scales.items()}

  action_cls = (
    McRtcResidualJointPositionActionCfg
    if control == "position"
    else McRtcResidualJointTorqueActionCfg
  )
  actions: dict[str, ActionTermCfg] = {
    "mc_rtc_residual": action_cls(
      entity_name="robot",
      actuator_names=(".*",),
      residual_actuator_names=residual_joints,
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      frameskip=2,
      num_workers=num_workers,
      # What the `zmp_tracking` reward compares the sim's centre of pressure
      # against, collected per controller step (see `mdp.zmp_tracking`).
      controller_vectors=("planned_zmp", "control_com", "control_com_vel"),
      pd_gains_path=str(robot.pd_gains_path),
      scale=residual_scales,
      clip=residual_clip,
      console_output=console_output,
      print_residual_every=print_residual_every,
    )
  }

  ##
  # Observations.
  ##

  # Noise is scaled to what this robot's signals actually are, measured over a
  # 32 s x 16 env zero-residual walk, not to mjlab's locomotion defaults: those
  # assume ~1 m/s travel, and this base controller walks at 0.09 m/s, so the
  # same absolute corruption buries the signal. RMS per channel was base_lin_vel
  # 0.110, base_ang_vel 0.121, joint_pos 0.085, joint_vel 0.125,
  # projected_gravity 0.577; the levels below keep noise near a tenth to a
  # quarter of that. `joint_vel` in particular carried noise 12x its own signal.
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      history_length=5,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.03, n_max=0.03),
      history_length=5,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    # `biased=True` is what makes the `encoder_bias` startup event take effect;
    # without it the event samples a bias nothing ever reads.
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action, history_length=5),
    # What the controller wants vs. reality, and where its gait is headed:
    # without these the policy cannot phase its residual with the plan it is
    # meant to protect. The error sees encoder-level noise; the reference
    # velocity is controller-internal and known exactly.
    "controller_ref_vel": ObservationTermCfg(
      func=mdp.controller_reference_velocity, history_length=5
    ),
    # The plan the two tracking rewards actually score, which the actor was
    # missing entirely: `zmp_tracking` pays for the measured centre of pressure
    # sitting where `planned_zmp` wants it and `com_velocity_tracking` for the
    # CoM matching `control_com_vel`, but neither quantity reached the policy.
    # `controller_ref_vel` is joint-level and an integration removed from both.
    # Being scored against an invisible target leaves "intervene less" as the
    # only safe strategy, which is what the iteration-200 measurement showed:
    # mean action down to 14% of clip, performance approaching the zero-residual
    # baseline from below instead of passing it. Controller-internal and exact,
    # so no noise, and given the same history as the reference velocity because
    # the plan steps foot to foot and the phase is the point.
    "controller_planned_zmp": ObservationTermCfg(
      func=mdp.controller_planned_zmp_offset, history_length=5
    ),
    "controller_planned_com_vel": ObservationTermCfg(
      func=mdp.controller_planned_com_velocity, history_length=5
    ),
  }

  # The critic sees the same signals without observation noise. Copy the terms
  # rather than rebuilding them from `func`/`params`: a rebuild silently leaves
  # behind every other field, which is how the critic came to lose the actor's
  # history. The group's `enable_corruption=False` is what drops the noise.
  critic_terms = {
    name: replace(term, params=dict(term.params)) for name, term in actor_terms.items()
  }
  # The encoder bias goes with the noise: a privileged critic should value states
  # from the true joint angles, not from the miscalibrated reading the actor has
  # to live with. mjlab's own tracking task splits the two groups the same way.
  critic_terms["joint_pos"] = replace(critic_terms["joint_pos"], params={})

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Rewards. Weights are per second: the manager scales them by step_dt.
  #
  # Structured so that full payment requires walking: `alive` alone pays less
  # than `tracking` + `plan_motion`, both of which a frozen robot (or a
  # controller disturbed into standing) forfeits. The residual penalties only
  # break ties toward minimal intervention; the action term's hard clip is
  # what actually bounds the residual.
  ##

  rewards = {
    # Sized by *gradient* scale, not just by discounted horizon. The manager
    # multiplies every weight by `step_dt`, so a fall lands as one step of
    # `weight * 0.02` against dense steps of ~0.03: at -2000 that is a single
    # -40 sample among -0.03 ones, a 1000x outlier inside a minibatch that then
    # gets advantage-normalized. Measured across every run to 2026-07-31 the
    # cost was total: `Loss/value` never converged (0.7-5.5) and the adaptive
    # KL schedule pinned the learning rate to rsl_rl's 1e-5 floor from
    # iteration 0 and never left it -- 20899 iterations of no learning. -200
    # keeps a fall worth ~4, a couple of seconds of dense reward, which is all
    # the 2 s discounted horizon can see anyway.
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    # The one term that pays for the residual doing its job rather than for
    # merely not falling: the controller plans a ZMP every period, and a push
    # it cannot absorb shows up here -- as the centre of pressure running away
    # from that plan -- well before the base tips far enough to terminate.
    "zmp_tracking": RewardTermCfg(
      func=mdp.zmp_tracking,
      weight=ZMP_TRACKING_WEIGHT,
      params={
        "std": ZMP_TRACKING_STD,
        "sensor_names": mdp.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "action_name": "mc_rtc_residual",
      },
    ),
    "com_velocity_tracking": RewardTermCfg(
      func=mdp.com_velocity_tracking,
      weight=COM_VELOCITY_TRACKING_WEIGHT,
      params={
        "std": COM_VELOCITY_TRACKING_STD,
        "std_vertical": COM_VELOCITY_TRACKING_STD_VERTICAL,
        "asset_cfg": SceneEntityCfg("robot"),
        "action_name": "mc_rtc_residual",
      },
    ),
    "residual_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.1),
    "residual_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(45.0)}
    ),
    # Crouch-collapse keeps the trunk upright, so `fell_over` misses it. This
    # is the termination that actually fires most (27 of 37 baseline failures
    # over 65 s x 16 envs); walking dips to z~0.75, so the threshold has room.
    "collapsed": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.7 * nominal_height},
    ),
    "controller_failed": TerminationTermCfg(
      func=mdp.controller_failed, params={"action_name": "mc_rtc_residual"}
    ),
  }

  ##
  # Events: the disturbances the residual has to earn its keep against.
  ##

  events = {
    # Must come first, and must not be dropped: this is mjlab's default reset
    # event, and defining an `events` dict at all replaces that default. It is
    # the only term that resets *joints*; without it every episode starts from
    # the model's qpos0 (legs straight, z~0.86) rather than the controller's
    # half-sitting stance, mc_rtc initializes against that posture, and its FSM
    # never reaches the walking state -- the robot stands still all episode
    # with no residual able to change it. `reset_base` below then offsets from
    # the default root state this writes, so the order matters too.
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, mode="reset"
    ),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # Deliberately empty, but nothing enforces that -- the action term's
        # `_check_initial_pose_agreement` reads the entity's *static* init state
        # and at most prints a line, and it never looks at this range.
        #
        # It used to be a hazard: mc_rtc seeds its walking plan from the base
        # pose in the *controller config*, never from the pose the sim hands
        # `init()`/`reset()`, and with yaw drawn over +/-pi that killed 39% of
        # episodes 4-7 s in, before any push, by `collapsed` -- the controller
        # chasing a motion it did not command (measured CoM speed 0.91 m/s
        # against a 0.1 m/s walk target). That is now reconciled every reset,
        # worker-side: `_seed_real_robot` teleports the estimated robot onto the
        # sim's pose and re-runs `MCController::reset` so the walking references
        # rebuild there. So this is empty because randomising it buys nothing,
        # not because something would stop you.
        #
        # Nothing is lost. Every actor observation is body-frame or
        # robot-internal (`base_lin_vel`, `base_ang_vel`, `projected_gravity`,
        # joints, controller channels), and the ground is a bare infinite plane,
        # so translating or spinning the start makes an episode neither
        # observationally nor dynamically different. The randomisation that
        # *would* matter -- initial tilt, height, joint state -- is not what this
        # was doing, and initial *velocity* is not an option either: `reset()`
        # takes encoders and a pose but no velocity, so the controller would
        # again start believing something false.
        #
        # No z offset either: dropping the robot injects a transient the
        # controller has to absorb before the episode even starts.
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 7.0),
      params={
        "velocity_range": {
          "x": (-push_velocity, push_velocity),
          "y": (-push_velocity, push_velocity),
          "roll": (-push_angular_velocity, push_angular_velocity),
          "pitch": (-push_angular_velocity, push_angular_velocity),
        }
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
  }

  # Metrics are per-step averages (sum / step_count), with no weight and no dt
  # scaling -- which is the whole point of logging the ZMP error here as well as
  # paying for it in the rewards. Every `Episode_Reward/*` curve is an episode
  # *sum*, and those turned out to correlate with episode length at r = +0.98:
  # they move when the robot survives longer, not when it tracks better, so they
  # could not answer "is the control improving". This one is in metres and is
  # length-independent, so it can.
  #
  # The two go together: that average runs over *every* step, and a step with
  # the feet unloaded contributes 0 m (there is no centre of pressure to place),
  # so `zmp_error` alone drops when the robot spends more time off the ground --
  # backwards for a tracking error. `MetricsTermCfg.reduce` has no masked mean,
  # so the denominator is published instead. Read the tracking quality as
  # `zmp_error / zmp_grounded`, and `zmp_grounded` on its own as a health curve.
  metric_params = {
    "sensor_names": mdp.GROUND_CONTACT_SENSORS,
    "asset_cfg": SceneEntityCfg("robot"),
    "action_name": "mc_rtc_residual",
  }
  metrics = {
    "zmp_error": MetricsTermCfg(func=mdp.zmp_error, params=dict(metric_params)),
    "zmp_grounded": MetricsTermCfg(func=mdp.zmp_grounded, params=dict(metric_params)),
  }

  ##
  # Assemble. Solver settings follow mc_mujoco's HRP5Pmain.xml, as in the demo.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminations=terminations,
    events=events,
    metrics=metrics,
    decimation=20,
    episode_length_s=episode_length_s,
    sim=SimulationCfg(
      # A standing robot needs a handful of constraint rows, so the default
      # heuristic (sized off the demo's stance) overflows the moment one falls
      # -- and `njmax` is a hard per-world cap: past it MuJoCo drops rows and
      # simulates the fall wrong rather than failing. Falls are most of this
      # task, so budget for a sprawled robot; these match mjlab's own humanoid
      # locomotion task.
      njmax=1500,
      nconmax=100,
      mujoco=MujocoCfg(
        timestep=0.001,
        integrator="euler",
        solver="newton",
        iterations=50,
        tolerance=1e-10,
        jacobian="dense",
      ),
    ),
  )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Retune a training cfg for a viewer session."""
  cfg.scene.num_envs = PLAY_NUM_ENVS
  cfg.episode_length_s = 1e10  # only a fall should end an episode
  cfg.observations["actor"].enable_corruption = False
  return cfg


# A viewer session is one controller, so silencing it hides the only thing worth
# watching; "single" rather than "all" keeps env 0 readable if `--num-envs` grows.
PLAY_CONSOLE_OUTPUT = "single"  # "none" #

# The residual is the whole point of watching this task, so print it -- but the
# policy acts at 50 Hz and no one reads 50 lines a second. Every 10th step is
# 5 Hz, fast enough to see the residual react to a push. `MC_MJLAB_PRINT_RESIDUAL`
# retunes it (0 silences) without editing this.
PLAY_PRINT_RESIDUAL_EVERY = 10


def residual_balance_position_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *position* targets."""
  cfg = _make_env_cfg(
    control="position",
    console_output=PLAY_CONSOLE_OUTPUT if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


def residual_balance_torque_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *torques*."""
  cfg = _make_env_cfg(
    control="torque",
    console_output=PLAY_CONSOLE_OUTPUT if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


# RL config.


def residual_balance_ppo_cfg(
  max_iterations: int = 500, experiment_name: str = "mc_rtc_residual_balance"
) -> RslRlOnPolicyRunnerCfg:
  """PPO settings, following mjlab's locomotion configs."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # The residual should start near zero: a unit init_std would hand the
        # controller a huge random offset on step one and knock it over. 0.2 is
        # still too much of one. Measured against the zero-residual baseline
        # through `scripts/compare_to_baseline.py`, a 500-iteration
        # policy scored 18.6% survival against the baseline's 26.5% and was
        # below it on both tracking terms -- it spends the whole run climbing
        # back out of the hole its own random initialization dug, and reached
        # only 1238 steps against 1330. A residual policy should start
        # indistinguishable from the controller it wraps and improve from
        # there, never much worse.
        "init_std": 0.1,
        "std_type": "scalar",
        # A floor, and it is aimed at the learning rate as much as at
        # exploration. The adaptive schedule halves the rate whenever measured
        # KL exceeds 2x `desired_kl`, and for Gaussians KL ~ dmu^2 / (2 sigma^2)
        # -- so a shrinking sigma inflates KL for an unchanged weight step. Over
        # the 2026-08-12_18-30-10 run `Policy/mean_std` fell 0.0999 -> 0.0440,
        # a 5x inflation on its own, and the rate sat pinned at its 1e-5 floor
        # for 54% of the last 500 iterations. Clamping sigma at roughly half
        # `init_std` addresses the collapse and the pinning together.
        #
        # The upper bound is the old worry, not the current one: at 0.005
        # entropy the std used to climb 0.2 -> 0.52 unchecked.
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
      # An order of magnitude below mjlab's locomotion configs (0.005), because
      # on a *residual* task the exploration noise is itself a disturbance: it
      # goes through the real PD gains onto the joints the controller is
      # balancing on. At 0.005 nothing pushed back -- `Policy/mean_std` climbed
      # 0.2 -> 0.52 over 20899 iterations, and 0.2 -> 0.62 over the 14417 before
      # that. That was ruinous only in combination with `residual_scale = 0.1`,
      # where it left the dither at 115-140% of the hardware torque limit; back
      # at 0.01 the same std costs ~13-16%, which is untidy rather than fatal.
      #
      # Hence reduced, not zeroed. Zero does not remove the noise -- `std` stays
      # learnable and starts at `init_std` either way -- it removes the pressure
      # to *grow* it, and the opposite failure (std collapsing early onto a
      # brittle local optimum) is the more expensive one to discover late. The
      # `std_range` clamp above now bounds the inflation from both ends anyway.
      entropy_coef=0.0005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      # The one parameter with authority over the learning rate on this task.
      # rsl_rl's adaptive schedule halves the rate whenever the measured KL
      # exceeds 2x this, clamped to a 1e-5 floor -- and at 0.01 it hit that
      # floor in the first iteration of *every* run so far and never left it
      # (8% of iterations above it across 500, peaking at 5e-5). The policy
      # still learns, but at a crawl: it was still climbing toward the
      # zero-residual baseline from below when the 500-iteration run ended.
      #
      # 0.02 is the conservative first step. If the rate still pins, the next
      # move is 0.03, and the suspect after that is `obs_normalization`: the
      # running normalizer shifts between when `old_actions_log_prob` is stored
      # at collection and when the update runs, which inflates *measured* KL
      # with no weight change at all. rsl_rl logs no KL scalar to confirm that
      # from the outside.
      desired_kl=0.02,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    # Longer rollouts than mjlab's locomotion default (24), because collection
    # here is ~99% mc_rtc: at 128 envs the measured split is 2.9 s collecting
    # against 0.03 s learning, so a longer rollout is nearly free per sample.
    # 24 steps is 0.48 s against episodes of 10-30 s, which leaves GAE almost
    # pure bootstrap off a critic that was not converging; 48 halves the number
    # of updates and doubles the horizon each advantage is estimated over,
    # which is also the cheapest relief for the KL blowups that floored the
    # learning rate.
    num_steps_per_env=48,
    max_iterations=max_iterations,
    logger="wandb",
  )
