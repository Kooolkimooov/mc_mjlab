"""Test script to demonstrate mc_rtc controller integration with mjlab.

Two modes:
  * benchmark (default): step a batch of envs for a fixed number of steps and
    report steady-state throughput.
  * viewer (``--viewer``): run the mc_rtc controllers continuously and display
    the robot in a MuJoCo viewer so its state can be inspected visually.
"""

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mc_mjlab.actions.mc_rtc_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.robots.HRP5P import hrp5p_constants
from mc_mjlab.robots.JVRC1 import jvrc1_constants

REPO_ROOT = Path(__file__).resolve().parents[2]
MC_RTC_YAML = REPO_ROOT / "etc" / "mc_rtc.yaml"


@dataclass(frozen=True)
class RobotSpec:
  """Per-robot wiring for the mc_rtc coupling."""

  cfg_fn: Callable[[], EntityCfg]
  residual_joints: tuple[str, ...]
  pd_gains_path: Path


ROBOTS = {
  "HRP5P": RobotSpec(
    cfg_fn=hrp5p_constants.get_hrp5p_robot_cfg,
    residual_joints=tuple(hrp5p_constants.HRP5P_NOMINAL_EFFORT_LIMITS),
    pd_gains_path=hrp5p_constants.PD_GAINS_PATH,
  ),
  "JVRC1": RobotSpec(
    cfg_fn=jvrc1_constants.get_jvrc1_robot_cfg,
    residual_joints=jvrc1_constants.JVRC1_RESIDUAL_JOINTS,
    pd_gains_path=jvrc1_constants.PD_GAINS_PATH,
  ),
}


def read_main_robot(path: Path) -> str:
  """The config's MainRobot; it selects the mjlab entity, keeping the two
  sides of the coupling on the same robot."""
  for line in path.read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if line.startswith("MainRobot:"):
      return line.split(":", 1)[1].strip()
  raise ValueError(f"No MainRobot key found in {path}")


def prep_cfg_for_mc_rtc(robot_cfg):
  """Prepare a robot cfg for the mc_rtc coupling.

  Re-enables the collision geoms (the geoms are unnamed, so the name-based
  presets match nothing and the robot would fall through the ground) and
  deletes the XML's own motors (mjlab adds its own; keeping both doubles
  ``nu`` with dead actuators).
  """
  base_spec_fn = robot_cfg.spec_fn

  def spec_fn():
    spec = base_spec_fn()
    for geom in spec.geoms:
      if geom.group == 3:  # collision geoms, disabled by get_spec
        geom.contype = 1
        geom.conaffinity = 1
    for act in list(spec.actuators):
      spec.delete(act)
    return spec

  robot_cfg.spec_fn = spec_fn
  robot_cfg.collisions = ()  # skip the name-based preset (matches nothing here)
  return robot_cfg


def print_root_positions(env, step: int) -> None:
  """Print root heights across envs (a dropping z means the robots are falling)."""
  pos = env.scene["robot"].data.root_link_pos_w
  z = pos[:, 2]
  print(
    f"[step {step:>5}] root z (m): min={z.min().item():.3f} "
    f"mean={z.mean().item():.3f} max={z.max().item():.3f} | "
    f"env0 xyz=({pos[0, 0].item():.2f}, {pos[0, 1].item():.2f}, {pos[0, 2].item():.2f})"
  )


class ZeroResidualPolicy:
  """Zero RL residual: the robot tracks the raw mc_rtc output. Optionally
  prints root heights every ``print_every`` steps."""

  def __init__(
    self,
    num_envs: int,
    action_dim: int,
    device: str,
    env=None,
    print_every: int = 0,
  ):
    self._action = torch.zeros((num_envs, action_dim), device=device)
    self._env = env
    self._print_every = print_every
    self._step = 0

  def __call__(self, obs: object) -> torch.Tensor:
    del obs
    if (
      self._env is not None
      and self._print_every
      and self._step % self._print_every == 0
    ):
      print_root_positions(self._env, self._step)
    self._step += 1
    return self._action


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--num-envs",
    type=int,
    default=None,
    help="Number of parallel environments (one mc_rtc controller each). "
    "Default: 512 for the benchmark (controllers cost ~70 MB each, so this "
    "is memory-bound), 2 with a viewer (construction must be quick and "
    "stepping must keep the real-time pace).",
  )
  parser.add_argument("--num-steps", type=int, default=100)
  parser.add_argument(
    "--warmup",
    type=int,
    default=20,
    help="Steps to run before timing, to exclude JIT/warp compilation warmup.",
  )
  parser.add_argument(
    "--device",
    default=None,
    help="Default: cuda for the benchmark, cpu with a viewer.",
  )
  parser.add_argument(
    "--num-workers",
    type=int,
    default=None,
    help="Worker processes hosting the mc_rtc controllers (default: cpu_count - 2).",
  )
  parser.add_argument(
    "--viewer",
    choices=("none", "auto", "native", "viser"),
    default="none",
    help=(
      "Run the controllers continuously and display them. 'auto' picks the "
      "native viewer if a display is available, otherwise the browser-based "
      "viser viewer. 'none' (default) runs the timed benchmark instead."
    ),
  )
  args = parser.parse_args()

  benchmark = args.viewer == "none"
  if args.num_envs is None:
    args.num_envs = 420 if benchmark else 2
  if args.device is None:
    args.device = "cuda" if benchmark else "cpu"

  robot_name = read_main_robot(MC_RTC_YAML)
  if robot_name not in ROBOTS:
    raise SystemExit(
      f"MainRobot '{robot_name}' in {MC_RTC_YAML} has no RobotSpec "
      f"(known: {', '.join(sorted(ROBOTS))})."
    )
  print(f"[mc_rtc] MainRobot: {robot_name}")
  robot = ROBOTS[robot_name]
  robot_cfg = prep_cfg_for_mc_rtc(robot.cfg_fn())

  # A terrain is required for a ground plane; it also spreads the env origins.
  scene_cfg = SceneCfg(
    num_envs=args.num_envs,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": robot_cfg},
  )

  # mc_mujoco parity: controller at 0.002s over a 1 kHz sim (frameskip=2), real
  # PD gains, all controlled joints track mc_rtc but the RL residual only
  # reaches the non-finger joints.
  action_cfg = McRtcResidualJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    residual_actuator_names=robot.residual_joints,
    mc_rtc_config_path=str(MC_RTC_YAML),
    mc_rtc_robot_name=robot_name,
    frameskip=2,
    num_workers=args.num_workers,
    pd_gains_path=str(robot.pd_gains_path),
  )

  # Solver/integrator settings from mc_mujoco's HRP5Pmain.xml.
  env_cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    actions={"robot_joints": action_cfg},
    decimation=2,
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.001,
        integrator="euler",
        solver="newton",
        iterations=50,
        tolerance=1e-10,
        jacobian="dense",
      )
    ),
  )

  print(f"Initializing ManagerBasedRlEnv with {args.num_envs} envs...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

  # Viewer mode: run continuously with a zero residual, paced to real time.
  if args.viewer != "none":
    if args.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved = "native" if has_display else "viser"
    else:
      resolved = args.viewer

    wrapped = RslRlVecEnvWrapper(env)
    # print_every=500: root height once per ~1s of sim time (step_dt=0.002).
    policy = ZeroResidualPolicy(
      wrapped.num_envs,
      wrapped.num_actions,
      args.device,
      env=env,
      print_every=500,
    )
    print(f"Launching {resolved} viewer (close the window or Ctrl+C to stop)...")
    print_root_positions(env, 0)
    if resolved == "native":
      NativeMujocoViewer(wrapped, policy).run()
    else:
      ViserPlayViewer(wrapped, policy).run()
    return

  # Benchmark mode; `warmup` steps are excluded to skip warp/JIT compilation.
  print("Resetting environment...")
  env.reset()
  print(f"Running simulation loop ({args.num_steps} steps, {args.warmup} warmup)...")
  print_root_positions(env, 0)
  action = torch.zeros(
    (env.num_envs, env.action_manager.total_action_dim), device=args.device
  )
  start = time.perf_counter()
  for i in range(args.num_steps):
    if i == args.warmup:
      start = time.perf_counter()
    obs, reward, terminated, truncated, extras = env.step(action)

    if (i + 1) % 50 == 0:
      print(f"Completed {i + 1} steps")
      print_root_positions(env, i + 1)

  elapsed = time.perf_counter() - start
  timed_steps = args.num_steps - args.warmup
  total_steps = timed_steps * env.num_envs
  print(
    f"Test finished successfully! steady-state over {timed_steps} steps x "
    f"{env.num_envs} envs: {elapsed:.2f}s "
    f"({total_steps / elapsed:.0f} env-steps/s, {timed_steps / elapsed:.1f} iters/s)"
  )


if __name__ == "__main__":
  main()
