"""First-use symlinks into the mc_rtc workspace's mc_mujoco share.

Robot assets (MJCF, meshes, PD gains) are not tracked in this repo:
each robot package symlinks them in on first use.
"""

from pathlib import Path

MC_MUJOCO_SHARE = Path.home() / "workspace/install/share/mc_mujoco"


def ensure_workspace_link(link: Path, target: Path) -> None:
  if link.exists():
    return
  if link.is_symlink():  # dangling: workspace moved/removed
    link.unlink()
  if not target.exists():
    raise FileNotFoundError(
      f"Robot asset not found: neither {link} nor the workspace copy at "
      f"{target} exists. Build/install mc_mujoco in the ROS workspace "
      "(or restore the file) and retry."
    )
  link.parent.mkdir(parents=True, exist_ok=True)
  try:
    link.symlink_to(target, target_is_directory=target.is_dir())
  except FileExistsError:  # concurrent first use
    pass
