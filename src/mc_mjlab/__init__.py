"""mc_mjlab: mc_rtc controller integration and HRP5P assets for mjlab."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2].absolute().resolve()
MC_RTC_YAML_PATH = REPO_ROOT / "etc" / "mc_rtc.yaml"
