# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _parse_bool(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class AppConfig:
    # Core mode flags
    is_energy: bool = _parse_bool(os.getenv("IS_ENERGY"))
    log_path = None
    json_path = None
    sample_path = None

    # Optional: keep other shared configuration here over time
    workflow_config_path: str = os.getenv("WORKFLOW_CONFIG", "ml_workflow.json")

    def apply_cli(self, *, is_energy: bool, workflow_config_path: str) -> None:
        """
        Apply CLI arguments (highest precedence) and propagate to environment for subprocesses.
        """
        self.is_energy = bool(is_energy)
        self.workflow_config_path = workflow_config_path

        # Propagate to environment so subprocesses / child processes can see it if needed
        os.environ["IS_ENERGY"] = "1" if self.is_energy else "0"
        os.environ["WORKFLOW_CONFIG"] = self.workflow_config_path
    
    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def set_json_path(self, json_path: str) -> None:
        self.json_path = json_path
    
    def set_sample_path(self, sample_path: str) -> None:
        self.sample_path = sample_path
    
# The singleton instance
CONFIG = AppConfig()
