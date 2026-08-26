"""
Configuration Manager
=====================

Loads all YAML configuration files used by the project.

Author: louis John
Project: Brachycera-CNN
"""

from pathlib import Path
from typing import Any
import yaml


class ConfigManager:
    """
    Loads and stores project configuration.
    """

    def __init__(self, config_dir: str = "config"):

        self.config_dir = Path(config_dir)
        
        self.config = self._load_yaml("config.yaml")
        self.paths = self._load_yaml("paths.yaml")
        self.logging = self._load_yaml("logging.yaml")
        self.views = self._load_yaml("views.yaml")

    def _load_yaml(self, filename: str) -> dict:

        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str) -> Any:
        """
        Generic access to config values.

        Example:
            cfg.get("training", "epochs")
        """

        value = self.config

        for key in keys:

            if key not in value:
                raise KeyError(
                    f"Configuration key {' -> '.join(keys)} not found."
                )

            value = value[key]

        return value



    def project(self):
        return self.config

    def path(self):
        return self.paths

    def log(self):
        return self.logging
        
    def view(self):
        return self.views
        
# Global singleton
CONFIG = ConfigManager()
