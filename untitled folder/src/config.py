"""Configuration management module."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_env_var(key: str, default: str = None) -> str:
    """Get environment variable with optional default.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


class Config:
    """Configuration class with lazy loading."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get configuration data."""
        if self._config is None:
            self._config = load_config()
        return self._config
    
    @property
    def project(self) -> Dict[str, str]:
        return self.data.get("project", {})
    
    @property
    def api(self) -> Dict[str, Any]:
        return self.data.get("api", {})
    
    @property
    def database(self) -> Dict[str, Any]:
        return self.data.get("database", {})
    
    @property
    def collection(self) -> Dict[str, Any]:
        return self.data.get("collection", {})
    
    @property
    def model(self) -> Dict[str, Any]:
        return self.data.get("model", {})
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.data.get("training", {})
    
    @property
    def privacy(self) -> Dict[str, Any]:
        return self.data.get("privacy", {})


# Global config instance
config = Config()
