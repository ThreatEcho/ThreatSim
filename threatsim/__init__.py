# File: threatsim/__init__.py
# Description: Initialization script for ThreatSim suite
# Purpose: Configuration and initialization for ThreatSim framework
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

"""ThreatSim: A cybersecurity adversarial RL simulation framework.

This package provides:
- ThreatSimEnv: Enhanced cybersecurity simulation environment
- RedTeamEnv / BlueTeamEnv: Single-agent wrappers for RL training
- Utilities for logging, visualization, and metrics
"""

# Version info
__version__ = "0.3.1"
__author__ = "ThreatSim Team"

# Import the core components
try:
    from .envs.threatsim_env import ThreatSimEnv, ConfigLoader
    from .envs.wrappers import make_red_env, make_blue_env, RedTeamEnv, BlueTeamEnv
    __all__ = ["ThreatSimEnv", "ConfigLoader", "make_red_env", "make_blue_env", "RedTeamEnv", "BlueTeamEnv"]
except ImportError as e:
    # Fallback if some components are missing
    print(f"Warning: Some ThreatSim components could not be imported: {e}")
    __all__ = []

# Export dynamic Tactic enum from configuration
try:
    config_loader = ConfigLoader()
    Tactic = config_loader.get_tactic_enum()
    __all__.append("Tactic")
except Exception as e:
    print(f"Warning: Could not load Tactic enum from configuration: {e}")
    Tactic = None
