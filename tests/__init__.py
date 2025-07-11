# File: tests/__init__.py
# Description: Initialization script for ThreatSim test suite
# Purpose: Test configuration and initialization for ThreatSim testing framework
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

"""
ThreatSim Test Suite

This test suite provides comprehensive testing for the ThreatSim framework:
- Environment functionality
- Training system integrity  
- Configuration validation
- Scenario compatibility
- Performance benchmarking

Usage:
    python -m tests.minimal_test        # Quick smoke test
    python -m tests.debug_training      # Full system debug
    python -m tests.test_all_scenarios  # Test all scenarios
    python -m tests.benchmark           # Performance testing
"""

__version__ = "0.3.1"
__author__ = "ThreatSim Team"

# Test configuration
TEST_CONFIG = {
    "quick_test": {
        "timesteps": 64,
        "max_steps": 10,
        "scenarios": ["simple_scenario.yaml"]
    },
    "full_test": {
        "timesteps": 1000,
        "max_steps": 50,
        "scenarios": ["simple_scenario.yaml", "debug_scenario.yaml"]
    },
    "benchmark": {
        "timesteps": 5000,
        "max_steps": 100,
        "scenarios": ["enterprise_scenario.yaml", "ransomware_scenario.yaml"]
    }
}
