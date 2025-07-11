# File: threatsim/utils/scenario_manager.py
# Description: Scenario management system for ThreatSim training
# Purpose: Load, validate, and manage cybersecurity training scenarios
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .logger import logger

@dataclass
class ScenarioInfo:
    """Data class for scenario metadata."""
    name: str
    file: str
    difficulty: str
    attack_type: str
    expected_red_win_rate: float
    description: str
    target_value: float
    total_value: float
    statistics: Dict[str, Any]

class ScenarioManager:
    """
    Scenario management system for ThreatSim training scenarios.
    
    Handles loading, validation, and management of cybersecurity training
    scenarios with comprehensive metadata extraction and validation.
    """
    
    def __init__(self, scenarios_dir: str = "scenarios"):
        """
        Initialize scenario manager.
        
        Args:
            scenarios_dir: Directory containing scenario YAML files
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.catalog = {}
        self.validation_cache = {}
        
        # Load all scenarios from directory
        self._load_scenario_catalog()
        
        logger.info(f"Scenario manager initialized with {len(self.catalog)} scenarios")
    
    def _load_scenario_catalog(self):
        """Load and catalog all scenarios from the scenarios directory."""
        if not self.scenarios_dir.exists():
            logger.error(f"Scenarios directory not found: {self.scenarios_dir}")
            return
        
        # Find all YAML files in scenarios directory (excluding subdirectories for now)
        scenario_files = list(self.scenarios_dir.glob("*.yaml"))
        
        logger.debug(f"Found {len(scenario_files)} scenario files")
        
        for scenario_file in scenario_files:
            try:
                scenario_info = self._load_scenario_metadata(scenario_file)
                if scenario_info:
                    scenario_key = scenario_file.stem
                    self.catalog[scenario_key] = scenario_info
                    logger.debug(f"Loaded scenario: {scenario_key}")
                else:
                    logger.warning(f"Failed to load scenario metadata: {scenario_file}")
            except Exception as e:
                logger.error(f"Error loading scenario {scenario_file}: {e}")
    
    def _load_scenario_metadata(self, scenario_file: Path) -> Optional[ScenarioInfo]:
        """
        Load metadata from a scenario file.
        
        Args:
            scenario_file: Path to scenario YAML file
            
        Returns:
            ScenarioInfo object or None if loading fails
        """
        try:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                scenario_data = yaml.safe_load(f)
            
            # Extract training configuration
            training_config = scenario_data.get('training_config', {})
            
            # Calculate scenario statistics
            nodes = scenario_data.get('nodes', [])
            total_value = sum(node.get('value', 0) for node in nodes)
            target_value = scenario_data.get('target_value', 0)
            
            # Extract metadata with defaults
            name = training_config.get('description', scenario_file.stem)
            difficulty = training_config.get('difficulty', 'unknown')
            attack_type = training_config.get('attack_type', 'general')
            expected_red_win_rate = training_config.get('expected_red_win_rate', 0.30)
            
            # Build statistics
            statistics = {
                'total_nodes': len(nodes),
                'total_value': total_value,
                'target_value': target_value,
                'target_percentage': (target_value / total_value * 100) if total_value > 0 else 0,
                'avg_exploitability': self._calculate_avg_exploitability(nodes),
                'avg_detection_prob': self._calculate_avg_detection_prob(nodes),
                'total_vulnerabilities': sum(len(node.get('vulnerabilities', [])) for node in nodes)
            }
            
            return ScenarioInfo(
                name=name,
                file=str(scenario_file),
                difficulty=difficulty,
                attack_type=attack_type,
                expected_red_win_rate=expected_red_win_rate,
                description=name,
                target_value=target_value,
                total_value=total_value,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"Failed to load scenario metadata from {scenario_file}: {e}")
            return None
    
    def _calculate_avg_exploitability(self, nodes: List[Dict]) -> float:
        """Calculate average exploitability across all vulnerabilities."""
        total_exploitability = 0
        total_vulnerabilities = 0
        
        for node in nodes:
            vulnerabilities = node.get('vulnerabilities', [])
            for vuln in vulnerabilities:
                total_exploitability += vuln.get('exploitability', 0.5)
                total_vulnerabilities += 1
        
        return total_exploitability / total_vulnerabilities if total_vulnerabilities > 0 else 0.5
    
    def _calculate_avg_detection_prob(self, nodes: List[Dict]) -> float:
        """Calculate average detection probability across all vulnerabilities."""
        total_detection = 0
        total_vulnerabilities = 0
        
        for node in nodes:
            vulnerabilities = node.get('vulnerabilities', [])
            for vuln in vulnerabilities:
                total_detection += vuln.get('detection_prob', 0.3)
                total_vulnerabilities += 1
        
        return total_detection / total_vulnerabilities if total_vulnerabilities > 0 else 0.3
    
    def get_scenario_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get list of available scenarios.
        
        Returns:
            List of (scenario_key, scenario_info_dict) tuples
        """
        scenario_list = []
        
        for key, scenario_info in self.catalog.items():
            scenario_dict = {
                'name': scenario_info.name,
                'file': scenario_info.file,
                'difficulty': scenario_info.difficulty,
                'attack_type': scenario_info.attack_type,
                'expected_red_win_rate': scenario_info.expected_red_win_rate,
                'description': scenario_info.description,
                'statistics': scenario_info.statistics
            }
            scenario_list.append((key, scenario_dict))
        
        return sorted(scenario_list, key=lambda x: x[1]['difficulty'])
    
    def get_scenario_info(self, scenario_key: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific scenario.
        
        Args:
            scenario_key: Scenario identifier
            
        Returns:
            Scenario information dictionary or None if not found
        """
        if scenario_key not in self.catalog:
            logger.warning(f"Scenario not found: {scenario_key}")
            return None
        
        scenario_info = self.catalog[scenario_key]
        return {
            'name': scenario_info.name,
            'file': scenario_info.file,
            'difficulty': scenario_info.difficulty,
            'attack_type': scenario_info.attack_type,
            'expected_red_win_rate': scenario_info.expected_red_win_rate,
            'description': scenario_info.description,
            'target_value': scenario_info.target_value,
            'total_value': scenario_info.total_value,
            'statistics': scenario_info.statistics
        }
    
    def validate_scenario(self, scenario_path: str, validation_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a scenario against scientific criteria.
        
        Args:
            scenario_path: Path to scenario file
            validation_thresholds: Validation criteria
            
        Returns:
            Validation result dictionary
        """
        try:
            # Check if already validated
            if scenario_path in self.validation_cache:
                return self.validation_cache[scenario_path]
            
            # Load scenario data
            with open(scenario_path, 'r', encoding='utf-8') as f:
                scenario_data = yaml.safe_load(f)
            
            # Extract validation parameters
            nodes = scenario_data.get('nodes', [])
            if not nodes:
                return {
                    'valid': False,
                    'error': 'No nodes found in scenario',
                    'issues': ['Scenario must contain at least one node']
                }
            
            # Calculate scenario statistics
            statistics = {
                'total_nodes': len(nodes),
                'total_value': sum(node.get('value', 0) for node in nodes),
                'target_value': scenario_data.get('target_value', 0),
                'avg_exploitability': self._calculate_avg_exploitability(nodes),
                'avg_detection_prob': self._calculate_avg_detection_prob(nodes),
                'total_vulnerabilities': sum(len(node.get('vulnerabilities', [])) for node in nodes)
            }
            
            statistics['target_percentage'] = (
                statistics['target_value'] / statistics['total_value'] * 100
                if statistics['total_value'] > 0 else 0
            )
            
            # Validate against thresholds
            issues = []
            warnings = []
            
            # Check exploitability range
            exp_range = validation_thresholds.get('exploitability_range', [0.1, 0.95])
            if not (exp_range[0] <= statistics['avg_exploitability'] <= exp_range[1]):
                issues.append(f"Average exploitability {statistics['avg_exploitability']:.2f} outside range {exp_range}")
            
            # Check detection probability range
            det_range = validation_thresholds.get('detection_range', [0.05, 0.98])
            if not (det_range[0] <= statistics['avg_detection_prob'] <= det_range[1]):
                issues.append(f"Average detection probability {statistics['avg_detection_prob']:.2f} outside range {det_range}")
            
            # Check target percentage range
            target_range = validation_thresholds.get('target_percentage_range', [15.0, 80.0])
            if not (target_range[0] <= statistics['target_percentage'] <= target_range[1]):
                issues.append(f"Target percentage {statistics['target_percentage']:.1f}% outside range {target_range}")
            
            # Check for potential balance issues
            if statistics['avg_exploitability'] < 0.3 and statistics['avg_detection_prob'] > 0.7:
                warnings.append("High detection probability combined with low exploitability may cause Red team learning issues")
            
            if statistics['target_percentage'] > 60:
                warnings.append("High target percentage may make Red team objectives too difficult")
            
            # Build validation result
            validation_result = {
                'valid': len(issues) == 0,
                'statistics': statistics,
                'issues': issues,
                'warnings': warnings
            }
            
            # Cache result
            self.validation_cache[scenario_path] = validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Scenario validation failed: {e}")
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'issues': ['Failed to validate scenario due to internal error']
            }
    
    def get_scenarios_by_difficulty(self, difficulty: str) -> List[str]:
        """
        Get scenarios filtered by difficulty level.
        
        Args:
            difficulty: Difficulty level to filter by
            
        Returns:
            List of scenario keys matching the difficulty
        """
        return [
            key for key, scenario_info in self.catalog.items()
            if scenario_info.difficulty == difficulty
        ]
    
    def get_scenarios_by_attack_type(self, attack_type: str) -> List[str]:
        """
        Get scenarios filtered by attack type.
        
        Args:
            attack_type: Attack type to filter by
            
        Returns:
            List of scenario keys matching the attack type
        """
        return [
            key for key, scenario_info in self.catalog.items()
            if scenario_info.attack_type == attack_type
        ]
    
    def reload_scenarios(self):
        """Reload all scenarios from the scenarios directory."""
        self.catalog.clear()
        self.validation_cache.clear()
        self._load_scenario_catalog()
        logger.info(f"Reloaded {len(self.catalog)} scenarios")
