# File: threatsim/envs/threatsim_env.py
# Description: Core ThreatSim multi-agent cybersecurity simulation environment
# Purpose: Scientifically rigorous cybersecurity simulation
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import json
import os
import yaml
import random
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum

import numpy as np
import gymnasium as gymn
from gymnasium import spaces
import networkx as nx

from ..utils.logger import logger

class ConfigLoader:
    """
    Configuration loader for unified ThreatSim configuration management.
    
    Handles loading and validation of configuration files including:
    - Unified system configuration (config.json)
    - MITRE ATT&CK framework data (mitre_attack.json)
    - Dynamic tactic enumeration from configuration
    """
    
    def __init__(self, 
                 unified_config_path: str = "data/config.json",
                 mitre_config_path: str = "data/mitre_attack.json"):
        """
        Initialize configuration loader.
        
        Args:
            unified_config_path: Path to unified configuration file
            mitre_config_path: Path to MITRE ATT&CK configuration file
        """
        self.unified_config_path = unified_config_path
        self.mitre_config_path = mitre_config_path
        
        logger.debug(f"Loading configurations from {unified_config_path} and {mitre_config_path}")
        
        # Load configurations with error handling
        self.unified_config = self._load_json(unified_config_path)
        self.mitre_config = self._load_json(mitre_config_path)
        
        # Extract and validate core configuration
        self.core_config = self._extract_core_config()
        
        # Build dynamic tactic enumeration
        self._build_tactic_enum()
        
        logger.success("Configuration loaded successfully")
    
    def _load_json(self, path: str) -> Dict:
        """
        Load JSON configuration file with error handling.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file is missing
        """
        if not os.path.exists(path):
            logger.error(f"Required configuration file missing: {path}")
            raise FileNotFoundError(f"Required configuration file missing: {path}")
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            logger.debug(f"Successfully loaded {path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            raise
    
    def _extract_core_config(self) -> Dict:
        """Extract and structure core configuration from unified config."""
        core_config = {
            "system_info": self.unified_config.get("system_config", {
                "version": "0.3.1",
                "description": "ThreatSim unified configuration"
            }),
            "reward_structure": self.unified_config.get("reward_structure", {}),
            "win_conditions": self.unified_config.get("win_conditions", {}),
            "environment_mechanics": self.unified_config.get("environment_mechanics", {}),
            "security_controls": self.unified_config.get("security_controls", {}),
            "blue_team_responses": self.unified_config.get("blue_team_responses", {}),
            "heuristic_opponents": self.unified_config.get("heuristic_opponents", {}),
            "debug_settings": self.unified_config.get("debug_settings", {"enabled": False})
        }
        
        logger.debug("Core configuration extracted from unified config")
        return core_config
    
    def _build_tactic_enum(self):
        """Build dynamic Tactic enumeration from MITRE configuration."""
        tactics_config = self.mitre_config["tactics"]
        
        # Create dynamic Tactic enum from configuration
        tactic_items = [(name, data["id"]) for name, data in tactics_config.items()]
        self.Tactic = Enum('Tactic', tactic_items)
        
        logger.debug(f"Built dynamic Tactic enum with {len(tactic_items)} tactics")
    
    def get_tactic_enum(self):
        """Get dynamically created Tactic enumeration."""
        return self.Tactic
    
    def get_debug_settings(self) -> Dict:
        """Get debug configuration settings."""
        return self.core_config.get("debug_settings", {"enabled": False})
    
    def get_reward_config(self) -> Dict:
        """Get complete reward configuration for both teams."""
        reward_structure = self.core_config["reward_structure"]
        
        red_rewards = reward_structure.get("red_team_rewards", {})
        red_penalties = reward_structure.get("red_team_penalties", {})
        blue_rewards = reward_structure.get("blue_team_rewards", {})
        blue_penalties = reward_structure.get("blue_team_penalties", {})
        
        return {**red_rewards, **red_penalties, **blue_rewards, **blue_penalties}
    
    def get_win_conditions(self) -> Dict:
        """Get win conditions for both teams."""
        win_conditions = self.core_config["win_conditions"]
        return {
            **win_conditions.get("red_team", {}),
            **win_conditions.get("blue_team", {})
        }
    
    def get_security_defaults(self) -> Dict:
        """Get default security control configurations."""
        return self.core_config["security_controls"].get("default_controls", {})
    
    def get_env_mechanics(self) -> Dict:
        """
        Get environment mechanics with proper defaults.
        
        Returns:
            Complete environment mechanics configuration with fallback defaults
        """
        mechanics = {}
        
        if "environment_mechanics" in self.core_config:
            env_mech = self.core_config["environment_mechanics"]
            
            # Merge all environment mechanics sections
            mechanics.update(env_mech.get("success_probability", {}))
            mechanics.update(env_mech.get("detection_system", {}))
            mechanics.update(env_mech.get("prevention_system", {}))
            mechanics.update(env_mech.get("tactic_prerequisites", {}))
            
            # Add configuration-specific values
            mechanics["noise_level_cap"] = env_mech.get("noise_level_cap", 1.0)
            mechanics["noise_reduction_monitor"] = env_mech.get("noise_reduction_monitor", 0.08)
            mechanics["noise_reduction_alert"] = env_mech.get("noise_reduction_alert", 0.12)
        
        # Ensure critical defaults are present with balanced values
        default_mechanics = {
            "noise_level_cap": 1.0,
            "max_success_probability": 0.95,
            "stealth_threshold": 0.3,
            "noise_impact_factor": 0.4,
            "base_prevention_probability": 0.10,  # REDUCED from 0.20 for balance
            "alert_increase_detection": 0.15,
            "alert_increase_prevention": 0.10,
            "alert_decay_rate": 0.05,
            "compromise_level_bonus": 0.1,
            # New balanced defense parameters
            "defense_calculation_mode": "additive",
            "max_defense_reduction": 0.55,  # Maximum 55% reduction from defenses
            "min_success_probability": 0.20  # Minimum 20% success chance
        }
        
        # Apply defaults for missing values
        for key, default_value in default_mechanics.items():
            if key not in mechanics:
                mechanics[key] = default_value
        
        return mechanics
    
    def get_ttp_tactic_mapping(self) -> Dict:
        """Get TTP to tactic mapping from MITRE configuration."""
        return self.mitre_config.get("technique_tactic_mapping", {})
    
    def get_implicit_vulnerability(self, tactic_name: str) -> Dict:
        """
        Get implicit vulnerability template for a given tactic.
        
        Args:
            tactic_name: Name of the tactic
            
        Returns:
            Vulnerability configuration dictionary
        """
        default_vulns = self.mitre_config.get("default_vulnerabilities", {})
        default_template = default_vulns.get("implicit_vulnerability_template", {})
        tactic_specific = default_vulns.get("tactic_specific_vulnerabilities", {})
        
        return tactic_specific.get(tactic_name, default_template)
    
    def get_security_control_effectiveness(self, tactic_name: str) -> Dict:
        """Get security control effectiveness mappings for a tactic."""
        control_mappings = self.mitre_config.get("security_control_mappings", {})
        effectiveness_by_tactic = control_mappings.get("control_effectiveness_by_tactic", {})
        return effectiveness_by_tactic.get(tactic_name, {})
    
    def get_blue_response_config(self) -> Dict:
        """Get Blue team response type configurations."""
        return self.core_config["blue_team_responses"].get("response_types", {})

class Node:
    """
    Network node representation with cybersecurity state tracking.
    
    Maintains compromise status, security controls, and activity metrics
    for realistic cybersecurity simulation.
    """
    
    def __init__(self, node_id: str, value: float, vulnerabilities: List[Dict], 
                 security_controls: Optional[Dict] = None, config_loader: ConfigLoader = None):
        """
        Initialize network node.
        
        Args:
            node_id: Unique identifier for the node
            value: Business/strategic value of the node
            vulnerabilities: List of vulnerability configurations
            security_controls: Security control configurations
            config_loader: Configuration loader instance
        """
        self.id = node_id
        self.value = float(value)
        self.vulnerabilities = deepcopy(vulnerabilities)
        
        # Security and compromise state
        self.compromised = False
        self.compromise_level = 0
        self.detected_ttps = set()
        self.prevented_ttps = set()
        self.successful_ttps = set()
        self.completed_tactics = set()
        
        # Activity and noise tracking
        self.noise_level = 0.0
        self.last_activity_step = -1
        self.hardening_level = 1.0
        
        # Load security controls from configuration
        if config_loader is None:
            config_loader = ConfigLoader()
        default_controls = config_loader.get_security_defaults()
        self.security_controls = security_controls or default_controls.copy()
    
    def reset(self):
        """Reset node to clean, uncompromised state."""
        self.compromised = False
        self.compromise_level = 0
        self.detected_ttps.clear()
        self.prevented_ttps.clear()
        self.successful_ttps.clear()
        self.completed_tactics.clear()
        self.noise_level = 0.0
        self.last_activity_step = -1
        self.hardening_level = 1.0
    
    def get_effective_security_control(self, control_type: str) -> float:
        """
        Get effective security control value accounting for hardening.
        
        Args:
            control_type: Type of security control (e.g., 'edr', 'firewall')
            
        Returns:
            Effective control value between 0.0 and 1.0
        """
        base_value = self.security_controls.get(control_type, 0.5)
        return min(1.0, base_value * self.hardening_level)

class ThreatSimEnv(gymn.Env):
    """
    Core ThreatSim multi-agent cybersecurity simulation environment.
    
    Implements realistic cybersecurity scenarios with empirically-derived parameters,
    supporting Red team (attackers) vs Blue team (defenders) training with balanced
    defense calculations.
    """
    
    def __init__(self, yaml_path: str, max_steps: int = 100, seed: int = None, debug_mode: bool = None):
        """
        Initialize ThreatSim environment.
        
        Args:
            yaml_path: Path to scenario YAML file
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            debug_mode: Override debug mode setting (None uses config value)
        """
        super().__init__()
        
        logger.debug(f"Initializing ThreatSim environment with scenario: {yaml_path}")
        
        # Load configuration system
        self.config_loader = ConfigLoader()
        self.reward_config = self.config_loader.get_reward_config()
        self.win_conditions = self.config_loader.get_win_conditions()
        self.env_mechanics = self.config_loader.get_env_mechanics()
        self.ttp_mapping = self.config_loader.get_ttp_tactic_mapping()
        self.blue_responses = self.config_loader.get_blue_response_config()
        self.Tactic = self.config_loader.get_tactic_enum()
        
        # Configure debug mode
        debug_settings = self.config_loader.get_debug_settings()
        self.debug_mode = debug_mode if debug_mode is not None else debug_settings.get("enabled", False)
        
        if self.debug_mode:
            logger.info("Debug mode enabled for ThreatSim environment")
        
        # Load scenario configuration
        with open(yaml_path, "r") as f:
            scenario = yaml.safe_load(f)
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Build network topology
        self._build_network(scenario)
        
        # Configure action and observation spaces
        self.n_nodes = len(self.nodes)
        self.n_tactics = len(self.Tactic)
        
        self.red_action_space = spaces.MultiDiscrete([self.n_nodes, self.n_tactics, 5])
        self.blue_action_space = spaces.MultiDiscrete([self.n_nodes, self.n_tactics, 5])
        
        # Observation space calculation
        self._obs_size = self._calculate_obs_size()
        self.observation_spaces = {
            "red": spaces.Box(low=0, high=1, shape=(self._obs_size,), dtype=np.float32),
            "blue": spaces.Box(low=0, high=1, shape=(self._obs_size,), dtype=np.float32)
        }
        
        # Game state initialization
        self.attack_chain = []
        self.blue_alerts = []
        self.global_alert_level = 0.0
        self.target_value = scenario.get("target_value", 10.0)
        
        if self.debug_mode:
            logger.debug(f"Target value: {self.target_value}")
            logger.debug(f"Network nodes: {self.n_nodes}")
            logger.debug(f"Tactics available: {self.n_tactics}")
            logger.debug(f"Total network value: {sum(n.value for n in self.nodes)}")
            logger.debug(f"Defense calculation mode: {self.env_mechanics.get('defense_calculation_mode', 'additive')}")
        
        logger.success(f"ThreatSim environment initialized with {self.n_nodes} nodes, {self.n_tactics} tactics")
        
        self.reset()
    
    def _build_network(self, scenario: Dict):
        """
        Build network topology from scenario configuration.
        
        Args:
            scenario: Scenario configuration dictionary
        """
        self.nodes = []
        self.node_lookup = {}
        
        # Create nodes from scenario
        for node_config in scenario.get("nodes", []):
            node = Node(
                node_id=node_config["id"],
                value=node_config.get("value", 0.0),
                vulnerabilities=node_config.get("vulnerabilities", []),
                security_controls=node_config.get("security_controls", None),
                config_loader=self.config_loader
            )
            self.nodes.append(node)
            self.node_lookup[node.id] = node
            
            if self.debug_mode:
                logger.debug(f"Node: {node.id} (Value: {node.value})")
                for vuln in node.vulnerabilities:
                    exploit = vuln.get('exploitability', 0.5)
                    detect = vuln.get('detection_prob', 0.3)
                    logger.debug(f"  Vulnerability {vuln['name']}: exploit={exploit:.3f}, detect={detect:.3f}")
        
        # Build network graph
        self.network = nx.Graph()
        for node in self.nodes:
            self.network.add_node(node.id)
        
        # Add edges from scenario
        for edge in scenario.get("edges", []):
            if len(edge) == 2:
                parent_id, child_id = edge
                if parent_id in self.node_lookup and child_id in self.node_lookup:
                    self.network.add_edge(parent_id, child_id)
        
        self.entry_points = set(scenario.get("entry_points", []))
        
        if self.debug_mode:
            logger.debug(f"Entry points: {self.entry_points}")
        
        logger.debug(f"Network built with {len(self.nodes)} nodes and {len(self.network.edges)} edges")
    
    def _calculate_obs_size(self):
        """Calculate observation vector size based on node and global features."""
        per_node_features = 4 + self.n_tactics  # compromise, detected, prevented, noise + tactics
        global_features = 3  # alert_level, step_ratio, progress_ratio
        return self.n_nodes * per_node_features + global_features
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for episode
            options: Additional reset options
            
        Returns:
            Tuple of (observations, info)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_step = 0
        self.attack_chain = []
        self.blue_alerts = []
        self.global_alert_level = 0.0
        
        # Reset all nodes
        for node in self.nodes:
            node.reset()
        
        if self.debug_mode:
            logger.debug(f"Environment reset - Step 0, Target: {self.target_value}")
        
        logger.debug("Environment reset to initial state")
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        Generate observations for both teams.
        
        Returns:
            Dictionary with 'red' and 'blue' observation arrays
        """
        red_obs = []
        blue_obs = []
        
        # Per-node features
        for node in self.nodes:
            node_features = [
                node.compromise_level / 3.0,  # Normalized compromise level
                float(len(node.detected_ttps) > 0),  # Detection indicator
                float(len(node.prevented_ttps) > 0),  # Prevention indicator
                node.noise_level  # Current noise level
            ]
            
            # Tactic completion status for each tactic
            for tactic in self.Tactic:
                node_features.append(float(tactic in node.completed_tactics))
            
            red_obs.extend(node_features)
            blue_obs.extend(node_features)
        
        # Global state features
        total_compromised_value = sum(node.value for node in self.nodes if node.compromised)
        global_features = [
            self.global_alert_level,  # Current alert level
            self.current_step / self.max_steps,  # Episode progress
            total_compromised_value / max(1.0, self.target_value)  # Compromise progress
        ]
        
        red_obs.extend(global_features)
        blue_obs.extend(global_features)
        
        return {
            "red": np.array(red_obs, dtype=np.float32),
            "blue": np.array(blue_obs, dtype=np.float32)
        }
    
    def _check_tactic_prerequisites(self, node: Node, tactic) -> bool:
        """
        Check if tactic prerequisites are met.
        
        Args:
            node: Target node
            tactic: Tactic to execute
            
        Returns:
            True if prerequisites are met
        """
        # Use permissive mode from configuration
        permissive_mode = self.env_mechanics.get("permissive_mode", True)
        return permissive_mode
    
    def _calculate_security_effectiveness(self, node: Node, tactic_name: str) -> float:
        """
        Calculate security control effectiveness using additive model.
        
        FIXED: Replaces multiplicative stacking with additive reduction to prevent
        exponential defensive advantage while maintaining empirical parameter basis.
        
        Args:
            node: Target node
            tactic_name: Name of the tactic being executed
            
        Returns:
            Prevention modifier between min_success_probability and 1.0
        """
        control_effectiveness = self.config_loader.get_security_control_effectiveness(tactic_name)
        
        if not control_effectiveness:
            return 1.0  # No security control reduction
        
        primary_controls = control_effectiveness.get("primary_controls", [])
        effectiveness_multipliers = control_effectiveness.get("effectiveness_multipliers", {})
        
        # FIXED: Use additive reduction instead of multiplicative stacking
        total_reduction = 0.0
        
        for control in primary_controls:
            if control in effectiveness_multipliers:
                control_value = node.get_effective_security_control(control)
                multiplier = effectiveness_multipliers[control]
                control_effectiveness_factor = control_value * multiplier
                
                # Add to total reduction instead of multiplying
                total_reduction += control_effectiveness_factor
                
                if self.debug_mode:
                    logger.debug(f"Control {control}: value={control_value:.3f}, multiplier={multiplier:.3f}, effectiveness={control_effectiveness_factor:.3f}")
        
        # Apply configuration limits to maintain balance
        max_reduction = self.env_mechanics.get('max_defense_reduction', 0.55)
        total_reduction = min(max_reduction, total_reduction)
        
        # Calculate prevention modifier
        prevention_modifier = 1.0 - total_reduction
        
        # Ensure minimum success probability
        min_success = self.env_mechanics.get('min_success_probability', 0.20)
        prevention_modifier = max(min_success, prevention_modifier)
        
        return prevention_modifier
    
    def _execute_red_ttp(self, node_idx: int, tactic, technique_idx: int) -> float:
        """
        Execute Red team Tactic, Technique, and Procedure (TTP).
        
        Args:
            node_idx: Target node index
            tactic: Tactic enumeration value
            technique_idx: Technique variant index
            
        Returns:
            Reward value for the action
        """
        if self.debug_mode:
            logger.debug(f"Red TTP execution - Node: {node_idx}, Tactic: {tactic}, Technique: {technique_idx}")
        
        # Validate node index
        if node_idx >= len(self.nodes):
            if self.debug_mode:
                logger.warning(f"Invalid node index: {node_idx} >= {len(self.nodes)}")
            return self.reward_config.get('invalid_action_penalty', -10.0)
        
        node = self.nodes[node_idx]
        
        if self.debug_mode:
            logger.debug(f"Target node: {node.id} (Value: {node.value}, Compromised: {node.compromised})")
        
        # Check prerequisites
        if not self._check_tactic_prerequisites(node, tactic):
            if self.debug_mode:
                logger.warning("Tactic prerequisites not met")
            return self.reward_config.get('invalid_action_penalty', -10.0)
        
        # Find matching vulnerabilities or create implicit one
        matching_vulns = [v for v in node.vulnerabilities 
                         if self._ttp_matches_tactic(v['name'], tactic)]
        
        if self.debug_mode:
            logger.debug(f"Node vulnerabilities: {len(node.vulnerabilities)}, Matching: {len(matching_vulns)}")
        
        if not matching_vulns:
            # Create implicit vulnerability from configuration
            implicit_vuln = self.config_loader.get_implicit_vulnerability(tactic.name)
            implicit_vuln = implicit_vuln.copy()
            implicit_vuln['name'] = f'T{1000 + tactic.value}'
            matching_vulns = [implicit_vuln]
            
            if self.debug_mode:
                logger.debug(f"Created implicit vulnerability: {implicit_vuln}")
        
        # Select vulnerability based on technique index
        vuln = matching_vulns[min(technique_idx, len(matching_vulns) - 1)]
        
        if self.debug_mode:
            logger.debug(f"Selected vulnerability: {vuln['name']}")
            logger.debug(f"Base exploitability: {vuln.get('exploitability', 0.5):.3f}")
            logger.debug(f"Detection probability: {vuln.get('detection_prob', 0.3):.3f}")
        
        # Calculate success probability with multiple factors
        base_success = vuln.get('exploitability', 0.5)
        
        # FIXED: Use new additive security control effectiveness calculation
        prevention_modifier = self._calculate_security_effectiveness(node, tactic.name)
        
        # Stealth bonus calculation
        stealth_threshold = self.env_mechanics.get('stealth_threshold', 0.3)
        stealth_modifier = 1.0
        if node.noise_level < stealth_threshold:
            stealth_modifier = 1.0 + (0.3 * (1.0 - node.noise_level))
        
        # Compromise level bonus
        compromise_bonus = self.env_mechanics.get('compromise_level_bonus', 0.1)
        if node.compromise_level > 0:
            prevention_modifier *= (1.0 + compromise_bonus)
        
        # Final success probability
        success_prob = base_success * prevention_modifier * stealth_modifier
        max_prob = self.env_mechanics.get('max_success_probability', 0.95)
        success_prob = min(max_prob, success_prob)
        
        if self.debug_mode:
            logger.debug(f"Success probability calculation:")
            logger.debug(f"  Base success: {base_success:.3f}")
            logger.debug(f"  Prevention modifier: {prevention_modifier:.3f}")
            logger.debug(f"  Stealth modifier: {stealth_modifier:.3f}")
            logger.debug(f"  Final success probability: {success_prob:.3f}")
        
        # Execute TTP attempt
        roll = random.random()
        success = roll < success_prob
        
        if self.debug_mode:
            logger.debug(f"Random roll: {roll:.3f}, Success: {success}")
        
        if success:
            reward = self._handle_ttp_success(node, tactic, vuln)
            if self.debug_mode:
                logger.debug(f"Success reward: {reward:.3f}")
            return reward
        else:
            reward = self._handle_ttp_failure(node, vuln)
            if self.debug_mode:
                logger.debug(f"Failure penalty: {reward:.3f}")
            return reward
    
    def _handle_ttp_success(self, node: Node, tactic, vuln: Dict) -> float:
        """Handle successful TTP execution."""
        if self.debug_mode:
            logger.debug("Handling TTP success")
        
        # Record success
        node.successful_ttps.add(vuln['name'])
        node.completed_tactics.add(tactic)
        node.last_activity_step = self.current_step
        
        # Calculate base reward
        base_reward = self._calculate_success_reward(node, tactic)
        
        if self.debug_mode:
            logger.debug(f"Base success reward: {base_reward:.3f}")
        
        # Apply configuration-based bonuses
        stealth_threshold = self.env_mechanics.get('stealth_threshold', 0.3)
        if node.noise_level < stealth_threshold:
            stealth_multiplier = self.reward_config.get('stealth_bonus_multiplier', 1.2)
            base_reward *= stealth_multiplier
            if self.debug_mode:
                logger.debug(f"Stealth bonus applied: x{stealth_multiplier:.2f}")
        
        # Chain bonus for multiple tactics
        chain_threshold = 3
        if len(node.completed_tactics) >= chain_threshold:
            chain_multiplier = self.reward_config.get('chain_bonus_multiplier', 1.5)
            base_reward *= chain_multiplier
            if self.debug_mode:
                logger.debug(f"Chain bonus applied: x{chain_multiplier:.2f}")
        
        # Add noise and handle detection
        noise_added = vuln.get('noise_level', 0.1) * 0.3
        noise_cap = self.env_mechanics.get('noise_level_cap', 1.0)
        node.noise_level = min(noise_cap, node.noise_level + noise_added)
        
        # Detection probability calculation
        detection_prob = vuln.get('detection_prob', 0.3)
        noise_impact = self.env_mechanics.get('noise_impact_factor', 0.4)
        detection_prob *= (1.0 + node.noise_level * noise_impact)
        
        detection_roll = random.random()
        detected = detection_roll < detection_prob
        
        if self.debug_mode:
            logger.debug(f"Detection probability: {detection_prob:.3f}")
            logger.debug(f"Detection roll: {detection_roll:.3f}, Detected: {detected}")
        
        if detected:
            node.detected_ttps.add(vuln['name'])
            alert_increase = self.env_mechanics.get('alert_increase_detection', 0.15)
            self.global_alert_level = min(1.0, self.global_alert_level + alert_increase)
            detection_penalty = self.reward_config.get('detection_penalty', -2.0)
            base_reward += detection_penalty
            if self.debug_mode:
                logger.debug(f"Detection penalty: {detection_penalty:.3f}")
        
        return base_reward
    
    def _handle_ttp_failure(self, node: Node, vuln: Dict) -> float:
        """Handle failed TTP execution."""
        if self.debug_mode:
            logger.debug("Handling TTP failure")
        
        base_prevention_prob = self.env_mechanics.get('base_prevention_probability', 0.10)
        
        prevention_roll = random.random()
        prevented = prevention_roll < base_prevention_prob
        
        if self.debug_mode:
            logger.debug(f"Prevention probability: {base_prevention_prob:.3f}")
            logger.debug(f"Prevention roll: {prevention_roll:.3f}, Prevented: {prevented}")
        
        if prevented:
            # TTP was actively prevented
            node.prevented_ttps.add(vuln['name'])
            alert_increase = self.env_mechanics.get('alert_increase_prevention', 0.10)
            self.global_alert_level = min(1.0, self.global_alert_level + alert_increase)
            penalty = self.reward_config.get('prevention_penalty', -5.0)
            if self.debug_mode:
                logger.debug(f"Prevention penalty: {penalty:.3f}")
            return penalty
        else:
            # TTP simply failed
            penalty = self.reward_config.get('failure_penalty', -1.0)
            if self.debug_mode:
                logger.debug(f"Failure penalty: {penalty:.3f}")
            return penalty
    
    def _calculate_success_reward(self, node: Node, tactic) -> float:
        """
        Calculate reward for successful tactic execution.
        
        Args:
            node: Target node
            tactic: Executed tactic
            
        Returns:
            Base reward value
        """
        tactic_name = tactic.name
        
        if self.debug_mode:
            logger.debug(f"Calculating success reward for {tactic_name}")
        
        if tactic_name == 'INITIAL_ACCESS':
            if not node.compromised:
                node.compromised = True
                node.compromise_level = 1
                reward = self.reward_config.get('initial_access_reward', 10.0)
                if node.id in self.entry_points:
                    progress_bonus = self.reward_config.get('progress_bonus', 2.0)
                    reward += progress_bonus
                    if self.debug_mode:
                        logger.debug(f"Entry point bonus: +{progress_bonus:.3f}")
                if self.debug_mode:
                    logger.debug(f"Node compromised! Reward: {reward:.3f}")
                return reward
            else:
                base_reward = self.reward_config.get('base_tactic_reward', 1.0)
                if self.debug_mode:
                    logger.debug(f"Already compromised, base reward: {base_reward:.3f}")
                return base_reward
        
        elif tactic_name == 'PRIVILEGE_ESCALATION':
            old_level = node.compromise_level
            node.compromise_level = min(3, node.compromise_level + 1)
            escalation_gain = node.compromise_level - old_level
            multiplier = self.reward_config.get('privilege_escalation_multiplier', 3.0)
            reward = multiplier * escalation_gain + node.value * 0.5
            if self.debug_mode:
                logger.debug(f"Privilege escalation: {old_level} -> {node.compromise_level}, reward: {reward:.3f}")
            return reward
        
        elif tactic_name == 'IMPACT':
            multiplier = self.reward_config.get('impact_multiplier', 2.0)
            if node.compromise_level >= 2:
                multiplier *= 1.5
            reward = node.value * multiplier
            if self.debug_mode:
                logger.debug(f"Impact reward: {node.value:.1f} x {multiplier:.2f} = {reward:.3f}")
            return reward
        
        elif tactic_name == 'LATERAL_MOVEMENT':
            bonus = self.reward_config.get('lateral_movement_bonus', 5.0)
            if node.value >= 3.0:
                progress_bonus = self.reward_config.get('progress_bonus', 2.0)
                bonus += progress_bonus
            if self.debug_mode:
                logger.debug(f"Lateral movement reward: {bonus:.3f}")
            return bonus
        
        else:
            base_reward = self.reward_config.get('base_tactic_reward', 1.0)
            value_bonus = node.value * 0.3
            total_reward = base_reward + value_bonus
            if self.debug_mode:
                logger.debug(f"Default tactic reward: {base_reward:.3f} + {value_bonus:.3f} = {total_reward:.3f}")
            return total_reward
    
    def _ttp_matches_tactic(self, ttp_name: str, tactic) -> bool:
        """
        Check if TTP matches the given tactic.
        
        Args:
            ttp_name: TTP identifier
            tactic: Tactic enumeration value
            
        Returns:
            True if TTP matches tactic
        """
        base_ttp = ttp_name.split('.')[0]
        mapped_tactic_name = self.ttp_mapping.get(base_ttp)
        
        if mapped_tactic_name is None:
            if self.debug_mode:
                logger.debug(f"Unknown TTP {base_ttp}, allowing match")
            return True  # Allow unknown TTPs
        
        matches = mapped_tactic_name == tactic.name
        if self.debug_mode:
            logger.debug(f"TTP {ttp_name} -> {mapped_tactic_name} vs {tactic.name}: {matches}")
        return matches
    
    def _execute_blue_response(self, node_idx: int, focus_tactic, response_type: int) -> float:
        """
        Execute Blue team response action.
        
        Args:
            node_idx: Target node index
            focus_tactic: Tactic to focus response on
            response_type: Type of response (0-4)
            
        Returns:
            Reward value for the response
        """
        if node_idx >= len(self.nodes):
            return self.reward_config.get('ineffective_response_penalty', -0.5)
        
        node = self.nodes[node_idx]
        response_names = ["monitor", "alert", "block", "isolate", "remediate"]
        
        if response_type >= len(response_names):
            return self.reward_config.get('ineffective_response_penalty', -0.5)
        
        response_name = response_names[response_type]
        response_config = self.blue_responses.get(response_name, {})
        effectiveness = response_config.get('effectiveness', 0.5)
        
        if self.debug_mode:
            logger.debug(f"Blue response: {response_name} on node {node.id}")
        
        # Execute specific response type
        if response_name == "monitor":
            return self._execute_monitor_response(node, effectiveness)
        elif response_name == "alert":
            return self._execute_alert_response(node, response_config, effectiveness)
        elif response_name == "block":
            return self._execute_block_response(node, focus_tactic, response_config, effectiveness)
        elif response_name == "isolate":
            return self._execute_isolate_response(node, response_config, effectiveness)
        elif response_name == "remediate":
            return self._execute_remediate_response(node, response_config, effectiveness)
        
        return 0.0
    
    def _execute_monitor_response(self, node: Node, effectiveness: float) -> float:
        """Execute monitoring response."""
        if node.detected_ttps or node.noise_level > 0.3:
            base_reward = self.reward_config.get('detection_reward', 2.0) * effectiveness
            noise_reduction = self.env_mechanics.get('noise_reduction_monitor', 0.08)
            node.noise_level = max(0, node.noise_level - noise_reduction)
            return base_reward
        else:
            return 0.2  # Small reward for monitoring
    
    def _execute_alert_response(self, node: Node, response_config: Dict, effectiveness: float) -> float:
        """Execute alert response."""
        if node.detected_ttps or node.noise_level > 0.5:
            alert_increase = response_config.get('alert_level_increase', 0.25)
            self.global_alert_level = min(1.0, self.global_alert_level + alert_increase)
            
            base_reward = self.reward_config.get('detection_reward', 2.0) * effectiveness
            noise_reduction = self.env_mechanics.get('noise_reduction_alert', 0.12)
            node.noise_level = max(0, node.noise_level - noise_reduction)
            return base_reward
        else:
            return self.reward_config.get('false_positive_penalty', -1.0)
    
    def _execute_block_response(self, node: Node, focus_tactic, response_config: Dict, effectiveness: float) -> float:
        """Execute blocking response."""
        control_improvement = response_config.get('control_improvement', 0.2)
        
        if focus_tactic.name in ['INITIAL_ACCESS', 'LATERAL_MOVEMENT']:
            control_type = 'edr' if focus_tactic.name == 'INITIAL_ACCESS' else 'firewall'
            current_control = node.security_controls.get(control_type, 0)
            node.security_controls[control_type] = min(1.0, current_control + control_improvement)
            return self.reward_config.get('prevention_reward', 3.0) * effectiveness
        else:
            return 0.5
    
    def _execute_isolate_response(self, node: Node, response_config: Dict, effectiveness: float) -> float:
        """Execute isolation response."""
        if node.compromised or node.detected_ttps:
            base_reward = self.reward_config.get('containment_reward', 4.0) * effectiveness
            compromise_reduction = response_config.get('compromise_reduction', 1)
            node.compromise_level = max(0, node.compromise_level - compromise_reduction)
            
            if node.compromise_level == 0:
                node.compromised = False
                isolation_bonus = self.reward_config.get('isolation_reward', 5.0)
                base_reward += isolation_bonus
            
            return base_reward
        else:
            return self.reward_config.get('ineffective_response_penalty', -0.5)
    
    def _execute_remediate_response(self, node: Node, response_config: Dict, effectiveness: float) -> float:
        """Execute remediation response."""
        if node.detected_ttps and node.compromised:
            success_prob = response_config.get('success_probability', 0.75)
            
            if random.random() < success_prob:
                # Successful remediation
                node.compromised = False
                node.compromise_level = 0
                node.successful_ttps.clear()
                node.completed_tactics.clear()
                node.noise_level = 0.0
                return self.reward_config.get('remediation_reward', 6.0) * effectiveness
            else:
                return self.reward_config.get('ineffective_response_penalty', -0.5)
        else:
            return self.reward_config.get('ineffective_response_penalty', -0.5)
    
    def step(self, actions):
        """
        Execute one simulation step.
        
        Args:
            actions: Dictionary with 'red' and 'blue' actions
            
        Returns:
            Tuple of (observations, rewards, terminations, truncations, info)
        """
        red_action = actions["red"]
        blue_action = actions["blue"]
        
        if self.debug_mode:
            logger.debug(f"Step {self.current_step}: Red action: {red_action}, Blue action: {blue_action}")
        
        # Parse actions
        red_node_idx, red_tactic_idx, red_technique_idx = red_action
        blue_node_idx, blue_tactic_idx, blue_response = blue_action
        
        red_tactic = list(self.Tactic)[red_tactic_idx]
        blue_focus_tactic = list(self.Tactic)[blue_tactic_idx]
        
        # Execute actions (Blue first for defensive advantage)
        blue_reward = self._execute_blue_response(blue_node_idx, blue_focus_tactic, blue_response)
        red_reward = self._execute_red_ttp(red_node_idx, red_tactic, red_technique_idx)
        
        if self.debug_mode:
            logger.debug(f"Blue reward: {blue_reward:.3f}, Red reward: {red_reward:.3f}")
        
        # Apply time penalties
        time_penalty = self.current_step * self.reward_config.get('time_penalty_factor', 0.01)
        red_reward -= time_penalty
        
        # Record attack chain
        self.attack_chain.append({
            'step': self.current_step,
            'red_action': [red_node_idx, red_tactic_idx, red_technique_idx],
            'blue_action': [blue_node_idx, blue_tactic_idx, blue_response],
            'red_reward': red_reward,
            'blue_reward': blue_reward
        })
        
        # Check win conditions
        red_wins, blue_wins = self._check_win_conditions()
        
        if self.debug_mode:
            compromised_value = sum(node.value for node in self.nodes if node.compromised)
            logger.debug(f"Current state - Compromised value: {compromised_value:.1f}/{self.target_value:.1f}")
            logger.debug(f"Progress: {compromised_value/self.target_value*100:.1f}%, Red wins: {red_wins}, Blue wins: {blue_wins}")
        
        # Apply win/loss bonuses
        if red_wins:
            efficiency = (self.max_steps - self.current_step) / self.max_steps
            win_bonus = self.reward_config.get('win_bonus', 20.0)
            efficiency_bonus = self.reward_config.get('efficiency_bonus', 5.0)
            red_reward += win_bonus + (efficiency * efficiency_bonus)
            blue_reward -= self.reward_config.get('loss_penalty', 10.0)
            if self.debug_mode:
                logger.debug(f"Red team wins! Win bonus: {win_bonus + (efficiency * efficiency_bonus):.3f}")
        elif blue_wins:
            red_reward -= self.reward_config.get('loss_penalty', 10.0)
            blue_reward += self.reward_config.get('win_bonus', 20.0)
            if self.debug_mode:
                logger.debug(f"Blue team wins! Win bonus: {self.reward_config.get('win_bonus', 20.0):.3f}")
        
        # Apply alert decay
        alert_decay = self.env_mechanics.get('alert_decay_rate', 0.05)
        self.global_alert_level = max(0, self.global_alert_level - alert_decay)
        
        # Prepare return values
        terminations = {"red": red_wins or blue_wins, "blue": red_wins or blue_wins}
        rewards = {"red": red_reward, "blue": blue_reward}
        truncations = {"red": False, "blue": False}
        
        self.current_step += 1
        
        # Comprehensive info dictionary
        compromised_value = sum(node.value for node in self.nodes if node.compromised)
        info = {
            "compromised_value": compromised_value,
            "target_value": self.target_value,
            "alert_level": self.global_alert_level,
            "compromised_nodes": sum(1 for n in self.nodes if n.compromised),
            "detected_ttps": sum(len(n.detected_ttps) for n in self.nodes),
            "prevented_ttps": sum(len(n.prevented_ttps) for n in self.nodes),
            "attack_chain_length": len(self.attack_chain),
            "avg_noise_level": np.mean([n.noise_level for n in self.nodes]),
            "red_win": red_wins,
            "blue_win": blue_wins,
            "target_reached": red_wins,
            "progress_ratio": compromised_value / self.target_value if self.target_value > 0 else 0,
            "prevention_ratio": 1.0 - (compromised_value / self.target_value) if self.target_value > 0 else 1.0,
            "config_version": self.config_loader.core_config["system_info"]["version"],
            "defense_calculation_mode": self.env_mechanics.get('defense_calculation_mode', 'additive')
        }
        
        return self._get_obs(), rewards, terminations, truncations, info
    
    def _check_win_conditions(self) -> Tuple[bool, bool]:
        """
        Check win conditions for both teams.
        
        Returns:
            Tuple of (red_wins, blue_wins)
        """
        compromised_value = sum(node.value for node in self.nodes if node.compromised)
        
        if self.debug_mode:
            logger.debug(f"Win condition check - Compromised value: {compromised_value:.1f}, Target: {self.target_value:.1f}")
            logger.debug(f"Progress ratio: {compromised_value/self.target_value:.3f}")
        
        # Red team win conditions with balanced thresholds
        primary_threshold = self.win_conditions.get('primary_success_threshold', 0.70)  # REDUCED from 0.80
        partial_threshold = self.win_conditions.get('partial_success_threshold', 0.50)   # REDUCED from 0.60
        progress_threshold = self.win_conditions.get('significant_progress_threshold', 0.35)  # REDUCED from 0.40
        node_threshold = self.win_conditions.get('node_compromise_threshold', 0.4)      # REDUCED from 0.50
        tactical_threshold = self.win_conditions.get('tactical_success_threshold', 8)   # REDUCED from 10
        
        if self.debug_mode:
            logger.debug(f"Win condition thresholds - Primary: {primary_threshold:.1%}, Partial: {partial_threshold:.1%}")
            logger.debug(f"Required values - Primary: {self.target_value * primary_threshold:.1f}, Partial: {self.target_value * partial_threshold:.1f}")
        
        compromised_nodes = len([n for n in self.nodes if n.compromised])
        total_tactics = sum(len(n.completed_tactics) for n in self.nodes)
        
        red_wins = (
            compromised_value >= self.target_value * primary_threshold or
            (compromised_value >= self.target_value * partial_threshold and 
             self.current_step >= self.max_steps * 0.6) or
            compromised_nodes >= len(self.nodes) * node_threshold or
            (compromised_value >= self.target_value * progress_threshold and
             self.current_step >= self.max_steps * 0.7) or
            total_tactics >= tactical_threshold
        )
        
        # Blue team win conditions
        prevention_threshold = self.win_conditions.get('prevention_threshold', 0.7)
        prevention_ratio = 1.0 - (compromised_value / self.target_value) if self.target_value > 0 else 1.0
        blue_wins = (
            (self.current_step >= self.max_steps and not red_wins) or
            (prevention_ratio >= prevention_threshold)
        )
        
        if self.debug_mode:
            logger.debug(f"Red win conditions met: {red_wins}")
            logger.debug(f"Blue win conditions met: {blue_wins}")
            logger.debug(f"Compromised nodes: {compromised_nodes}/{len(self.nodes)}")
            logger.debug(f"Total tactics completed: {total_tactics}")
        
        return red_wins, blue_wins
    
    def render(self, mode="human"):
        """
        Render current environment state.
        
        Args:
            mode: Rendering mode
        """
        config_version = self.config_loader.core_config["system_info"]["version"]
        defense_mode = self.env_mechanics.get('defense_calculation_mode', 'additive')
        
        logger.info(f"Step {self.current_step} | Alert Level: {self.global_alert_level:.2f}")
        logger.config("Config Version", config_version)
        logger.config("Defense Calculation", defense_mode)
        
        # Display node status
        active_nodes = [node for node in self.nodes 
                       if node.compromised or node.detected_ttps or node.noise_level > 0.1]
        
        if active_nodes:
            logger.info("Active Node Status:")
            for node in active_nodes:
                status_parts = []
                if node.compromised:
                    status_parts.append(f"Compromised-L{node.compromise_level}")
                if node.detected_ttps:
                    status_parts.append(f"Detected: {len(node.detected_ttps)}")
                if node.prevented_ttps:
                    status_parts.append(f"Prevented: {len(node.prevented_ttps)}")
                if node.noise_level > 0.1:
                    status_parts.append(f"Noise: {node.noise_level:.2f}")
                
                logger.debug(f"{node.id}: {', '.join(status_parts)}")
                
                if node.completed_tactics:
                    tactics = [t.name for t in list(node.completed_tactics)[:3]]
                    if len(node.completed_tactics) > 3:
                        tactics.append("...")
                    logger.debug(f"  Completed: {', '.join(tactics)}")
        
        # Display progress
        compromised_value = sum(node.value for node in self.nodes if node.compromised)
        progress = (compromised_value / self.target_value) * 100 if self.target_value > 0 else 0
        primary_threshold = self.win_conditions.get('primary_success_threshold', 0.70)
        
        logger.metric("Progress", f"{compromised_value:.1f}/{self.target_value:.1f} ({progress:.1f}%)")
        logger.metric("Red Win Threshold", f"{self.target_value * primary_threshold:.1f} ({primary_threshold*100:.0f}%)")
