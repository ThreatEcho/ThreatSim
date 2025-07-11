# File: threatsim/envs/wrappers.py
# Description: Single-agent environment wrappers for Red and Blue team training
# Purpose: Gymnasium-compatible wrappers with configuration-driven heuristic opponents
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import json
import os
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium
from gymnasium.spaces import Discrete, Box, MultiDiscrete

from .threatsim_env import ThreatSimEnv, ConfigLoader
from ..utils.logger import logger

class RedTeamEnv(gymnasium.Env):
    """
    Red team (attacker) environment wrapper for single-agent RL training.
    
    Wraps the multi-agent ThreatSim environment to provide a single-agent
    interface for Red team training against configurable Blue team opponents.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 scenario_path: str,
                 max_steps: int = 100,
                 blue_policy: str = "heuristic",
                 action_masking: bool = True):
        """
        Initialize Red team environment.
        
        Args:
            scenario_path: Path to scenario YAML file
            max_steps: Maximum steps per episode
            blue_policy: Blue team policy ('passive', 'aggressive', 'heuristic')
            action_masking: Whether to use action masking
        """
        super().__init__()
        
        logger.debug(f"Initializing Red team environment with scenario: {scenario_path}")
        
        # Load configuration for heuristic opponents
        self.config_loader = ConfigLoader()
        self.heuristic_config = self.config_loader.unified_config.get("heuristic_opponents", {})
        
        # Initialize core environment
        self.env = ThreatSimEnv(yaml_path=scenario_path, max_steps=max_steps)
        self.blue_policy = blue_policy
        self.action_masking = action_masking
        
        # Get Blue team policy configuration
        blue_policies = self.heuristic_config.get("blue_team_policies", {})
        self.blue_config = blue_policies.get(blue_policy, {})
        
        # Configure action space (discrete actions for RL training)
        self.n_nodes = self.env.n_nodes
        self.n_tactics = self.env.n_tactics
        self.n_techniques = 3
        
        self.n_actions = self.n_nodes * self.n_tactics * self.n_techniques
        self.action_space = Discrete(self.n_actions)
        
        # Observation space matches core environment
        self.observation_space = Box(
            low=0, high=1, shape=(self.env._obs_size,), dtype=np.float32
        )
        
        # Episode tracking for statistics
        self.episode_count = 0
        self.success_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Blue team state tracking
        self.blue_memory = {
            'threat_level': 0.0,
            'last_detections': [],
            'response_history': []
        }
        
        logger.success(f"Red team environment initialized with {blue_policy} Blue policy")
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode flat discrete action into multi-discrete components.
        
        Args:
            action: Flat action integer
            
        Returns:
            Tuple of (node_idx, tactic_idx, technique_idx)
        """
        technique = action % self.n_techniques
        action = action // self.n_techniques
        tactic = action % self.n_tactics
        node = action // self.n_tactics
        return node, tactic, technique
    
    def _get_blue_action(self, obs: Dict) -> np.ndarray:
        """
        Generate Blue team action based on configured policy.
        
        Args:
            obs: Current observations
            
        Returns:
            Blue team action array
        """
        current_state = self._analyze_current_state()
        threat_level = self._calculate_threat_level(current_state)
        
        if self.blue_policy == "passive":
            return self._get_passive_blue_action(threat_level)
        elif self.blue_policy == "aggressive":
            return self._get_aggressive_blue_action(threat_level, current_state)
        else:  # heuristic (default)
            return self._get_heuristic_blue_action(threat_level, current_state)
    
    def _analyze_current_state(self) -> Dict:
        """
        Analyze current environment state for Blue team decision making.
        
        Returns:
            Dictionary with current state metrics
        """
        state = {
            'compromised_nodes': [i for i, n in enumerate(self.env.nodes) if n.compromised],
            'detected_activities': sum(len(n.detected_ttps) for n in self.env.nodes),
            'prevented_activities': sum(len(n.prevented_ttps) for n in self.env.nodes),
            'high_noise_nodes': [i for i, n in enumerate(self.env.nodes) if n.noise_level > 0.4],
            'alert_level': self.env.global_alert_level,
            'step': self.env.current_step
        }
        return state
    
    def _calculate_threat_level(self, state: Dict) -> float:
        """
        Calculate current threat level for Blue team response.
        
        Args:
            state: Current state metrics
            
        Returns:
            Threat level between 0.0 and 1.0
        """
        threat_score = 0.0
        
        # Compromised nodes have highest weight
        threat_score += len(state['compromised_nodes']) * 0.4
        
        # Detection events indicate active attacks
        threat_score += min(state['detected_activities'], 10) * 0.02
        
        # Global alert level reflects system-wide threat
        threat_score += state['alert_level'] * 0.3
        
        # High noise activities suggest ongoing attacks
        threat_score += len(state['high_noise_nodes']) * 0.1
        
        # Time pressure increases threat perception
        time_factor = state['step'] / self.env.max_steps
        threat_score += time_factor * 0.2
        
        return min(1.0, threat_score)
    
    def _get_passive_blue_action(self, threat_level: float) -> np.ndarray:
        """
        Generate passive Blue team action.
        
        Args:
            threat_level: Current threat level
            
        Returns:
            Blue team action array
        """
        detection_capability = self.blue_config.get("detection_capability", 0.35)
        
        # Passive teams primarily monitor
        if threat_level > 0.7 and np.random.random() < detection_capability:
            response = np.random.choice([1, 2], p=[0.8, 0.2])  # Alert or block
        else:
            response = 0  # Monitor
        
        node = np.random.randint(self.n_nodes)
        tactic = np.random.randint(self.n_tactics)
        
        return np.array([node, tactic, response])
    
    def _get_aggressive_blue_action(self, threat_level: float, state: Dict) -> np.ndarray:
        """
        Generate aggressive Blue team action.
        
        Args:
            threat_level: Current threat level
            state: Current state metrics
            
        Returns:
            Blue team action array
        """
        detection_capability = self.blue_config.get("detection_capability", 0.80)
        response_speed = self.blue_config.get("response_speed", 0.80)
        
        # Aggressive teams prioritize immediate containment
        if state['compromised_nodes'] and np.random.random() < response_speed:
            node = np.random.choice(state['compromised_nodes'])
            tactic = 8  # LATERAL_MOVEMENT tactic index
            
            if threat_level > 0.6:
                response = np.random.choice([3, 4], p=[0.6, 0.4])  # Isolate or remediate
            else:
                response = np.random.choice([2, 3], p=[0.7, 0.3])  # Block or isolate
        else:
            # Focus on entry points
            entry_nodes = [i for i, n in enumerate(self.env.nodes) if n.id in self.env.entry_points]
            node = np.random.choice(entry_nodes) if entry_nodes else np.random.randint(self.n_nodes)
            tactic = 1  # INITIAL_ACCESS tactic index
            response = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        
        return np.array([node, tactic, response])
    
    def _get_heuristic_blue_action(self, threat_level: float, state: Dict) -> np.ndarray:
        """
        Generate heuristic Blue team action using intelligent targeting.
        
        Args:
            threat_level: Current threat level
            state: Current state metrics
            
        Returns:
            Blue team action array
        """
        detection_capability = self.blue_config.get("detection_capability", 0.60)
        threat_prediction = self.blue_config.get("threat_prediction", 0.45)
        
        target_node = None
        target_tactic = None
        
        # Priority-based targeting
        if state['compromised_nodes'] and np.random.random() < detection_capability:
            target_node = np.random.choice(state['compromised_nodes'])
            target_tactic = 8  # LATERAL_MOVEMENT
        elif state['high_noise_nodes'] and np.random.random() < detection_capability * 0.8:
            target_node = np.random.choice(state['high_noise_nodes'])
            target_tactic = 7  # DISCOVERY
        elif np.random.random() < threat_prediction:
            entry_nodes = [i for i, n in enumerate(self.env.nodes) if n.id in self.env.entry_points]
            target_node = np.random.choice(entry_nodes) if entry_nodes else 0
            target_tactic = 1  # INITIAL_ACCESS
        
        # Fallback to random targeting
        if target_node is None:
            target_node = np.random.randint(self.n_nodes)
            target_tactic = np.random.randint(self.n_tactics)
        
        # Response selection based on threat level
        if threat_level > 0.7:
            response = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])  # Block, isolate, remediate
        elif threat_level > 0.3:
            response = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])  # Alert, block, isolate
        else:
            response = np.random.choice([0, 1], p=[0.7, 0.3])  # Monitor, alert
        
        return np.array([target_node, target_tactic, response])
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        obs_dict, info = self.env.reset(seed=seed, options=options)
        
        # Update episode tracking
        self.episode_count += 1
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Reset Blue team memory
        self.blue_memory = {
            'threat_level': 0.0,
            'last_detections': [],
            'response_history': []
        }
        
        logger.debug(f"Red team environment reset for episode {self.episode_count}")
        
        return obs_dict["red"], info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.
        
        Args:
            action: Red team action (discrete)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode Red team action
        red_node, red_tactic, red_technique = self._decode_action(action)
        red_action = np.array([red_node, red_tactic, red_technique])
        
        # Generate Blue team action
        blue_action = self._get_blue_action({"obs": self.env._get_obs()})
        
        # Execute environment step
        obs_dict, rewards, terminations, truncations, info = self.env.step({
            "red": red_action,
            "blue": blue_action
        })
        
        # Extract Red team results
        red_obs = obs_dict["red"]
        red_reward = rewards["red"]
        red_terminated = terminations["red"]
        red_truncated = truncations["red"]
        
        # Update episode tracking
        self.episode_reward += red_reward
        self.episode_length += 1
        
        # Enhanced success detection
        target_reached = info.get('compromised_value', 0) >= self.env.target_value
        progress_threshold = 0.35
        significant_progress = info.get('compromised_value', 0) >= self.env.target_value * progress_threshold
        positive_reward = self.episode_reward > 0
        
        is_success = target_reached or significant_progress or positive_reward
        
        if red_terminated and is_success:
            self.success_count += 1
        
        # Enhanced info for analysis
        info.update({
            "blue_action": blue_action.tolist(),
            "blue_reward": rewards["blue"],
            "blue_policy": self.blue_policy,
            "episode_number": self.episode_count,
            "success_rate": self.success_count / self.episode_count if self.episode_count > 0 else 0,
            "is_success": is_success,
            "significant_progress": significant_progress,
            "target_value": self.env.target_value
        })
        
        return red_obs, red_reward, red_terminated, red_truncated, info
    
    def render(self, mode: str = "human") -> None:
        """
        Render environment state.
        
        Args:
            mode: Rendering mode
        """
        self.env.render(mode=mode)
        
        logger.info("Red Team Status:")
        logger.metric("Episode", self.episode_count)
        logger.metric("Success Rate", f"{self.success_count/max(1,self.episode_count):.2%}")
        logger.metric("Episode Reward", f"{self.episode_reward:.2f}")
        logger.config("Blue Policy", self.blue_policy)
    
    def close(self):
        """Close environment and cleanup resources."""
        pass

class BlueTeamEnv(gymnasium.Env):
    """
    Blue team (defender) environment wrapper for single-agent RL training.
    
    Wraps the multi-agent ThreatSim environment to provide a single-agent
    interface for Blue team training against configurable Red team opponents.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, 
                 scenario_path: str,
                 max_steps: int = 100,
                 red_strategy: str = "apt"):
        """
        Initialize Blue team environment.
        
        Args:
            scenario_path: Path to scenario YAML file
            max_steps: Maximum steps per episode
            red_strategy: Red team strategy ('apt', 'ransomware', 'insider')
        """
        super().__init__()
        
        logger.debug(f"Initializing Blue team environment with scenario: {scenario_path}")
        
        # Load configuration for heuristic opponents
        self.config_loader = ConfigLoader()
        self.heuristic_config = self.config_loader.unified_config.get("heuristic_opponents", {})
        
        # Initialize core environment
        self.env = ThreatSimEnv(yaml_path=scenario_path, max_steps=max_steps)
        self.red_strategy = red_strategy
        
        # Get Red team strategy configuration
        red_strategies = self.heuristic_config.get("red_team_strategies", {})
        self.red_config = red_strategies.get(red_strategy, {})
        
        # Configure action space (multi-discrete for Blue responses)
        self.action_space = MultiDiscrete([
            self.env.n_nodes,  # Target node
            self.env.n_tactics,  # Focus tactic
            5  # Response type (monitor, alert, block, isolate, remediate)
        ])
        
        # Observation space matches core environment
        self.observation_space = Box(
            low=0, high=1, shape=(self.env._obs_size,), dtype=np.float32
        )
        
        # Red team state tracking for strategy implementation
        self.red_state = {
            'phase': 'reconnaissance',
            'targets': [],
            'compromised': set(),
            'patience': 0,
            'stealth_level': self.red_config.get('stealth_priority', 0.5),
            'aggressiveness': self.red_config.get('aggressiveness', 0.6),
            'success_probability': self.red_config.get('success_probability_base', 0.5)
        }
        
        # Episode tracking
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        
        logger.success(f"Blue team environment initialized with {red_strategy} Red strategy")
    
    def _get_red_action(self) -> np.ndarray:
        """
        Generate Red team action based on configured strategy.
        
        Returns:
            Red team action array
        """
        if self.red_strategy == "apt":
            return self._get_red_action_apt()
        elif self.red_strategy == "ransomware":
            return self._get_red_action_ransomware()
        elif self.red_strategy == "insider":
            return self._get_red_action_insider()
        else:
            return self._get_red_action_apt()  # Default to APT
    
    def _get_red_action_apt(self) -> np.ndarray:
        """
        Generate APT (Advanced Persistent Threat) strategy action.
        
        Returns:
            Red team action for APT strategy
        """
        # Entry points for initial access
        entry_nodes = [i for i, n in enumerate(self.env.nodes) 
                      if n.id in self.env.entry_points]
        
        compromised = [i for i, n in enumerate(self.env.nodes) if n.compromised]
        
        if not compromised and entry_nodes:
            # Initial access phase
            node = np.random.choice(entry_nodes)
            return np.array([node, 1, 0])  # INITIAL_ACCESS
        elif compromised:
            # Post-compromise phase
            node = np.random.choice(compromised)
            
            # APT progression: persistence -> privilege escalation -> lateral movement
            node_obj = self.env.nodes[node]
            if node_obj.compromise_level < 2:
                return np.array([node, 4, 0])  # PRIVILEGE_ESCALATION
            else:
                return np.array([node, 8, 0])  # LATERAL_MOVEMENT
        else:
            # Fallback reconnaissance
            node = np.random.randint(self.env.n_nodes)
            return np.array([node, 0, 0])  # RECONNAISSANCE
    
    def _get_red_action_ransomware(self) -> np.ndarray:
        """
        Generate ransomware strategy action (fast, aggressive).
        
        Returns:
            Red team action for ransomware strategy
        """
        entry_nodes = [i for i, n in enumerate(self.env.nodes) 
                      if n.id in self.env.entry_points]
        compromised = [i for i, n in enumerate(self.env.nodes) if n.compromised]
        
        if not compromised and entry_nodes:
            node = np.random.choice(entry_nodes)
            return np.array([node, 1, 0])  # INITIAL_ACCESS
        elif compromised:
            # Immediate impact for ransomware
            node = np.random.choice(compromised)
            return np.array([node, 11, 0])  # IMPACT
        else:
            node = np.random.randint(self.env.n_nodes)
            return np.array([node, 1, 0])  # INITIAL_ACCESS
    
    def _get_red_action_insider(self) -> np.ndarray:
        """
        Generate insider threat strategy action (legitimate access advantage).
        
        Returns:
            Red team action for insider strategy
        """
        # Insider threats target high-value systems they have access to
        high_value_nodes = [i for i, n in enumerate(self.env.nodes) if n.value >= 3.0]
        
        if high_value_nodes:
            node = np.random.choice(high_value_nodes)
            # Insider has legitimate access
            if not self.env.nodes[node].compromised:
                return np.array([node, 1, 0])  # INITIAL_ACCESS
            else:
                return np.array([node, 9, 0])  # COLLECTION
        else:
            node = np.random.randint(self.env.n_nodes)
            return np.array([node, 9, 0])  # COLLECTION
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        obs_dict, info = self.env.reset(seed=seed, options=options)
        
        # Reset Red team state for strategy implementation
        self.red_state = {
            'phase': 'reconnaissance',
            'targets': [],
            'compromised': set(),
            'patience': 0,
            'stealth_level': self.red_config.get('stealth_priority', 0.5),
            'aggressiveness': self.red_config.get('aggressiveness', 0.6),
            'success_probability': self.red_config.get('success_probability_base', 0.5)
        }
        
        # Update episode tracking
        self.episode_count += 1
        self.episode_reward = 0.0
        self.episode_length = 0
        
        logger.debug(f"Blue team environment reset for episode {self.episode_count}")
        
        return obs_dict["blue"], info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.
        
        Args:
            action: Blue team action (multi-discrete)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Generate Red team action based on strategy
        red_action = self._get_red_action()
        
        # Execute environment step
        obs_dict, rewards, terminations, truncations, info = self.env.step({
            "red": red_action,
            "blue": action
        })
        
        # Extract Blue team results
        blue_obs = obs_dict["blue"]
        blue_reward = rewards["blue"]
        blue_terminated = terminations["blue"]
        blue_truncated = truncations["blue"]
        
        # Update episode tracking
        self.episode_reward += blue_reward
        self.episode_length += 1
        
        # Enhanced info for analysis
        info.update({
            "red_action": red_action.tolist(),
            "red_reward": rewards["red"],
            "red_strategy": self.red_strategy,
            "episode_number": self.episode_count
        })
        
        return blue_obs, blue_reward, blue_terminated, blue_truncated, info
    
    def render(self, mode: str = "human") -> None:
        """
        Render environment state.
        
        Args:
            mode: Rendering mode
        """
        self.env.render(mode=mode)
        
        logger.info("Blue Team Status:")
        logger.metric("Episode", self.episode_count)
        logger.metric("Episode Reward", f"{self.episode_reward:.2f}")
        logger.config("Red Strategy", self.red_strategy)
    
    def close(self):
        """Close environment and cleanup resources."""
        pass

# Convenience factory functions
def make_red_env(scenario_path: str, **kwargs) -> RedTeamEnv:
    """
    Create Red team training environment.
    
    Args:
        scenario_path: Path to scenario YAML file
        **kwargs: Additional environment parameters
        
    Returns:
        Configured Red team environment
    """
    logger.debug(f"Creating Red team environment for scenario: {scenario_path}")
    return RedTeamEnv(scenario_path=scenario_path, **kwargs)

def make_blue_env(scenario_path: str, **kwargs) -> BlueTeamEnv:
    """
    Create Blue team training environment.
    
    Args:
        scenario_path: Path to scenario YAML file
        **kwargs: Additional environment parameters
        
    Returns:
        Configured Blue team environment
    """
    logger.debug(f"Creating Blue team environment for scenario: {scenario_path}")
    return BlueTeamEnv(scenario_path=scenario_path, **kwargs)
