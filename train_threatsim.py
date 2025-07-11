# File: train_threatsim.py
# Description: ThreatSim training system with integrated visualization and balance validation
# Purpose: Complete training pipeline with analysis and visualization
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import os
import sys
import csv
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import random
from dataclasses import dataclass
from scipy import stats
import warnings
from pathlib import Path

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import importlib
import sys
from types import ModuleType
def purge_package(pkg_root: str) -> None:
    """Remove pkg_root and all its sub-modules from sys.modules."""
    for name in list(sys.modules):          # list() to copy keys during iteration
        if name == pkg_root or name.startswith(pkg_root + "."):
            del sys.modules[name]

def fresh_import(pkg_root: str) -> ModuleType:
    """Invalidate caches, purge, and return a *new* top-level module object."""
    importlib.invalidate_caches()           # forget filesystem path caches
    purge_package(pkg_root)
    return importlib.import_module(pkg_root)

# --- Usage ---------------------------------------------------------------
threatsim = fresh_import("threatsim")

# ThreatSim imports
from threatsim.envs.wrappers import make_red_env, make_blue_env
from threatsim.utils.logger import logger
from threatsim.utils.balance_validator import BalanceValidator
from threatsim.utils.scenario_manager import ScenarioManager
from threatsim.utils.visualization import ThreatSimVisualizer
from threatsim.utils.output_generator import OutputGenerator

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

class ConfigurationManager:
    """Configuration manager for ThreatSim training system - FIXED VERSION."""
    
    def __init__(self, config_path: str = "data/config.json"):
        self.config_path = config_path
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load configuration or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "system_config": {
                "version": "0.3.2",
                "description": "ThreatSim Training System"
            },
            "training_configurations": {
                "training_modes": {
                    "quick": {
                        "description": "Quick testing (5k steps, 1 seed)",
                        "timesteps": 5000,
                        "seeds": [42],
                        "max_steps_per_episode": 20
                    },
                    "standard": {
                        "description": "Standard training (20k steps, 3 seeds)",
                        "timesteps": 20000,
                        "seeds": [42, 123, 456],
                        "max_steps_per_episode": 50
                    },
                    "research": {
                        "description": "Research-grade training (100k steps, 5 seeds)",
                        "timesteps": 100000,
                        "seeds": [42, 123, 456, 789, 999],
                        "max_steps_per_episode": 100
                    }
                },
                "ppo_hyperparameters": {
                    "learning_rate": 3e-4,
                    "n_steps": 256,
                    "batch_size": 64,
                    "n_epochs": 4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5
                }
            }
        }
    
    def get_training_modes(self) -> Dict[str, Any]:
        return self.config.get("training_configurations", {}).get("training_modes", {})
    
    def get_ppo_hyperparameters(self) -> Dict[str, Any]:
        return self.config.get("training_configurations", {}).get("ppo_hyperparameters", {})
    
    def get_system_version(self) -> str:
        return self.config.get("system_config", {}).get("version", "0.3.2")
    
    def get_balance_validator(self, scenario_info: Optional[Dict] = None) -> BalanceValidator:
        """
        Get balance validator with scenario-specific target support.
        
        FIXED: Now supports scenario-specific targets from scenario_info.
        
        Args:
            scenario_info: Optional scenario information with expected_red_win_rate
            
        Returns:
            BalanceValidator configured with appropriate target
        """
        # Use scenario-specific target if available
        if scenario_info and 'expected_red_win_rate' in scenario_info:
            target_rate = scenario_info['expected_red_win_rate']
            logger.info(f"Using scenario-specific target: {target_rate:.1%}")
        else:
            target_rate = 0.30  # Fallback default
            logger.info(f"Using default target: {target_rate:.1%}")
        
        return BalanceValidator(
            target_red_win_rate=target_rate,
            tolerance=0.15,
            min_episodes=100
        )

class TrainingOrchestrator:
    """Training orchestration system - FIXED VERSION."""
    
    def __init__(self, config_manager, scenario_manager):
        """Initialize training orchestrator."""
        self.config_manager = config_manager
        self.scenario_manager = scenario_manager
        # NOTE: Don't create balance_validator here - create it per-scenario
        
        logger.debug("Training orchestrator initialized with scenario-specific balance validation support")
    
    def execute_experiment(self, mode: str, scenario: str) -> Dict[str, Any]:
        """Execute complete training experiment with scenario-specific validation."""
        # Setup experiment
        results = self._setup_experiment(mode, scenario)
        if 'error' in results:
            return results
        
        experiment_config = results['experiment_config']
        scenario_info = results['scenario_info']
        
        # FIXED: Create scenario-specific balance validator
        self.balance_validator = self.config_manager.get_balance_validator(scenario_info)
        
        # Create output directories
        output_dir = experiment_config['output_dir']
        
        # Initialize visualization system
        visualizer = ThreatSimVisualizer(output_dir, style='publication')
        
        # Initialize output generator
        output_generator = OutputGenerator(output_dir)
        
        # Execute training trials
        results = self._execute_training_trials(experiment_config, scenario_info)
        
        # Generate statistical analysis
        results['statistical_analysis'] = self._enhance_statistical_analysis(
            results['trials'], scenario_info
        )
        
        # FIXED: Pass complete results with scenario info to balance validation
        balance_report = self.balance_validator.validate_scenario_balance(
            results, scenario_info['name']
        )
        results['balance_validation'] = balance_report
        
        # Generate comprehensive visualizations
        try:
            visualizer.create_comprehensive_dashboard(results)
            visualizer.create_publication_figures(results)
            logger.success("Comprehensive visualizations generated")
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        # Generate all output files
        try:
            output_generator.generate_complete_output(results)
            logger.success("Complete output files generated")
        except Exception as e:
            logger.warning(f"Output generation failed: {e}")
        
        # Save results with proper serialization
        self._save_experiment_results(results)

        # Generate summary report - DIRECT IMPLEMENTATION
        def create_summary_report_directly(results):
            """Create summary report directly in training script"""
            summary = "ThreatSim Visualization Summary\n"
            summary += "=" * 50 + "\n\n"
            
            # Basic info
            trials = results.get('trials', {})
            summary += f"Training completed with {len(trials)} seeds\n"
            
            # Performance data
            red_wins = []
            blue_wins = []
            
            for seed, trial_data in trials.items():
                if trial_data.get('red', {}).get('success', False):
                    red_eval = trial_data['red']['evaluation_results']
                    red_wins.append(red_eval.get('win_rate', 0))
                
                if trial_data.get('blue', {}).get('success', False):
                    blue_eval = trial_data['blue']['evaluation_results']
                    blue_wins.append(blue_eval.get('win_rate', 0))
            
            # Performance summary
            if red_wins:
                red_mean = np.mean(red_wins)
                red_std = np.std(red_wins)
                summary += f"Red Team: {red_mean:.1f}% ± {red_std:.1f}% win rate\n"
            else:
                summary += "Red Team: No data\n"
                
            if blue_wins:
                blue_mean = np.mean(blue_wins)
                blue_std = np.std(blue_wins)
                summary += f"Blue Team: {blue_mean:.1f}% ± {blue_std:.1f}% win rate\n"
            else:
                summary += "Blue Team: No data\n"
            
            # Balance analysis with scenario-specific target
            balance_validation = results.get('balance_validation', {})
            balance_metrics = balance_validation.get('balance_metrics', {})
            
            if balance_metrics:
                target_rate = balance_metrics.get('target_red_win_rate', 0) * 100
                actual_rate = balance_metrics.get('red_win_rate', 0) * 100
                deviation = abs(actual_rate - target_rate)
                
                summary += "Balance Analysis:\n"
                summary += f"  Target: {target_rate:.1f}%, Actual: {actual_rate:.1f}%\n"
                summary += f"  Deviation: {deviation:.1f}%\n"
                summary += f"  Status: {'Balanced' if deviation < 10 else 'Needs Adjustment'}\n"
                
                # Add target source info
                target_source = balance_validation.get('target_source', 'unknown')
                summary += f"  Target Source: {target_source}\n"
            else:
                summary += "Balance Analysis: No data available\n"
            
            # Statistical analysis
            stats_analysis = results.get('statistical_analysis', {})
            statistical_tests = stats_analysis.get('statistical_tests', {})
            
            if statistical_tests and 'p_value' in statistical_tests:
                p_value = statistical_tests.get('p_value', 1.0)
                effect_size = statistical_tests.get('effect_size', 0.0)
                
                summary += "Statistical Analysis:\n"
                summary += f"  P-value: {p_value:.4f}\n"
                summary += f"  Effect Size: {effect_size:.3f}\n"
                summary += f"  Significance: {'Yes' if p_value < 0.05 else 'No'}\n"
            else:
                summary += "Statistical Analysis: No data available\n"
            
            # Scenario-specific information
            scenario_info = results.get('scenario_info', {})
            if scenario_info:
                summary += "Scenario Information:\n"
                summary += f"  Name: {scenario_info.get('name', 'Unknown')}\n"
                summary += f"  Difficulty: {scenario_info.get('difficulty', 'Unknown')}\n"
                if 'expected_red_win_rate' in scenario_info:
                    expected_rate = scenario_info['expected_red_win_rate'] * 100
                    summary += f"  Expected Red Win Rate: {expected_rate:.1f}%\n"
            
            # Visualization files
            summary += "Visualization Files Created:\n"
            plots_dir = output_dir / "plots"
            if plots_dir.exists():
                plot_files = sorted(plots_dir.glob('*.png'))
                if plot_files:
                    for plot_file in plot_files:
                        summary += f"  - {plot_file.name}\n"
                else:
                    summary += "  - No plot files found\n"
            else:
                summary += "  - Plots directory not found\n"
            
            return summary
        
        # Generate summary using direct function
        try:
            logger.info("Generating summary report directly...")
            summary_report = create_summary_report_directly(results)
            
            summary_path = output_dir / 'summary_report.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            logger.success(f"Summary report saved: {summary_path}")
            logger.info(f"Summary length: {len(summary_report)} characters")
            
        except Exception as e:
            logger.error(f"Direct summary generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return results
    
    def _setup_experiment(self, mode: str, scenario: str) -> Dict[str, Any]:
        """Setup and validate experiment configuration - FIXED VERSION."""
        # Validate inputs
        training_modes = self.config_manager.get_training_modes()
        if mode not in training_modes:
            available_modes = list(training_modes.keys())
            return {'error': f"Unknown mode: {mode}. Available: {available_modes}"}
        
        scenario_info = self.scenario_manager.get_scenario_info(scenario)
        if scenario_info is None:
            available_scenarios = [key for key, _ in self.scenario_manager.get_scenario_list()]
            return {'error': f"Unknown scenario: {scenario}. Available: {available_scenarios}"}
        
        # Create experiment configuration
        mode_config = training_modes[mode]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        experiment_config = {
            'name': f"threatsim_{scenario}_{mode}_{timestamp}",
            'mode': mode,
            'scenario_name': scenario,
            'scenario_file': scenario_info['file'],
            'output_dir': Path(f"outputs/{timestamp}_{scenario}_{mode}/"),
            'model_dir': Path(f"outputs/{timestamp}_{scenario}_{mode}/models/"),
            # FIXED: Include scenario-specific expected win rate in experiment config
            'expected_red_win_rate': scenario_info['expected_red_win_rate'],
            'target_value': scenario_info['target_value'],
            **mode_config
        }
        
        # Create output directories
        experiment_config['output_dir'].mkdir(parents=True, exist_ok=True)
        experiment_config['model_dir'].mkdir(parents=True, exist_ok=True)
        
        # Log experiment start with scenario-specific info
        logger.header("ThreatSim Training Experiment")
        logger.config("Scenario", f"{scenario_info['name']} ({scenario_info['difficulty']})")
        logger.config("Mode", f"{mode_config['description']}")
        logger.config("Seeds", len(mode_config['seeds']))
        logger.config("Expected Red Win Rate", f"{scenario_info['expected_red_win_rate']*100:.1f}%")
        logger.config("Output Directory", experiment_config['output_dir'])
        logger.config("Visualization", "enabled")
        logger.config("Balance Validation", "enabled with scenario-specific targets")
        
        return {
            'experiment_config': experiment_config,
            'scenario_info': scenario_info,
            'mode_config': mode_config
        }
    
    def _execute_training_trials(self, experiment_config: Dict[str, Any], 
                                scenario_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training trials with monitoring."""
        results = {
            'experiment_config': experiment_config,
            'scenario_info': scenario_info,
            'trials': {}
        }
        
        mode_config = self.config_manager.get_training_modes()[experiment_config['mode']]
        
        for i, seed in enumerate(mode_config['seeds'], 1):
            logger.info(f"Executing trial {i}/{len(mode_config['seeds'])} (seed {seed})")
            
            trial_config = experiment_config.copy()
            trial_config['seed'] = seed
            
            # Train both agents
            red_result = self.train_agent("red", trial_config)
            blue_result = self.train_agent("blue", trial_config)
            
            results['trials'][seed] = {
                'red': red_result,
                'blue': blue_result
            }
            
            # Log trial completion
            if red_result['success'] and blue_result['success']:
                red_eval = red_result['evaluation_results']
                blue_eval = blue_result['evaluation_results']
                logger.success(f"Trial {i} completed successfully")
                logger.metric("Red Win Rate", f"{red_eval['win_rate']:.1f}%")
                logger.metric("Blue Win Rate", f"{blue_eval['win_rate']:.1f}%")
                logger.metric("Red Avg Reward", f"{red_eval['mean_reward']:.2f}")
                logger.metric("Blue Avg Reward", f"{blue_eval['mean_reward']:.2f}")
            else:
                logger.error(f"Trial {i} failed")
        
        return results
    
    def train_agent(self, agent_type: str, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train single agent with evaluation and monitoring."""
        logger.info(f"Training {agent_type} agent")
        
        try:
            # Create environment
            if agent_type == "red":
                env = make_red_env(
                    scenario_path=experiment_config['scenario_file'],
                    max_steps=experiment_config['max_steps_per_episode'],
                    blue_policy="heuristic"
                )
            else:
                env = make_blue_env(
                    scenario_path=experiment_config['scenario_file'],
                    max_steps=experiment_config['max_steps_per_episode'],
                    red_strategy="apt"
                )
            
            # Vectorize environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Create model
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=0,
                seed=experiment_config['seed'],
                **self.config_manager.get_ppo_hyperparameters()
            )
            
            # Train model
            model.learn(total_timesteps=experiment_config['timesteps'])
            
            # Evaluate agent
            evaluation_results = self._evaluate_agent(model, vec_env, env, 100)
            
            # Save model
            model_path = experiment_config['model_dir'] / f"{agent_type}_agent_{experiment_config['seed']}.zip"
            model.save(str(model_path))
            
            return {
                'success': True,
                'model_path': str(model_path),
                'evaluation_results': evaluation_results,
                'training_data': {
                    'episode_count': 100,
                    'timesteps': experiment_config['timesteps'],
                    'seed': experiment_config['seed']
                }
            }
            
        except Exception as e:
            logger.error(f"Training failed for {agent_type} agent: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_agent(self, model, vec_env, base_env, num_episodes: int) -> Dict[str, Any]:
        """Agent evaluation with detailed win condition tracking."""
        episode_rewards = []
        episode_lengths = []
        wins = 0
        total_episodes = 0
        
        # Performance tracking
        compromise_progress = []
        detection_events = []
        prevention_events = []
        win_condition_details = []
        
        logger.info(f"Evaluation over {num_episodes} episodes")
        
        for episode in range(num_episodes):
            try:
                obs = vec_env.reset()
                total_reward = 0
                episode_length = 0
                max_compromise_progress = 0
                episode_detections = 0
                episode_preventions = 0
                
                # Episode execution
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = vec_env.step(action)
                    
                    total_reward += reward[0]
                    episode_length += 1
                    
                    # Episode information extraction
                    if info and len(info) > 0:
                        episode_info = info[0]
                        
                        # Track compromise progress
                        progress = episode_info.get('progress_ratio', 0)
                        max_compromise_progress = max(max_compromise_progress, progress)
                        
                        # Track security events
                        episode_detections += episode_info.get('detected_ttps', 0)
                        episode_preventions += episode_info.get('prevented_ttps', 0)
                        
                        # Check for episode termination
                        if done[0]:
                            # Win condition detection
                            red_win = episode_info.get('red_win', False)
                            blue_win = episode_info.get('blue_win', False)
                            target_reached = episode_info.get('target_reached', False)
                            
                            # Win condition validation
                            win_condition_validation = episode_info.get('win_condition_validation', {})
                            mathematically_consistent = win_condition_validation.get('mathematical_consistency', True)
                            
                            # Success criteria based on agent type
                            is_success = False
                            if hasattr(base_env, 'blue_policy'):
                                # Red team environment
                                is_success = red_win and mathematically_consistent
                            elif hasattr(base_env, 'red_strategy'):
                                # Blue team environment
                                is_success = blue_win and mathematically_consistent
                            else:
                                # Fallback
                                is_success = (
                                    episode_info.get('is_success', False) and
                                    mathematically_consistent
                                )
                            
                            # Record win condition details
                            win_condition_details.append({
                                'episode': episode,
                                'red_win': red_win,
                                'blue_win': blue_win,
                                'target_reached': target_reached,
                                'progress_ratio': max_compromise_progress,
                                'mathematically_consistent': mathematically_consistent,
                                'is_success': is_success
                            })
                            
                            if is_success:
                                wins += 1
                            
                            break
                    
                    # Safety check for maximum episode length
                    if episode_length >= 200:
                        logger.warning(f"Episode {episode} exceeded maximum length, terminating")
                        break
                
                # Record episode results
                episode_rewards.append(total_reward)
                episode_lengths.append(episode_length)
                compromise_progress.append(max_compromise_progress)
                detection_events.append(episode_detections)
                prevention_events.append(episode_preventions)
                total_episodes += 1
                
                # Progress logging
                if (episode + 1) % 25 == 0:
                    logger.debug(f"Evaluation progress: {episode + 1}/{num_episodes} episodes completed")
                    
            except Exception as e:
                logger.warning(f"Episode {episode} failed: {e}")
                continue
        
        # Calculate comprehensive statistics
        if total_episodes == 0:
            logger.error("No episodes completed successfully")
            return {
                'success': False,
                'error': 'No episodes completed successfully'
            }
        
        # Performance metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        win_rate = (wins / total_episodes) * 100
        
        # Advanced performance metrics
        mean_compromise_progress = np.mean(compromise_progress)
        mean_detections = np.mean(detection_events)
        mean_preventions = np.mean(prevention_events)
        
        # Win condition analysis
        win_condition_consistency = np.mean([
            wd['mathematically_consistent'] for wd in win_condition_details
        ])
        
        # Performance categorization
        if win_rate > 70:
            performance_category = "excellent"
        elif win_rate > 50:
            performance_category = "good"
        elif win_rate > 30:
            performance_category = "moderate"
        elif win_rate > 10:
            performance_category = "poor"
        else:
            performance_category = "very_poor"
        
        evaluation_results = {
            'success': True,
            'num_episodes': total_episodes,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'win_rate': float(win_rate),
            'wins': int(wins),
            'performance_category': performance_category,
            'advanced_metrics': {
                'mean_compromise_progress': float(mean_compromise_progress),
                'mean_detections_per_episode': float(mean_detections),
                'mean_preventions_per_episode': float(mean_preventions),
                'reward_consistency': float(std_reward / abs(mean_reward)) if mean_reward != 0 else float('inf'),
                'success_rate': float(wins / total_episodes),
                'win_condition_consistency': float(win_condition_consistency)
            },
            'statistical_data': {
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_lengths': [int(l) for l in episode_lengths],
                'compromise_progress': [float(p) for p in compromise_progress],
                'win_condition_details': win_condition_details
            }
        }
        
        # Log evaluation summary
        logger.info(f"Evaluation completed: {win_rate:.1f}% win rate, {mean_reward:.2f} avg reward")
        logger.debug(f"Performance category: {performance_category}")
        logger.debug(f"Win condition consistency: {win_condition_consistency:.2f}")
        logger.debug(f"Average compromise progress: {mean_compromise_progress:.2f}")
        
        return evaluation_results
    
    def _enhance_statistical_analysis(self, trials: Dict[str, Any], 
                                    scenario_info: Dict[str, Any]) -> Dict[str, Any]:
        """Statistical analysis with comprehensive validation."""
        # Extract successful trials
        successful_trials = []
        red_metrics = []
        blue_metrics = []
        
        for seed, trial in trials.items():
            if trial['red']['success'] and trial['blue']['success']:
                successful_trials.append(seed)
                red_metrics.append(trial['red']['evaluation_results'])
                blue_metrics.append(trial['blue']['evaluation_results'])
        
        if not successful_trials:
            return {'error': 'no_successful_trials'}
        
        # Extract performance metrics
        red_win_rates = [m['win_rate'] for m in red_metrics]
        blue_win_rates = [m['win_rate'] for m in blue_metrics]
        red_rewards = [m['mean_reward'] for m in red_metrics]
        blue_rewards = [m['mean_reward'] for m in blue_metrics]
        
        # Win condition consistency analysis
        red_consistencies = [m['advanced_metrics']['win_condition_consistency'] for m in red_metrics]
        blue_consistencies = [m['advanced_metrics']['win_condition_consistency'] for m in blue_metrics]
        
        # Calculate aggregate statistics
        red_stats = {
            'mean_win_rate': float(np.mean(red_win_rates)),
            'std_win_rate': float(np.std(red_win_rates)),
            'mean_reward': float(np.mean(red_rewards)),
            'std_reward': float(np.std(red_rewards)),
            'performance_consistency': float(np.std(red_win_rates) / np.mean(red_win_rates)) if np.mean(red_win_rates) > 0 else float('inf'),
            'win_condition_consistency': float(np.mean(red_consistencies))
        }
        
        blue_stats = {
            'mean_win_rate': float(np.mean(blue_win_rates)),
            'std_win_rate': float(np.std(blue_win_rates)),
            'mean_reward': float(np.mean(blue_rewards)),
            'std_reward': float(np.std(blue_rewards)),
            'performance_consistency': float(np.std(blue_win_rates) / np.mean(blue_win_rates)) if np.mean(blue_win_rates) > 0 else float('inf'),
            'win_condition_consistency': float(np.mean(blue_consistencies))
        }
        
        # Bootstrap confidence intervals
        red_ci = self._calculate_bootstrap_confidence_interval(red_win_rates)
        blue_ci = self._calculate_bootstrap_confidence_interval(blue_win_rates)
        
        # Statistical significance testing
        statistical_tests = {}
        if len(red_rewards) > 1 and len(blue_rewards) > 1:
            try:
                # Win rate comparison
                win_rate_t_stat, win_rate_p_value = stats.ttest_ind(red_win_rates, blue_win_rates)
                win_rate_effect_size = self._calculate_effect_size_cohens_d(red_win_rates, blue_win_rates)
                
                # Reward comparison
                reward_t_stat, reward_p_value = stats.ttest_ind(red_rewards, blue_rewards)
                reward_effect_size = self._calculate_effect_size_cohens_d(red_rewards, blue_rewards)
                
                # Statistical power
                sample_size = len(successful_trials)
                win_rate_power = self._calculate_power_analysis(sample_size, win_rate_effect_size)
                
                statistical_tests = {
                    'win_rate_t_statistic': float(win_rate_t_stat),
                    'win_rate_p_value': float(win_rate_p_value),
                    'win_rate_effect_size': float(win_rate_effect_size),
                    'reward_t_statistic': float(reward_t_stat),
                    'reward_p_value': float(reward_p_value),
                    'reward_effect_size': float(reward_effect_size),
                    'significant_difference': bool(win_rate_p_value < 0.05),
                    'statistical_power': float(win_rate_power),
                    'effect_size_interpretation': self._interpret_effect_size(win_rate_effect_size)
                }
            except Exception as e:
                logger.warning(f"Statistical testing failed: {e}")
                statistical_tests = {
                    'error': 'statistical_testing_failed',
                    'significant_difference': False,
                    'statistical_power': 0.0
                }
        else:
            statistical_tests = {
                'insufficient_data': True,
                'significant_difference': False,
                'statistical_power': 0.0
            }
        
        return {
            'trial_summary': {
                'total_trials': len(trials),
                'successful_trials': len(successful_trials),
                'success_rate': float(len(successful_trials) / len(trials))
            },
            'team_performance': {
                'red_team': {
                    **red_stats,
                    'confidence_interval': red_ci
                },
                'blue_team': {
                    **blue_stats,
                    'confidence_interval': blue_ci
                }
            },
            'statistical_tests': statistical_tests,
            'sample_size_analysis': {
                'current_sample_size': len(successful_trials),
                'recommended_sample_size': max(5, len(successful_trials)),
                'sample_size_adequate': len(successful_trials) >= 3,
                'confidence_level': 0.95 if len(successful_trials) >= 5 else 0.80
            },
            'metrics': {
                'overall_win_condition_consistency': float(np.mean(red_consistencies + blue_consistencies)),
                'balance_deviation': abs(np.mean(red_win_rates) - scenario_info.get('expected_red_win_rate', 0.3) * 100),
                'competitive_balance': abs(np.mean(red_win_rates) - np.mean(blue_win_rates)) / 100.0
            }
        }
    
    def _calculate_bootstrap_confidence_interval(self, data: List[float], 
                                              confidence_level: float = 0.95,
                                              n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a dataset."""
        if len(data) < 2:
            return (0.0, 100.0)
        
        data_array = np.array(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def _calculate_effect_size_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size between two groups."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return float(cohens_d)
    
    def _calculate_power_analysis(self, sample_size: int, effect_size: float, alpha: float = 0.05) -> float:
        """Calculate statistical power for given parameters."""
        if sample_size < 2:
            return 0.0
        
        try:
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = abs(effect_size) * np.sqrt(sample_size/2) - z_alpha
            power = 1 - stats.norm.cdf(z_beta)
            return float(max(0.0, min(1.0, power)))
        except Exception:
            return 0.0
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _prepare_for_json_serialization(self, obj: Any) -> Any:
        """Prepare object for JSON serialization by converting NumPy types and Path objects."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json_serialization(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._prepare_for_json_serialization(item) for item in obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    def _save_experiment_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results to file with proper NumPy type handling."""
        output_dir = results['experiment_config']['output_dir']
        results_file = output_dir / 'experiment_results.json'
        
        # Add metadata
        results['metadata'] = {
            'system_version': self.config_manager.get_system_version(),
            'generation_timestamp': datetime.now().isoformat(),
            'configuration_file': self.config_manager.config_path,
            'numpy_types_converted': True,
            'features': {
                'visualization_enabled': True,
                'balance_validation_enabled': True,
                'win_condition_tracking': True
            }
        }
        
        try:
            # Prepare data for JSON serialization
            json_results = self._prepare_for_json_serialization(results)
            
            # Save with custom encoder
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.success(f"Experiment results saved: {results_file}")
            
            # Create summary file
            summary_file = output_dir / 'experiment_summary.json'
            summary = self._create_experiment_summary(results)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Experiment summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
    
    def _create_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a clean summary of experiment results."""
        statistical_analysis = results.get('statistical_analysis', {})
        balance_validation = results.get('balance_validation', {})
        experiment_config = results.get('experiment_config', {})
        
        team_performance = statistical_analysis.get('team_performance', {})
        red_stats = team_performance.get('red_team', {})
        blue_stats = team_performance.get('blue_team', {})
        
        balance_metrics = balance_validation.get('balance_metrics', {})
        metrics = statistical_analysis.get('metrics', {})
        
        return {
            'experiment_info': {
                'scenario_name': experiment_config.get('scenario_name', 'unknown'),
                'mode': experiment_config.get('mode', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'total_trials': statistical_analysis.get('trial_summary', {}).get('total_trials', 0),
                'successful_trials': statistical_analysis.get('trial_summary', {}).get('successful_trials', 0)
            },
            'performance_summary': {
                'red_team': {
                    'win_rate': float(red_stats.get('mean_win_rate', 0)),
                    'avg_reward': float(red_stats.get('mean_reward', 0)),
                    'confidence_interval': red_stats.get('confidence_interval', (0, 0)),
                    'win_condition_consistency': float(red_stats.get('win_condition_consistency', 0))
                },
                'blue_team': {
                    'win_rate': float(blue_stats.get('mean_win_rate', 0)),
                    'avg_reward': float(blue_stats.get('mean_reward', 0)),
                    'confidence_interval': blue_stats.get('confidence_interval', (0, 0)),
                    'win_condition_consistency': float(blue_stats.get('win_condition_consistency', 0))
                }
            },
            'balance_assessment': {
                'target_red_win_rate': float(balance_metrics.get('target_red_win_rate', 0)),
                'actual_red_win_rate': float(balance_metrics.get('red_win_rate', 0)),
                'deviation': float(balance_metrics.get('deviation', 0)),
                'within_tolerance': bool(balance_metrics.get('within_tolerance', False)),
                'balance_category': balance_validation.get('balance_assessment', {}).get('balance_category', 'unknown'),
                'competitive_balance': float(metrics.get('competitive_balance', 0))
            },
            'recommendations': [
                {
                    'priority': rec.get('priority', 'unknown'),
                    'action': rec.get('action', 'unknown'),
                    'description': rec.get('description', 'unknown')
                }
                for rec in balance_validation.get('recommendations', [])
            ],
            'visualization_files': {
                'training_overview': 'plots/training_overview.png',
                'performance_analysis': 'plots/performance_analysis.png',
                'balance_analysis': 'plots/balance_analysis.png',
                'statistical_dashboard': 'plots/statistical_dashboard.png',
                'publication_figure': 'plots/publication_figure.png'
            }
        }

class ThreatSimCLI:
    """Command-line interface for ThreatSim training system."""
    
    def __init__(self):
        """Initialize CLI with configuration and scenario management."""
        try:
            self.config_manager = ConfigurationManager()
            self.scenario_manager = ScenarioManager()
            self.orchestrator = TrainingOrchestrator(self.config_manager, self.scenario_manager)
            
        except Exception as e:
            logger.error(f"Failed to initialize ThreatSim: {e}")
            sys.exit(1)
    
    def run(self) -> None:
        """Execute CLI interface."""
        logger.header("ThreatSim Training System")
        logger.config("System Version", self.config_manager.get_system_version())
        logger.config("Configuration", self.config_manager.config_path)
        logger.config("Balance Validation", "enabled")
        logger.config("Visualization", "enabled")
        
        try:
            self._display_available_options()
            mode, scenario = self._get_user_selections()
            self._execute_user_choice(mode, scenario)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)
    
    def _display_available_options(self) -> None:
        """Display available scenarios and training modes."""
        scenarios = self.scenario_manager.get_scenario_list()
        training_modes = self.config_manager.get_training_modes()
        
        if not scenarios:
            logger.error("No scenarios available. Please check the scenarios/ directory.")
            sys.exit(1)
        
        logger.info("Available Scenarios:")
        for i, (key, scenario) in enumerate(scenarios, 1):
            description = scenario['name'][:50] + "..." if len(scenario['name']) > 50 else scenario['name']
            expected_rate = scenario.get('expected_red_win_rate', 0.30)
            logger.config(f"{i}", f"{description} ({scenario['difficulty']}) - Expected Red: {expected_rate:.1%}")
        
        logger.info("Available Training Modes:")
        for i, (key, mode) in enumerate(training_modes.items(), 1):
            seeds_count = len(mode.get('seeds', []))
            timesteps = mode.get('timesteps', 0)
            logger.config(f"{i}", f"{key.title()} - {mode['description']} ({timesteps:,} timesteps, {seeds_count} seed{'s' if seeds_count != 1 else ''})")
        
        # Additional options
        next_option = len(training_modes) + 1
        logger.config(f"{next_option}", "validate - Validate scenario balance")
        logger.config(f"{next_option + 1}", "config - Show configuration details")
        logger.config(f"{next_option + 2}", "visualize - Create visualization from existing results")
    
    def _get_user_selections(self) -> Tuple[str, str]:
        """Get and validate user selections."""
        scenarios = self.scenario_manager.get_scenario_list()
        training_modes = self.config_manager.get_training_modes()
        
        # Get scenario selection
        scenario_key = self._get_scenario_selection(scenarios)
        
        # Get mode selection
        mode_key = self._get_mode_selection(training_modes, scenario_key)
        
        return mode_key, scenario_key
    
    def _get_scenario_selection(self, scenarios: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Get and validate scenario selection from user."""
        while True:
            try:
                choice = input(f"\nSelect scenario (1-{len(scenarios)}): ").strip()
                scenario_idx = int(choice) - 1
                
                if 0 <= scenario_idx < len(scenarios):
                    scenario_key = scenarios[scenario_idx][0]
                    selected_scenario = scenarios[scenario_idx][1]
                    
                    logger.config("Selected Scenario", selected_scenario['name'])
                    logger.config("Difficulty", selected_scenario['difficulty'])
                    logger.config("Expected Red Win Rate", f"{selected_scenario['expected_red_win_rate']*100:.1f}%")
                    
                    return scenario_key
                else:
                    logger.warning(f"Invalid choice. Please enter 1-{len(scenarios)}")
                    
            except ValueError:
                logger.warning("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                raise
    
    def _get_mode_selection(self, training_modes: Dict[str, Any], scenario_key: str) -> str:
        """Get and validate training mode selection."""
        num_training_modes = len(training_modes)
        validation_option = num_training_modes + 1
        config_option = num_training_modes + 2
        visualize_option = num_training_modes + 3
        
        while True:
            try:
                choice = input(f"\nSelect training mode (1-{num_training_modes}, {validation_option} for validate, {config_option} for config, {visualize_option} for visualize): ").strip()
                
                # Handle special options
                if choice == str(validation_option):
                    self._handle_scenario_validation(scenario_key)
                    continue
                
                if choice == str(config_option):
                    self._display_configuration_details()
                    continue
                
                if choice == str(visualize_option):
                    self._handle_visualization()
                    continue
                
                # Handle training mode selection
                mode_idx = int(choice) - 1
                mode_keys = list(training_modes.keys())
                
                if 0 <= mode_idx < len(mode_keys):
                    mode_key = mode_keys[mode_idx]
                    selected_mode = training_modes[mode_key]
                    
                    logger.config("Selected Mode", selected_mode['description'])
                    logger.config("Timesteps", f"{selected_mode['timesteps']:,}")
                    logger.config("Seeds", len(selected_mode['seeds']))
                    
                    return mode_key
                else:
                    logger.warning(f"Invalid choice. Please enter 1-{num_training_modes}, {validation_option}, {config_option}, or {visualize_option}")
                    
            except ValueError:
                logger.warning("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                raise
    
    def _handle_scenario_validation(self, scenario_key: str) -> None:
        """Handle scenario validation request."""
        logger.info("Validating scenario...")
        
        try:
            scenario_info = self.scenario_manager.get_scenario_info(scenario_key)
            if scenario_info is None:
                logger.error("Scenario not found")
                return
            
            scenario_path = scenario_info['file']
            validation_result = self.scenario_manager.validate_scenario(
                scenario_path, {'exploitability_range': [0.1, 0.95], 'detection_range': [0.05, 0.98], 'target_percentage_range': [15.0, 80.0]}
            )
            
            logger.header("Scenario Validation Results")
            
            if validation_result['valid']:
                logger.success("Scenario is valid")
                
                stats = validation_result['statistics']
                logger.metric("Total Nodes", stats['total_nodes'])
                logger.metric("Target Percentage", f"{stats['target_percentage']:.1f}%")
                logger.metric("Average Exploitability", f"{stats['avg_exploitability']:.2f}")
                logger.metric("Average Detection", f"{stats['avg_detection_prob']:.2f}")
                logger.metric("Total Vulnerabilities", stats['total_vulnerabilities'])
                
                if validation_result.get('warnings'):
                    logger.warning("Validation Warnings:")
                    for warning in validation_result['warnings']:
                        logger.warning(f"  {warning}")
            else:
                logger.error("Scenario validation failed")
                if 'error' in validation_result:
                    logger.error(f"Error: {validation_result['error']}")
                if 'issues' in validation_result:
                    logger.error("Issues found:")
                    for issue in validation_result['issues']:
                        logger.error(f"  {issue}")
                        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
    
    def _handle_visualization(self) -> None:
        """Handle visualization of existing results."""
        results_path = input("Enter path to experiment results JSON: ").strip()
        
        if not os.path.exists(results_path):
            logger.error(f"Results file not found: {results_path}")
            return
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Create visualization
            output_dir = Path(results_path).parent
            visualizer = ThreatSimVisualizer(output_dir, style='publication')
            
            logger.info("Creating visualizations from existing results...")
            visualizer.create_comprehensive_dashboard(results)
            visualizer.create_publication_figures(results)
            
            logger.success("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def _display_configuration_details(self) -> None:
        """Display detailed configuration information."""
        logger.header("Configuration Details")
        
        logger.config("Configuration File", self.config_manager.config_path)
        logger.config("System Version", self.config_manager.get_system_version())
        
        logger.info("Training Modes:")
        for mode_key, mode_config in self.config_manager.get_training_modes().items():
            seeds_count = len(mode_config.get('seeds', []))
            timesteps = mode_config.get('timesteps', 0)
            logger.config(mode_key.title(), f"{mode_config['description']} ({timesteps:,} timesteps, {seeds_count} seed{'s' if seeds_count != 1 else ''})")
        
        logger.info("PPO Hyperparameters:")
        ppo_config = self.config_manager.get_ppo_hyperparameters()
        for param, value in ppo_config.items():
            logger.config(param, value)
        
        logger.info("Features:")
        logger.config("Visualization", "enabled")
        logger.config("Balance Validation", "enabled")
        logger.config("Win Condition Tracking", "enabled")
        logger.config("Statistical Analysis", "enabled")
    
    def _execute_user_choice(self, mode: str, scenario: str) -> None:
        """Execute user's training choice."""
        scenario_info = self.scenario_manager.get_scenario_info(scenario)
        mode_config = self.config_manager.get_training_modes()[mode]
        
        # Display final selection summary
        logger.header("Training Configuration Summary")
        logger.config("Scenario", f"{scenario_info['name']} ({scenario_info['difficulty']})")
        logger.config("Mode", mode_config['description'])
        logger.config("Expected Red Win Rate", f"{scenario_info['expected_red_win_rate']*100:.1f}%")
        logger.config("Total Training Steps", f"{mode_config['timesteps']:,}")
        logger.config("Number of Seeds", len(mode_config['seeds']))
        
        # Execute training experiment
        try:
            logger.info("Starting training experiment...")
            results = self.orchestrator.execute_experiment(mode, scenario)
            
            # Display results
            self._display_experiment_results(results)
            
        except Exception as e:
            logger.error(f"Training experiment failed: {e}")
    
    def _display_experiment_results(self, results: Dict[str, Any]) -> None:
        """Display experiment results summary."""
        logger.header("Experiment Results Summary")
        
        statistical_analysis = results.get('statistical_analysis', {})
        balance_validation = results.get('balance_validation', {})
        
        # Trial summary
        trial_summary = statistical_analysis.get('trial_summary', {})
        logger.metric("Total Trials", trial_summary.get('total_trials', 'N/A'))
        logger.metric("Successful Trials", trial_summary.get('successful_trials', 'N/A'))
        logger.metric("Success Rate", f"{trial_summary.get('success_rate', 0)*100:.1f}%")
        
        # Performance results
        team_performance = statistical_analysis.get('team_performance', {})
        if team_performance:
            red_stats = team_performance.get('red_team', {})
            blue_stats = team_performance.get('blue_team', {})
            
            logger.info("Team Performance:")
            red_ci = red_stats.get('confidence_interval', (0, 0))
            blue_ci = blue_stats.get('confidence_interval', (0, 0))
            
            logger.metric("Red Team Win Rate", f"{red_stats.get('mean_win_rate', 0):.1f}% (CI: {red_ci[0]:.1f}%-{red_ci[1]:.1f}%)")
            logger.metric("Blue Team Win Rate", f"{blue_stats.get('mean_win_rate', 0):.1f}% (CI: {blue_ci[0]:.1f}%-{blue_ci[1]:.1f}%)")
            logger.metric("Red Team Avg Reward", f"{red_stats.get('mean_reward', 0):.2f}")
            logger.metric("Blue Team Avg Reward", f"{blue_stats.get('mean_reward', 0):.2f}")
            
            logger.metric("Red Win Condition Consistency", f"{red_stats.get('win_condition_consistency', 0):.1%}")
            logger.metric("Blue Win Condition Consistency", f"{blue_stats.get('win_condition_consistency', 0):.1%}")
        
        # Statistical analysis
        statistical_tests = statistical_analysis.get('statistical_tests', {})
        metrics = statistical_analysis.get('metrics', {})
        if statistical_tests and not statistical_tests.get('error'):
            logger.info("Statistical Analysis:")
            logger.metric("Statistical Significance", 
                         "Yes" if statistical_tests.get('significant_difference', False) else "No")
            logger.metric("Statistical Power", f"{statistical_tests.get('statistical_power', 0):.1%}")
            logger.metric("Effect Size", f"{statistical_tests.get('win_rate_effect_size', 0):.3f} ({statistical_tests.get('effect_size_interpretation', 'unknown')})")
            
            if metrics:
                logger.metric("Overall Win Condition Consistency", f"{metrics.get('overall_win_condition_consistency', 0):.1%}")
                logger.metric("Balance Deviation", f"{metrics.get('balance_deviation', 0):.1f}%")
                logger.metric("Competitive Balance", f"{metrics.get('competitive_balance', 0):.2f}")
        
        # Balance validation results
        if balance_validation:
            logger.header("Balance Validation Results")
            
            balance_metrics = balance_validation.get('balance_metrics', {})
            assessment = balance_validation.get('balance_assessment', {})
            
            logger.metric("Target Red Win Rate", f"{balance_metrics.get('target_red_win_rate', 0):.1%}")
            logger.metric("Actual Red Win Rate", f"{balance_metrics.get('red_win_rate', 0):.1%}")
            logger.metric("Deviation", f"{balance_metrics.get('deviation', 0):.1%}")
            
            within_tolerance = balance_metrics.get('within_tolerance', False)
            if within_tolerance:
                logger.success("Balance validation: PASSED")
            else:
                balance_category = assessment.get('balance_category', 'unknown')
                logger.warning(f"Balance validation: {balance_category.upper()}")
                
                # Show recommendations
                recommendations = balance_validation.get('recommendations', [])
                if recommendations:
                    logger.info("Balance improvement recommendations:")
                    for i, rec in enumerate(recommendations[:3], 1):
                        logger.config(f"Recommendation {i}", f"{rec['description']} ({rec['priority']} priority)")
        
        # Output files
        config = results['experiment_config']
        logger.info("Output Files:")
        logger.config("Results Directory", str(config['output_dir']))
        logger.config("Experiment Data", str(config['output_dir'] / 'experiment_results.json'))
        logger.config("Summary", str(config['output_dir'] / 'experiment_summary.json'))
        logger.config("Visualizations", str(config['output_dir'] / 'plots/'))
        logger.config("CSV Data", str(config['output_dir'] / 'data/'))
        logger.config("HTML Report", str(config['output_dir'] / 'experiment_report.html'))
        logger.config("Summary Report", str(config['output_dir'] / 'summary_report.txt'))

def main() -> None:
    """Main entry point for ThreatSim training system."""
    try:
        cli = ThreatSimCLI()
        cli.run()
        
    except KeyboardInterrupt:
        logger.info("Training system shutdown by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Critical system error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
