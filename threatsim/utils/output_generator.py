# File: threatsim/utils/output_generator.py
# Description: Output generation for ThreatSim experiments
# Purpose: Generate plots, CSV files, and HTML reports from training results
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
from io import BytesIO

from .logger import logger

class OutputGenerator:
    """
    Fixed comprehensive output generation system.
    
    Ensures ALL expected output files are generated:
    - Training progress plots (PNG files)
    - Win rate analysis charts
    - CSV data exports for each seed
    - JSON data exports
    - HTML experiment reports
    - Statistical analysis visualizations
    """
    
    def __init__(self, output_dir: Path):
        """Initialize with proper directory structure."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        
        # Ensure directories exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib for file generation
        plt.style.use('default')  # Use default instead of seaborn to avoid issues
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        logger.debug(f"Output generator initialized: {self.output_dir}")
    
    def generate_complete_output(self, results: Dict[str, Any]) -> None:
        """Generate ALL output files for an experiment."""
        logger.info("Generating complete output suite")
        
        try:
            # 1. Generate all plots (CRITICAL)
            self._generate_all_plots(results)
            
            # 2. Export all CSV data (CRITICAL)
            self._export_all_csv_data(results)
            
            # 3. Export all JSON data (CRITICAL)
            self._export_all_json_data(results)
            
            # 4. Generate HTML report (CRITICAL)
            self._generate_html_report(results)
            
            # 5. Generate summary statistics
            self._generate_statistics_summary(results)
            
            # 6. Verify all files were created
            self._verify_output_files()
            
            logger.success(f"Complete output generated in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Output generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_all_plots(self, results: Dict[str, Any]) -> None:
        """Generate all visualization plots."""
        logger.info("Generating all visualization plots")
        
        try:
            # Plot 1: Training curves
            self._plot_training_curves(results)
            
            # Plot 2: Win rate analysis
            self._plot_win_rate_analysis(results)
            
            # Plot 3: Reward distributions
            self._plot_reward_distributions(results)
            
            # Plot 4: Balance validation
            self._plot_balance_validation(results)
            
            # Plot 5: Statistical analysis
            self._plot_statistical_analysis(results)
            
            # Plot 6: Performance comparison
            self._plot_performance_comparison(results)
            
            logger.success("All plots generated successfully")
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            raise
    
    def _plot_training_curves(self, results: Dict[str, Any]) -> None:
        """Generate training progress curves."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
            
            # Extract data from trials
            trials = results.get('trials', {})
            red_rewards = []
            blue_rewards = []
            episode_lengths = []
            
            for seed, trial_data in trials.items():
                if trial_data.get('red', {}).get('success', False):
                    red_eval = trial_data['red']['evaluation_results']
                    red_rewards.extend(red_eval.get('statistical_data', {}).get('episode_rewards', []))
                    episode_lengths.extend(red_eval.get('statistical_data', {}).get('episode_lengths', []))
                
                if trial_data.get('blue', {}).get('success', False):
                    blue_eval = trial_data['blue']['evaluation_results']
                    blue_rewards.extend(blue_eval.get('statistical_data', {}).get('episode_rewards', []))
            
            # Plot 1: Raw reward curves
            if red_rewards and blue_rewards:
                episodes = range(1, min(len(red_rewards), len(blue_rewards)) + 1)
                axes[0, 0].plot(episodes, red_rewards[:len(episodes)], 'r-', alpha=0.7, label='Red Team')
                axes[0, 0].plot(episodes, blue_rewards[:len(episodes)], 'b-', alpha=0.7, label='Blue Team')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].set_title('Training Rewards Over Time')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Moving averages
            if red_rewards and len(red_rewards) > 20:
                window = min(50, len(red_rewards) // 10)
                red_ma = pd.Series(red_rewards).rolling(window=window, min_periods=1).mean()
                axes[0, 1].plot(red_ma, 'r-', linewidth=2, label=f'Red Team (MA-{window})')
                if blue_rewards:
                    blue_ma = pd.Series(blue_rewards).rolling(window=window, min_periods=1).mean()
                    axes[0, 1].plot(blue_ma, 'b-', linewidth=2, label=f'Blue Team (MA-{window})')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Moving Average Reward')
                axes[0, 1].set_title('Smoothed Learning Curves')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Episode length distribution
            if episode_lengths:
                axes[1, 0].hist(episode_lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[1, 0].axvline(np.mean(episode_lengths), color='red', linestyle='--', linewidth=2)
                axes[1, 0].set_xlabel('Episode Length')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Episode Length Distribution')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Reward comparison
            if red_rewards and blue_rewards:
                axes[1, 1].hist(red_rewards, bins=20, alpha=0.5, color='red', label='Red Team', density=True)
                axes[1, 1].hist(blue_rewards, bins=20, alpha=0.5, color='blue', label='Blue Team', density=True)
                axes[1, 1].set_xlabel('Reward')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Reward Distribution Comparison')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug("Training curves plot saved")
            
        except Exception as e:
            logger.error(f"Training curves plot failed: {e}")
            # Create empty plot as fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Training data not available', ha='center', va='center')
            ax.set_title('Training Curves (No Data)')
            plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_win_rate_analysis(self, results: Dict[str, Any]) -> None:
        """Generate win rate analysis plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Win Rate Analysis', fontsize=16, fontweight='bold')
            
            # Extract data
            stats_analysis = results.get('statistical_analysis', {})
            team_performance = stats_analysis.get('team_performance', {})
            
            if team_performance:
                red_stats = team_performance.get('red_team', {})
                blue_stats = team_performance.get('blue_team', {})
                
                # Plot 1: Win rate comparison
                teams = ['Red Team', 'Blue Team']
                win_rates = [
                    red_stats.get('mean_win_rate', 0),
                    blue_stats.get('mean_win_rate', 0)
                ]
                colors = ['red', 'blue']
                
                bars = axes[0].bar(teams, win_rates, color=colors, alpha=0.7, edgecolor='black')
                axes[0].set_ylabel('Win Rate (%)')
                axes[0].set_title('Team Win Rate Comparison')
                axes[0].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, win_rates):
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # Plot 2: Confidence intervals
                red_ci = red_stats.get('confidence_interval', (0, 0))
                blue_ci = blue_stats.get('confidence_interval', (0, 0))
                
                axes[1].errorbar(['Red Team'], [red_stats.get('mean_win_rate', 0)], 
                                yerr=[[red_stats.get('mean_win_rate', 0) - red_ci[0]], 
                                      [red_ci[1] - red_stats.get('mean_win_rate', 0)]], 
                                fmt='o', color='red', capsize=5, capthick=2, markersize=8)
                axes[1].errorbar(['Blue Team'], [blue_stats.get('mean_win_rate', 0)], 
                                yerr=[[blue_stats.get('mean_win_rate', 0) - blue_ci[0]], 
                                      [blue_ci[1] - blue_stats.get('mean_win_rate', 0)]], 
                                fmt='o', color='blue', capsize=5, capthick=2, markersize=8)
                axes[1].set_ylabel('Win Rate (%)')
                axes[1].set_title('Win Rate with 95% Confidence Intervals')
                axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Balance validation
            balance_validation = results.get('balance_validation', {})
            balance_metrics = balance_validation.get('balance_metrics', {})
            
            if balance_metrics:
                target_rate = balance_metrics.get('target_red_win_rate', 0.3) * 100
                actual_rate = balance_metrics.get('red_win_rate', 0) * 100
                
                axes[2].axhline(y=target_rate, color='green', linestyle='--', linewidth=2, label='Target')
                axes[2].bar(['Actual'], [actual_rate], color='orange', alpha=0.7, edgecolor='black')
                axes[2].set_ylabel('Red Team Win Rate (%)')
                axes[2].set_title('Balance Validation')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'win_rate_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug("Win rate analysis plot saved")
            
        except Exception as e:
            logger.error(f"Win rate analysis plot failed: {e}")
            # Create fallback plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Win rate data not available', ha='center', va='center')
            ax.set_title('Win Rate Analysis (No Data)')
            plt.savefig(self.plots_dir / 'win_rate_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_reward_distributions(self, results: Dict[str, Any]) -> None:
        """Generate reward distribution plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Reward Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Extract data
            trials = results.get('trials', {})
            red_rewards = []
            blue_rewards = []
            
            for seed, trial_data in trials.items():
                if trial_data.get('red', {}).get('success', False):
                    red_eval = trial_data['red']['evaluation_results']
                    red_rewards.extend(red_eval.get('statistical_data', {}).get('episode_rewards', []))
                
                if trial_data.get('blue', {}).get('success', False):
                    blue_eval = trial_data['blue']['evaluation_results']
                    blue_rewards.extend(blue_eval.get('statistical_data', {}).get('episode_rewards', []))
            
            if red_rewards:
                # Red team distribution
                axes[0, 0].hist(red_rewards, bins=30, alpha=0.7, color='red', edgecolor='black')
                axes[0, 0].axvline(np.mean(red_rewards), color='darkred', linestyle='--', linewidth=2)
                axes[0, 0].set_xlabel('Reward')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Red Team Reward Distribution')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Red team box plot
                axes[0, 1].boxplot(red_rewards, patch_artist=True, 
                                  boxprops=dict(facecolor='red', alpha=0.7))
                axes[0, 1].set_ylabel('Reward')
                axes[0, 1].set_title('Red Team Reward Box Plot')
                axes[0, 1].grid(True, alpha=0.3)
            
            if blue_rewards:
                # Blue team distribution
                axes[1, 0].hist(blue_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[1, 0].axvline(np.mean(blue_rewards), color='darkblue', linestyle='--', linewidth=2)
                axes[1, 0].set_xlabel('Reward')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Blue Team Reward Distribution')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Blue team box plot
                axes[1, 1].boxplot(blue_rewards, patch_artist=True, 
                                  boxprops=dict(facecolor='blue', alpha=0.7))
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].set_title('Blue Team Reward Box Plot')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'reward_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug("Reward distributions plot saved")
            
        except Exception as e:
            logger.error(f"Reward distributions plot failed: {e}")
            # Create fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Reward data not available', ha='center', va='center')
            ax.set_title('Reward Distributions (No Data)')
            plt.savefig(self.plots_dir / 'reward_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_balance_validation(self, results: Dict[str, Any]) -> None:
        """Generate balance validation plots."""
        try:
            balance_validation = results.get('balance_validation', {})
            if not balance_validation:
                # Create placeholder
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, 'Balance validation data not available', ha='center', va='center')
                ax.set_title('Balance Validation (No Data)')
                plt.savefig(self.plots_dir / 'balance_validation.png', dpi=300, bbox_inches='tight')
                plt.close()
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Balance Validation Analysis', fontsize=16, fontweight='bold')
            
            balance_metrics = balance_validation.get('balance_metrics', {})
            
            # Plot 1: Target vs Actual
            if balance_metrics:
                categories = ['Target', 'Actual']
                values = [
                    balance_metrics.get('target_red_win_rate', 0) * 100,
                    balance_metrics.get('red_win_rate', 0) * 100
                ]
                colors = ['green', 'orange']
                
                bars = axes[0].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
                axes[0].set_ylabel('Red Team Win Rate (%)')
                axes[0].set_title('Target vs Actual Win Rate')
                axes[0].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Balance quality
            assessment = balance_validation.get('balance_assessment', {})
            if assessment:
                quality_score = assessment.get('balance_quality_score', 0)
                
                # Simple bar chart for quality
                axes[1].bar(['Balance Quality'], [quality_score], 
                           color='green' if quality_score > 0.7 else 'orange', alpha=0.7)
                axes[1].set_ylabel('Quality Score')
                axes[1].set_title('Balance Quality Assessment')
                axes[1].set_ylim(0, 1)
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'balance_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug("Balance validation plot saved")
            
        except Exception as e:
            logger.error(f"Balance validation plot failed: {e}")
            # Create fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Balance validation failed', ha='center', va='center')
            ax.set_title('Balance Validation (Error)')
            plt.savefig(self.plots_dir / 'balance_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_statistical_analysis(self, results: Dict[str, Any]) -> None:
        """Generate statistical analysis plots."""
        try:
            stats_analysis = results.get('statistical_analysis', {})
            statistical_tests = stats_analysis.get('statistical_tests', {})
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: P-value
            if statistical_tests and 'p_value' in statistical_tests:
                p_value = statistical_tests.get('p_value', 1.0)
                alpha = 0.05
                
                bars = axes[0].bar(['P-value', 'Alpha (0.05)'], [p_value, alpha], 
                                  color=['red' if p_value < alpha else 'blue', 'green'], 
                                  alpha=0.7, edgecolor='black')
                axes[0].set_ylabel('Value')
                axes[0].set_title('Statistical Significance Test')
                axes[0].grid(True, alpha=0.3)
                
                # Add significance text
                if p_value < alpha:
                    axes[0].text(0.5, max(p_value, alpha) * 0.8, 'Significant', 
                                ha='center', va='center', fontweight='bold')
            
            # Plot 2: Effect size
            if statistical_tests and 'effect_size' in statistical_tests:
                effect_size = abs(statistical_tests.get('effect_size', 0.0))
                
                # Effect size categories
                categories = ['Small\n(0.2)', 'Medium\n(0.5)', 'Large\n(0.8)']
                thresholds = [0.2, 0.5, 0.8]
                
                colors = ['lightgray'] * 3
                for i, threshold in enumerate(thresholds):
                    if effect_size >= threshold:
                        colors[i] = 'orange'
                
                bars = axes[1].bar(categories, thresholds, color=colors, alpha=0.7)
                axes[1].axhline(y=effect_size, color='red', linestyle='--', linewidth=2)
                axes[1].set_ylabel('Effect Size')
                axes[1].set_title('Effect Size Analysis')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug("Statistical analysis plot saved")
            
        except Exception as e:
            logger.error(f"Statistical analysis plot failed: {e}")
            # Create fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Statistical analysis data not available', ha='center', va='center')
            ax.set_title('Statistical Analysis (No Data)')
            plt.savefig(self.plots_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_comparison(self, results: Dict[str, Any]) -> None:
        """Generate performance comparison plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle('Performance Comparison Dashboard', fontsize=16, fontweight='bold')
            
            # Extract performance data
            stats_analysis = results.get('statistical_analysis', {})
            team_performance = stats_analysis.get('team_performance', {})
            
            if team_performance:
                red_stats = team_performance.get('red_team', {})
                blue_stats = team_performance.get('blue_team', {})
                
                # Create comparison chart
                metrics = ['Win Rate', 'Avg Reward']
                red_values = [red_stats.get('mean_win_rate', 0), red_stats.get('mean_reward', 0)]
                blue_values = [blue_stats.get('mean_win_rate', 0), blue_stats.get('mean_reward', 0)]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, red_values, width, label='Red Team', color='red', alpha=0.7)
                bars2 = ax.bar(x + width/2, blue_values, width, label='Blue Team', color='blue', alpha=0.7)
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Values')
                ax.set_title('Team Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Performance data not available', ha='center', va='center')
                ax.set_title('Performance Comparison (No Data)')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug("Performance comparison plot saved")
            
        except Exception as e:
            logger.error(f"Performance comparison plot failed: {e}")
            # Create fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Performance comparison failed', ha='center', va='center')
            ax.set_title('Performance Comparison (Error)')
            plt.savefig(self.plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _export_all_csv_data(self, results: Dict[str, Any]) -> None:
        """Export all data to CSV files for each seed."""
        logger.info("Exporting CSV data files")
        
        try:
            trials = results.get('trials', {})
            
            for seed, trial_data in trials.items():
                # Export Red team data
                if trial_data.get('red', {}).get('success', False):
                    self._export_seed_csv(trial_data['red'], f'red_seed{seed}')
                
                # Export Blue team data  
                if trial_data.get('blue', {}).get('success', False):
                    self._export_seed_csv(trial_data['blue'], f'blue_seed{seed}')
            
            # Export summary CSV
            self._export_summary_csv(results)
            
            logger.success("CSV data export completed")
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise
    
    def _export_seed_csv(self, trial_data: Dict[str, Any], filename: str) -> None:
        """Export individual seed data to CSV."""
        try:
            eval_results = trial_data.get('evaluation_results', {})
            statistical_data = eval_results.get('statistical_data', {})
            
            if not statistical_data:
                logger.warning(f"No statistical data for {filename}")
                return
            
            # Prepare data
            episode_rewards = statistical_data.get('episode_rewards', [])
            episode_lengths = statistical_data.get('episode_lengths', [])
            compromise_progress = statistical_data.get('compromise_progress', [])
            
            # Ensure equal lengths
            min_length = min(len(episode_rewards), len(episode_lengths), 
                           len(compromise_progress) if compromise_progress else len(episode_rewards))
            
            if min_length == 0:
                logger.warning(f"No episode data for {filename}")
                return
            
            # Create DataFrame
            data = {
                'episode': list(range(1, min_length + 1)),
                'reward': episode_rewards[:min_length],
                'length': episode_lengths[:min_length],
                'compromise_progress': compromise_progress[:min_length] if compromise_progress else [0] * min_length
            }
            
            df = pd.DataFrame(data)
            
            # Save CSV
            csv_path = self.data_dir / f'{filename}_results.csv'
            df.to_csv(csv_path, index=False)
            
            logger.debug(f"CSV saved: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV for {filename}: {e}")
    
    def _export_all_json_data(self, results: Dict[str, Any]) -> None:
        """Export all JSON data files."""
        logger.info("Exporting JSON data files")
        
        try:
            trials = results.get('trials', {})
            
            # Export individual seed JSON files
            for seed, trial_data in trials.items():
                if trial_data.get('red', {}).get('success', False):
                    red_eval = trial_data['red']['evaluation_results']
                    json_path = self.data_dir / f'red_seed{seed}_results.json'
                    with open(json_path, 'w') as f:
                        json.dump(red_eval, f, indent=2, default=str)
                    logger.debug(f"JSON saved: {json_path}")
                
                if trial_data.get('blue', {}).get('success', False):
                    blue_eval = trial_data['blue']['evaluation_results']
                    json_path = self.data_dir / f'blue_seed{seed}_results.json'
                    with open(json_path, 'w') as f:
                        json.dump(blue_eval, f, indent=2, default=str)
                    logger.debug(f"JSON saved: {json_path}")
            
            # Export combined analysis
            analysis_data = {
                'statistical_analysis': results.get('statistical_analysis', {}),
                'balance_validation': results.get('balance_validation', {}),
                'experiment_config': results.get('experiment_config', {}),
                'scenario_info': results.get('scenario_info', {})
            }
            
            analysis_path = self.data_dir / 'experiment_analysis.json'
            with open(analysis_path, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            logger.success("JSON data export completed")
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise
    
    def _export_summary_csv(self, results: Dict[str, Any]) -> None:
        """Export summary statistics to CSV."""
        try:
            stats_analysis = results.get('statistical_analysis', {})
            team_performance = stats_analysis.get('team_performance', {})
            
            if not team_performance:
                logger.warning("No team performance data for summary CSV")
                return
            
            # Create summary data
            summary_data = []
            
            for team_name, team_stats in team_performance.items():
                row = {
                    'team': team_name,
                    'mean_win_rate': team_stats.get('mean_win_rate', 0),
                    'std_win_rate': team_stats.get('std_win_rate', 0),
                    'mean_reward': team_stats.get('mean_reward', 0),
                    'std_reward': team_stats.get('std_reward', 0),
                    'confidence_interval_lower': team_stats.get('confidence_interval', (0, 0))[0],
                    'confidence_interval_upper': team_stats.get('confidence_interval', (0, 0))[1]
                }
                summary_data.append(row)
            
            # Export to CSV
            df = pd.DataFrame(summary_data)
            csv_path = self.data_dir / 'experiment_summary.csv'
            df.to_csv(csv_path, index=False)
            
            logger.debug(f"Summary CSV saved: {csv_path}")
            
        except Exception as e:
            logger.error(f"Summary CSV export failed: {e}")
    
    def _generate_html_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive HTML experiment report."""
        logger.info("Generating HTML experiment report")
        
        try:
            # Load plot images as base64
            plot_images = {}
            for plot_file in self.plots_dir.glob('*.png'):
                try:
                    with open(plot_file, 'rb') as f:
                        plot_images[plot_file.stem] = base64.b64encode(f.read()).decode()
                except Exception as e:
                    logger.warning(f"Failed to encode {plot_file}: {e}")
            
            # Generate HTML content
            html_content = self._create_html_template(results, plot_images)
            
            # Save HTML report
            html_path = self.output_dir / 'experiment_report.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.success(f"HTML report generated: {html_path}")
            
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            # Create minimal HTML as fallback
            self._create_minimal_html_report(results)
    
    def _create_html_template(self, results: Dict[str, Any], plot_images: Dict[str, str]) -> str:
        """Create comprehensive HTML template."""
        experiment_config = results.get('experiment_config', {})
        scenario_info = results.get('scenario_info', {})
        stats_analysis = results.get('statistical_analysis', {})
        balance_validation = results.get('balance_validation', {})
        
        # Extract key metrics
        team_performance = stats_analysis.get('team_performance', {})
        red_stats = team_performance.get('red_team', {})
        blue_stats = team_performance.get('blue_team', {})
        balance_metrics = balance_validation.get('balance_metrics', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ThreatSim Experiment Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f7fa;
                    color: #333;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 4px solid #3498db;
                    padding-bottom: 15px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 5px solid #3498db;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 25px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .plot-container {{
                    text-align: center;
                    margin: 25px 0;
                    padding: 20px;
                    background-color: #fafbfc;
                    border-radius: 8px;
                    border: 1px solid #e1e5e9;
                }}
                .plot-container img {{
                    max-width: 100%;
                    border: 2px solid #ddd;
                    border-radius: 8px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 25px;
                    margin: 25px 0;
                }}
                .info-card {{
                    background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #3498db;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                .info-card h3 {{
                    margin-top: 0;
                    color: #2c3e50;
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    margin: 8px 0;
                    padding: 5px 0;
                    border-bottom: 1px solid #bdc3c7;
                }}
                .metric:last-child {{
                    border-bottom: none;
                }}
                .metric-label {{
                    font-weight: 600;
                    color: #2c3e50;
                }}
                .metric-value {{
                    color: #27ae60;
                    font-weight: 700;
                }}
                .status-good {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .status-warning {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .status-error {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    font-style: italic;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 2px solid #ecf0f1;
                }}
                .summary-banner {{
                    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .summary-banner h2 {{
                    margin: 0;
                    border: none;
                    padding: 0;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ThreatSim Experiment Report</h1>
                
                <div class="summary-banner">
                    <h2>Experiment Summary</h2>
                    <p>Scenario: {scenario_info.get('name', 'Unknown')} | Mode: {experiment_config.get('mode', 'Unknown')} | Red Win Rate: {red_stats.get('mean_win_rate', 0):.1f}% | Blue Win Rate: {blue_stats.get('mean_win_rate', 0):.1f}%</p>
                </div>
                
                <div class="section">
                    <h2>📊 Experiment Configuration</h2>
                    <div class="info-grid">
                        <div class="info-card">
                            <h3>🎯 Scenario Information</h3>
                            <div class="metric">
                                <span class="metric-label">Scenario Name:</span>
                                <span class="metric-value">{scenario_info.get('name', 'Unknown')}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Difficulty:</span>
                                <span class="metric-value">{scenario_info.get('difficulty', 'Unknown')}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Attack Type:</span>
                                <span class="metric-value">{scenario_info.get('attack_type', 'Unknown')}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Expected Red Win Rate:</span>
                                <span class="metric-value">{scenario_info.get('expected_red_win_rate', 0)*100:.1f}%</span>
                            </div>
                        </div>
                        
                        <div class="info-card">
                            <h3>⚙️ Training Configuration</h3>
                            <div class="metric">
                                <span class="metric-label">Training Mode:</span>
                                <span class="metric-value">{experiment_config.get('mode', 'Unknown')}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Timesteps:</span>
                                <span class="metric-value">{experiment_config.get('timesteps', 'Unknown'):,}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Max Steps/Episode:</span>
                                <span class="metric-value">{experiment_config.get('max_steps_per_episode', 'Unknown')}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Number of Seeds:</span>
                                <span class="metric-value">{len(experiment_config.get('seeds', []))}</span>
                            </div>
                        </div>
                        
                        <div class="info-card">
                            <h3>📈 Performance Results</h3>
                            <div class="metric">
                                <span class="metric-label">Red Team Win Rate:</span>
                                <span class="metric-value">{red_stats.get('mean_win_rate', 0):.1f}% ± {red_stats.get('std_win_rate', 0):.1f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Blue Team Win Rate:</span>
                                <span class="metric-value">{blue_stats.get('mean_win_rate', 0):.1f}% ± {blue_stats.get('std_win_rate', 0):.1f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Red Avg Reward:</span>
                                <span class="metric-value">{red_stats.get('mean_reward', 0):.2f}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Blue Avg Reward:</span>
                                <span class="metric-value">{blue_stats.get('mean_reward', 0):.2f}</span>
                            </div>
                        </div>
                        
                        <div class="info-card">
                            <h3>⚖️ Balance Validation</h3>
                            <div class="metric">
                                <span class="metric-label">Target Red Win Rate:</span>
                                <span class="metric-value">{balance_metrics.get('target_red_win_rate', 0)*100:.1f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Actual Red Win Rate:</span>
                                <span class="metric-value">{balance_metrics.get('red_win_rate', 0)*100:.1f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Deviation:</span>
                                <span class="metric-value">{balance_metrics.get('deviation', 0)*100:.1f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Balance Status:</span>
                                <span class="{'status-good' if balance_metrics.get('within_tolerance', False) else 'status-warning'}">
                                    {'✅ PASSED' if balance_metrics.get('within_tolerance', False) else '⚠️ NEEDS REVIEW'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Training Progress</h2>
                    {'<div class="plot-container"><img src="data:image/png;base64,' + plot_images.get('training_curves', '') + '" alt="Training Curves"></div>' if 'training_curves' in plot_images else '<p>Training curves not available</p>'}
                </div>
                
                <div class="section">
                    <h2>🏆 Win Rate Analysis</h2>
                    {'<div class="plot-container"><img src="data:image/png;base64,' + plot_images.get('win_rate_analysis', '') + '" alt="Win Rate Analysis"></div>' if 'win_rate_analysis' in plot_images else '<p>Win rate analysis not available</p>'}
                </div>
                
                <div class="section">
                    <h2>📊 Reward Analysis</h2>
                    {'<div class="plot-container"><img src="data:image/png;base64,' + plot_images.get('reward_distributions', '') + '" alt="Reward Distributions"></div>' if 'reward_distributions' in plot_images else '<p>Reward distributions not available</p>'}
                </div>
                
                <div class="section">
                    <h2>⚖️ Balance Validation</h2>
                    {'<div class="plot-container"><img src="data:image/png;base64,' + plot_images.get('balance_validation', '') + '" alt="Balance Validation"></div>' if 'balance_validation' in plot_images else '<p>Balance validation plots not available</p>'}
                </div>
                
                <div class="section">
                    <h2>📊 Statistical Analysis</h2>
                    {'<div class="plot-container"><img src="data:image/png;base64,' + plot_images.get('statistical_analysis', '') + '" alt="Statistical Analysis"></div>' if 'statistical_analysis' in plot_images else '<p>Statistical analysis not available</p>'}
                </div>
                
                <div class="section">
                    <h2>⚔️ Performance Comparison</h2>
                    {'<div class="plot-container"><img src="data:image/png;base64,' + plot_images.get('performance_comparison', '') + '" alt="Performance Comparison"></div>' if 'performance_comparison' in plot_images else '<p>Performance comparison not available</p>'}
                </div>
                
                <div class="timestamp">
                    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by ThreatSim Enhanced Training System v{experiment_config.get('system_version', '0.3.1')}</p>
                    <p>Full experiment data and CSV files available in the data/ directory</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _create_minimal_html_report(self, results: Dict[str, Any]) -> None:
        """Create minimal HTML report as fallback."""
        try:
            minimal_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ThreatSim Experiment Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1 class="header">ThreatSim Experiment Report</h1>
                <p>Experiment completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Full data available in JSON and CSV files in the data/ directory.</p>
                <p>Visualization plots available in the plots/ directory.</p>
            </body>
            </html>
            """
            
            html_path = self.output_dir / 'experiment_report.html'
            with open(html_path, 'w') as f:
                f.write(minimal_html)
            
            logger.warning(f"Minimal HTML report created: {html_path}")
            
        except Exception as e:
            logger.error(f"Failed to create minimal HTML report: {e}")
    
    def _generate_statistics_summary(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive statistics summary."""
        try:
            stats_summary = {
                'experiment_metadata': {
                    'generation_time': datetime.now().isoformat(),
                    'scenario': results.get('scenario_info', {}).get('name', 'Unknown'),
                    'mode': results.get('experiment_config', {}).get('mode', 'Unknown'),
                    'system_version': '0.3.1'
                },
                'statistical_analysis': results.get('statistical_analysis', {}),
                'balance_validation': results.get('balance_validation', {}),
                'trial_summary': {
                    'total_trials': len(results.get('trials', {})),
                    'successful_trials': sum(1 for t in results.get('trials', {}).values() 
                                           if t.get('red', {}).get('success', False) and t.get('blue', {}).get('success', False))
                }
            }
            
            # Save comprehensive statistics
            stats_path = self.data_dir / 'comprehensive_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats_summary, f, indent=2, default=str)
            
            logger.success(f"Statistics summary saved: {stats_path}")
            
        except Exception as e:
            logger.error(f"Statistics summary generation failed: {e}")
    
    def _verify_output_files(self) -> None:
        """Verify that all expected output files were created."""
        logger.info("Verifying output file generation")
        
        expected_files = {
            'plots': ['training_curves.png', 'win_rate_analysis.png', 'reward_distributions.png', 
                     'balance_validation.png', 'statistical_analysis.png', 'performance_comparison.png'],
            'data': ['experiment_summary.csv', 'experiment_analysis.json', 'comprehensive_statistics.json'],
            'reports': ['experiment_report.html']
        }
        
        # Check plots
        plots_created = []
        for plot_file in expected_files['plots']:
            plot_path = self.plots_dir / plot_file
            if plot_path.exists():
                plots_created.append(plot_file)
                logger.debug(f"✅ Plot created: {plot_file}")
            else:
                logger.warning(f"❌ Plot missing: {plot_file}")
        
        # Check data files
        data_created = []
        for data_file in expected_files['data']:
            data_path = self.data_dir / data_file
            if data_path.exists():
                data_created.append(data_file)
                logger.debug(f"✅ Data file created: {data_file}")
            else:
                logger.warning(f"❌ Data file missing: {data_file}")
        
        # Check seed-specific files
        seed_files = list(self.data_dir.glob('*_seed*_results.*'))
        logger.debug(f"✅ Seed-specific files created: {len(seed_files)}")
        
        # Check reports
        reports_created = []
        for report_file in expected_files['reports']:
            report_path = self.output_dir / report_file
            if report_path.exists():
                reports_created.append(report_file)
                logger.debug(f"✅ Report created: {report_file}")
            else:
                logger.warning(f"❌ Report missing: {report_file}")
        
        # Summary
        total_expected = len(expected_files['plots']) + len(expected_files['data']) + len(expected_files['reports'])
        total_created = len(plots_created) + len(data_created) + len(reports_created)
        
        logger.config("Files Created", f"{total_created}/{total_expected}")
        logger.config("Plots Created", f"{len(plots_created)}/{len(expected_files['plots'])}")
        logger.config("Data Files Created", f"{len(data_created)}/{len(expected_files['data'])}")
        logger.config("Seed Files Created", len(seed_files))
        logger.config("Reports Created", f"{len(reports_created)}/{len(expected_files['reports'])}")
        
        if total_created == total_expected:
            logger.success("All expected output files generated successfully")
        else:
            logger.warning(f"Some output files missing: {total_expected - total_created} files not created")
