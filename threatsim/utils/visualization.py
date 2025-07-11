# File: threatsim/utils/visualization.py
# Description: Visualization system for ThreatSim training
# Purpose: Generate plots and interactive visualizations
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .logger import logger

class ThreatSimVisualizer:
    """
    Advanced visualization system for ThreatSim training analysis.
    
    Provides comprehensive visualization capabilities including:
    - Real-time training monitoring
    - Statistical analysis plots
    - Interactive dashboards
    - Publication-quality figures
    - Animated training progress
    """
    
    def __init__(self, output_dir: Path, style: str = 'publication'):
        """
        Initialize visualization system.
        
        Args:
            output_dir: Directory for output files
            style: Visualization style ('publication', 'presentation', 'interactive')
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.interactive_dir = self.output_dir / "interactive"
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        self.style = style
        self._configure_style()
        
        # Color schemes - individual color values for matplotlib compatibility
        self.colors = {
            'red_team': '#E74C3C',      # Red team color
            'blue_team': '#3498DB',     # Blue team color
            'success': '#27AE60',       # Success green
            'warning': '#F39C12',       # Warning orange
            'error': '#E74C3C',         # Error red
            'neutral': '#95A5A6',       # Neutral gray
            'background': '#ECF0F1',    # Light background
            'text': '#2C3E50',          # Dark text
            'accent': '#9B59B6'         # Purple accent
        }
        
        # Individual color references for matplotlib
        self.red_color = '#E74C3C'
        self.blue_color = '#3498DB'
        self.success_color = '#27AE60'
        self.warning_color = '#F39C12'
        self.error_color = '#E74C3C'
        self.neutral_color = '#95A5A6'
        self.accent_color = '#9B59B6'

        logger.info(f"Visualization system initialized with {style} style")
    
    def _configure_style(self):
        """Configure matplotlib and seaborn styles."""
        if self.style == 'publication':
            plt.style.use('seaborn-v0_8-paper')
            sns.set_context("paper", font_scale=1.2)
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            sns.set_context("talk", font_scale=1.3)
        else:
            plt.style.use('seaborn-v0_8')
            sns.set_context("notebook", font_scale=1.1)
        
        # Custom style parameters
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#2C3E50'
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    def create_comprehensive_dashboard(self, results: Dict[str, Any]) -> None:
        """
        Create comprehensive training dashboard.
        
        Args:
            results: Complete experiment results
        """
        logger.info("Creating comprehensive training dashboard")
        
        # Create static plots
        self._create_training_overview(results)
        self._create_performance_analysis(results)
        self._create_balance_analysis(results)
        self._create_statistical_dashboard(results)
        
        # Create interactive visualizations
        self._create_interactive_dashboard(results)
        
        # Create animated training progress
        self._create_training_animation(results)
        
        logger.success("Comprehensive dashboard created")
    
    def _create_training_overview(self, results: Dict[str, Any]) -> None:
        """Create training overview visualization."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('ThreatSim Training Overview', fontsize=20, fontweight='bold', y=0.95)
        
        # Extract data
        trials = results.get('trials', {})
        stats_analysis = results.get('statistical_analysis', {})
        team_performance = stats_analysis.get('team_performance', {})
        
        if not trials:
            return
        
        # Collect all training data
        red_rewards = []
        blue_rewards = []
        win_rates = {'red': [], 'blue': []}
        
        for seed, trial_data in trials.items():
            if trial_data.get('red', {}).get('success', False):
                red_eval = trial_data['red']['evaluation_results']
                red_rewards.extend(red_eval.get('statistical_data', {}).get('episode_rewards', []))
                win_rates['red'].append(red_eval.get('win_rate', 0))
            
            if trial_data.get('blue', {}).get('success', False):
                blue_eval = trial_data['blue']['evaluation_results']
                blue_rewards.extend(blue_eval.get('statistical_data', {}).get('episode_rewards', []))
                win_rates['blue'].append(blue_eval.get('win_rate', 0))
        
        # Plot 1: Training Progress (Large)
        ax1 = fig.add_subplot(gs[0, :2])
        if red_rewards and blue_rewards:
            episodes = range(1, min(len(red_rewards), len(blue_rewards)) + 1)
            ax1.plot(episodes, red_rewards[:len(episodes)], 
                    color=self.red_color, label='Red Team', linewidth=2, alpha=0.8)
            ax1.plot(episodes, blue_rewards[:len(episodes)], 
                    color=self.blue_color, label='Blue Team', linewidth=2, alpha=0.8)
            ax1.set_title('Training Progress: Rewards Over Time', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win Rate Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        if win_rates['red'] and win_rates['blue']:
            mean_red = np.mean(win_rates['red'])
            mean_blue = np.mean(win_rates['blue'])
            std_red = np.std(win_rates['red'])
            std_blue = np.std(win_rates['blue'])
            
            bars = ax2.bar(['Red Team', 'Blue Team'], [mean_red, mean_blue], 
                          color=[self.red_color, self.blue_color], alpha=0.7, edgecolor='black', linewidth=1.5)
            ax2.errorbar(['Red Team', 'Blue Team'], [mean_red, mean_blue], 
                        yerr=[std_red, std_blue], fmt='none', color='black', capsize=5)
            ax2.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Win Rate (%)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, [mean_red, mean_blue]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_red/2,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Performance Distribution
        ax3 = fig.add_subplot(gs[0, 3])
        if red_rewards and blue_rewards:
            ax3.hist(red_rewards, bins=20, alpha=0.6, color=self.red_color, 
                    label='Red Team', density=True, edgecolor='black')
            ax3.hist(blue_rewards, bins=20, alpha=0.6, color=self.blue_color, 
                    label='Blue Team', density=True, edgecolor='black')
            ax3.set_title('Reward Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Curves with Smoothing
        ax4 = fig.add_subplot(gs[1, :2])
        if red_rewards:
            window = min(50, len(red_rewards) // 10)
            if window > 1:
                red_smooth = pd.Series(red_rewards).rolling(window=window).mean()
                blue_smooth = pd.Series(blue_rewards).rolling(window=window).mean()
                
                ax4.plot(red_smooth, color=self.red_color, 
                        label=f'Red Team (MA-{window})', linewidth=3)
                ax4.plot(blue_smooth, color=self.blue_color, 
                        label=f'Blue Team (MA-{window})', linewidth=3)
                ax4.fill_between(range(len(red_smooth)), red_smooth, alpha=0.3, 
                               color=self.red_color)
                ax4.fill_between(range(len(blue_smooth)), blue_smooth, alpha=0.3, 
                               color=self.blue_color)
                ax4.set_title('Smoothed Learning Curves', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Moving Average Reward')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Statistical Significance
        ax5 = fig.add_subplot(gs[1, 2])
        statistical_tests = stats_analysis.get('statistical_tests', {})
        if statistical_tests:
            p_value = statistical_tests.get('p_value', 1.0)
            alpha = 0.05
            
            bars = ax5.bar(['P-value', 'Alpha'], [p_value, alpha], 
                          color=[self.error_color if p_value < alpha else self.blue_color, self.success_color], 
                          alpha=0.7, edgecolor='black')
            ax5.set_title('Statistical Significance', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Value')
            ax5.grid(True, alpha=0.3)
            
            # Add significance annotation
            if p_value < alpha:
                ax5.text(0.5, max(p_value, alpha) * 0.8, 'Significant', 
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        # Plot 6: Balance Validation
        ax6 = fig.add_subplot(gs[1, 3])
        balance_validation = results.get('balance_validation', {})
        balance_metrics = balance_validation.get('balance_metrics', {})
        
        if balance_metrics:
            target_rate = balance_metrics.get('target_red_win_rate', 0) * 100
            actual_rate = balance_metrics.get('red_win_rate', 0) * 100
            deviation = abs(actual_rate - target_rate)
            
            bars = ax6.bar(['Target', 'Actual'], [target_rate, actual_rate], 
                          color=[self.success_color, self.warning_color], alpha=0.7, edgecolor='black')
            ax6.set_title('Balance Validation', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Red Win Rate (%)')
            ax6.grid(True, alpha=0.3)
            
            # Add deviation annotation
            ax6.text(0.5, max(target_rate, actual_rate) * 0.8, 
                    f'Deviation: {deviation:.1f}%', ha='center', va='center',
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightcoral' if deviation > 10 else 'lightgreen'))
        
        # Plot 7: Episode Length Analysis
        ax7 = fig.add_subplot(gs[2, :2])
        episode_lengths = []
        for seed, trial_data in trials.items():
            for team in ['red', 'blue']:
                if trial_data.get(team, {}).get('success', False):
                    eval_results = trial_data[team]['evaluation_results']
                    lengths = eval_results.get('statistical_data', {}).get('episode_lengths', [])
                    episode_lengths.extend(lengths)
        
        if episode_lengths:
            ax7.hist(episode_lengths, bins=20, alpha=0.7, color=self.neutral_color, 
                    edgecolor='black')
            ax7.axvline(np.mean(episode_lengths), color=self.error_color, linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
            ax7.set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Episode Length')
            ax7.set_ylabel('Frequency')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Training Efficiency
        ax8 = fig.add_subplot(gs[2, 2])
        if trials:
            success_rates = []
            for seed, trial_data in trials.items():
                red_success = trial_data.get('red', {}).get('success', False)
                blue_success = trial_data.get('blue', {}).get('success', False)
                success_rates.append(int(red_success and blue_success))
            
            success_rate = np.mean(success_rates) * 100
            bars = ax8.bar(['Success Rate'], [success_rate], 
                          color=self.success_color, alpha=0.7, edgecolor='black')
            ax8.set_title('Training Success Rate', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Success Rate (%)')
            ax8.set_ylim(0, 100)
            ax8.grid(True, alpha=0.3)
            
            # Add value label
            ax8.text(0, success_rate + 5, f'{success_rate:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 9: Experiment Summary
        ax9 = fig.add_subplot(gs[2, 3])
        ax9.axis('off')
        
        # Create summary text
        experiment_config = results.get('experiment_config', {})
        summary_text = f"""
        EXPERIMENT SUMMARY
        ==================
        
        Scenario: {results.get('scenario_info', {}).get('name', 'Unknown')}
        Mode: {experiment_config.get('mode', 'Unknown')}
        Seeds: {len(trials)}
        
        Total Episodes: {sum(len(t.get('red', {}).get('evaluation_results', {}).get('statistical_data', {}).get('episode_rewards', [])) for t in trials.values())}
        
        Training Status: {'✓ Complete' if trials else '✗ Incomplete'}
        Balance Status: {'✓ Balanced' if balance_metrics.get('within_tolerance', False) else '⚠ Needs Review'}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success("Training overview visualization created")
    
    def _create_performance_analysis(self, results: Dict[str, Any]) -> None:
        """Create detailed performance analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Extract trial data
        trials = results.get('trials', {})
        red_data = []
        blue_data = []
        
        for seed, trial_data in trials.items():
            if trial_data.get('red', {}).get('success', False):
                red_eval = trial_data['red']['evaluation_results']
                red_data.append({
                    'seed': seed,
                    'win_rate': red_eval.get('win_rate', 0),
                    'mean_reward': red_eval.get('mean_reward', 0),
                    'std_reward': red_eval.get('std_reward', 0),
                    'mean_length': red_eval.get('mean_length', 0),
                    'rewards': red_eval.get('statistical_data', {}).get('episode_rewards', [])
                })
            
            if trial_data.get('blue', {}).get('success', False):
                blue_eval = trial_data['blue']['evaluation_results']
                blue_data.append({
                    'seed': seed,
                    'win_rate': blue_eval.get('win_rate', 0),
                    'mean_reward': blue_eval.get('mean_reward', 0),
                    'std_reward': blue_eval.get('std_reward', 0),
                    'mean_length': blue_eval.get('mean_length', 0),
                    'rewards': blue_eval.get('statistical_data', {}).get('episode_rewards', [])
                })
        
        # Plot 1: Win Rate Variance Across Seeds
        if red_data and blue_data:
            seeds = [d['seed'] for d in red_data]
            red_wins = [d['win_rate'] for d in red_data]
            blue_wins = [d['win_rate'] for d in blue_data]
            
            x = np.arange(len(seeds))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, red_wins, width, label='Red Team', 
                          color=self.red_color, alpha=0.7)
            axes[0, 0].bar(x + width/2, blue_wins, width, label='Blue Team', 
                          color=self.blue_color, alpha=0.7)
            axes[0, 0].set_title('Win Rate Variance Across Seeds')
            axes[0, 0].set_xlabel('Seed')
            axes[0, 0].set_ylabel('Win Rate (%)')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(seeds)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Reward Consistency Analysis
        if red_data:
            red_means = [d['mean_reward'] for d in red_data]
            red_stds = [d['std_reward'] for d in red_data]
            
            axes[0, 1].scatter(red_means, red_stds, color=self.red_color, 
                             alpha=0.7, s=100, label='Red Team')
            if blue_data:
                blue_means = [d['mean_reward'] for d in blue_data]
                blue_stds = [d['std_reward'] for d in blue_data]
                axes[0, 1].scatter(blue_means, blue_stds, color=self.blue_color, 
                                 alpha=0.7, s=100, label='Blue Team')
            
            axes[0, 1].set_title('Reward Consistency Analysis')
            axes[0, 1].set_xlabel('Mean Reward')
            axes[0, 1].set_ylabel('Standard Deviation')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance Correlation
        if red_data and blue_data:
            red_wins = [d['win_rate'] for d in red_data]
            blue_wins = [d['win_rate'] for d in blue_data]
            
            axes[0, 2].scatter(red_wins, blue_wins, color=self.neutral_color, 
                             alpha=0.7, s=100)
            axes[0, 2].plot([0, 100], [100, 0], 'r--', alpha=0.5, label='Perfect Inverse')
            axes[0, 2].set_title('Team Performance Correlation')
            axes[0, 2].set_xlabel('Red Team Win Rate (%)')
            axes[0, 2].set_ylabel('Blue Team Win Rate (%)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Learning Stability
        if red_data:
            for i, data in enumerate(red_data):
                rewards = data['rewards']
                if len(rewards) > 20:
                    # Calculate rolling variance
                    rolling_var = pd.Series(rewards).rolling(window=20).var()
                    axes[1, 0].plot(rolling_var, alpha=0.7, label=f'Seed {data["seed"]}')
            
            axes[1, 0].set_title('Learning Stability (Rolling Variance)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Rolling Variance')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Episode Length Distribution
        if red_data and blue_data:
            red_lengths = [d['mean_length'] for d in red_data]
            blue_lengths = [d['mean_length'] for d in blue_data]
            
            bp = axes[1, 1].boxplot([red_lengths, blue_lengths], 
                                   labels=['Red Team', 'Blue Team'],
                                   patch_artist=True)
            bp['boxes'][0].set_facecolor(self.red_color)
            bp['boxes'][1].set_facecolor(self.blue_color)
            for box in bp['boxes']:
                box.set_alpha(0.7)
            
            axes[1, 1].set_title('Episode Length Distribution')
            axes[1, 1].set_ylabel('Mean Episode Length')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Performance Metrics Heatmap
        if red_data and blue_data:
            # Create performance matrix
            metrics = ['Win Rate', 'Mean Reward', 'Consistency', 'Efficiency']
            teams = ['Red Team', 'Blue Team']
            
            red_metrics = [
                np.mean([d['win_rate'] for d in red_data]),
                np.mean([d['mean_reward'] for d in red_data]),
                1 / (np.mean([d['std_reward'] for d in red_data]) + 1),  # Inverse of std
                1 / (np.mean([d['mean_length'] for d in red_data]) + 1)   # Inverse of length
            ]
            
            blue_metrics = [
                np.mean([d['win_rate'] for d in blue_data]),
                np.mean([d['mean_reward'] for d in blue_data]),
                1 / (np.mean([d['std_reward'] for d in blue_data]) + 1),
                1 / (np.mean([d['mean_length'] for d in blue_data]) + 1)
            ]
            
            # Normalize metrics
            all_metrics = red_metrics + blue_metrics
            max_val = max(all_metrics)
            min_val = min(all_metrics)
            
            red_norm = [(x - min_val) / (max_val - min_val) for x in red_metrics]
            blue_norm = [(x - min_val) / (max_val - min_val) for x in blue_metrics]
            
            data_matrix = np.array([red_norm, blue_norm])
            
            im = axes[1, 2].imshow(data_matrix, cmap='RdYlBu', aspect='auto')
            axes[1, 2].set_xticks(range(len(metrics)))
            axes[1, 2].set_yticks(range(len(teams)))
            axes[1, 2].set_xticklabels(metrics)
            axes[1, 2].set_yticklabels(teams)
            axes[1, 2].set_title('Performance Metrics Heatmap')
            
            # Add text annotations
            for i in range(len(teams)):
                for j in range(len(metrics)):
                    text = axes[1, 2].text(j, i, f'{data_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success("Performance analysis visualization created")
    
    def _create_balance_analysis(self, results: Dict[str, Any]) -> None:
        """Create balance analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Balance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        balance_validation = results.get('balance_validation', {})
        balance_metrics = balance_validation.get('balance_metrics', {})
        balance_assessment = balance_validation.get('balance_assessment', {})
        
        # Plot 1: Target vs Actual Balance
        if balance_metrics:
            target_rate = balance_metrics.get('target_red_win_rate', 0) * 100
            actual_rate = balance_metrics.get('red_win_rate', 0) * 100
            deviation = balance_metrics.get('deviation', 0) * 100
            
            categories = ['Target', 'Actual']
            values = [target_rate, actual_rate]
            colors = [self.success_color, self.warning_color if abs(deviation) > 10 else self.success_color]
            
            bars = axes[0, 0].bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Target vs Actual Balance')
            axes[0, 0].set_ylabel('Red Team Win Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels and deviation
            for bar, value in zip(bars, values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            axes[0, 0].text(0.5, max(values) * 0.8, f'Deviation: {deviation:.1f}%',
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='lightcoral' if abs(deviation) > 10 else 'lightgreen'))
        
        # Plot 2: Balance Quality Assessment
        if balance_assessment:
            quality_score = balance_assessment.get('balance_quality_score', 0)
            confidence_level = balance_assessment.get('confidence_level', 0)
            
            # Create gauge chart
            theta = np.linspace(0, 2*np.pi, 100)
            r_outer = 1
            r_inner = 0.7
            
            # Quality score ring
            quality_theta = np.linspace(0, 2*np.pi * quality_score, int(100 * quality_score))
            
            axes[0, 1].fill_between(theta, r_inner, r_outer, alpha=0.3, color='lightgray')
            axes[0, 1].fill_between(quality_theta, r_inner, r_outer, alpha=0.8, 
                                   color=self.success_color if quality_score > 0.7 else self.warning_color if quality_score > 0.4 else self.error_color)
            axes[0, 1].set_ylim(0, 1.2)
            axes[0, 1].set_xlim(-1.2, 1.2)
            axes[0, 1].set_title('Balance Quality Score')
            axes[0, 1].text(0, 0, f'{quality_score:.2f}', ha='center', va='center', 
                           fontsize=20, fontweight='bold')
            axes[0, 1].axis('equal')
            axes[0, 1].axis('off')
        
        # Plot 3: Statistical Validation
        statistical_validation = balance_validation.get('statistical_validation', {})
        if statistical_validation:
            validation_items = ['Sufficient Episodes', 'Statistically Significant', 
                              'Adequate Sample Size', 'Confidence Level']
            validation_status = [
                statistical_validation.get('sufficient_episodes', False),
                statistical_validation.get('statistically_significant', False),
                statistical_validation.get('sample_size_adequate', False),
                statistical_validation.get('confidence_level', 0) >= 0.95
            ]
            
            colors = [self.success_color if status else self.error_color for status in validation_status]
            y_pos = np.arange(len(validation_items))
            
            bars = axes[1, 0].barh(y_pos, [1] * len(validation_items), color=colors, alpha=0.7)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(validation_items)
            axes[1, 0].set_xlabel('Validation Status')
            axes[1, 0].set_title('Statistical Validation Checklist')
            axes[1, 0].set_xlim(0, 1)
            
            # Add status labels
            for i, (bar, status) in enumerate(zip(bars, validation_status)):
                axes[1, 0].text(0.5, i, '✓' if status else '✗', 
                               ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # Plot 4: Recommendations Summary
        recommendations = balance_validation.get('recommendations', [])
        if recommendations:
            priorities = ['Critical', 'High', 'Medium', 'Low']
            priority_counts = {p: 0 for p in priorities}
            
            for rec in recommendations:
                priority = rec.get('priority', 'low').title()
                if priority in priority_counts:
                    priority_counts[priority] += 1
            
            sizes = [priority_counts[p] for p in priorities]
            colors_pie = ['darkred', 'orange', 'yellow', 'lightgreen']
            
            # Only show non-zero segments
            non_zero_sizes = [s for s in sizes if s > 0]
            non_zero_labels = [p for p, s in zip(priorities, sizes) if s > 0]
            non_zero_colors = [c for c, s in zip(colors_pie, sizes) if s > 0]
            
            if non_zero_sizes:
                axes[1, 1].pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
                              autopct='%1.0f', startangle=90)
                axes[1, 1].set_title('Recommendations by Priority')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Recommendations', ha='center', va='center',
                               fontsize=14, fontweight='bold')
                axes[1, 1].set_title('Recommendations by Priority')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'balance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success("Balance analysis visualization created")
    
    def _create_statistical_dashboard(self, results: Dict[str, Any]) -> None:
        """Create statistical analysis dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Analysis Dashboard', fontsize=16, fontweight='bold')
        
        stats_analysis = results.get('statistical_analysis', {})
        statistical_tests = stats_analysis.get('statistical_tests', {})
        team_performance = stats_analysis.get('team_performance', {})
        
        # Plot 1: P-value and Effect Size
        if statistical_tests:
            p_value = statistical_tests.get('p_value', 1.0)
            effect_size = statistical_tests.get('effect_size', 0.0)
            
            # P-value visualization
            alpha_levels = [0.05, 0.01, 0.001]
            alpha_labels = ['α = 0.05', 'α = 0.01', 'α = 0.001']
            
            bars = axes[0, 0].bar(['P-value'] + alpha_labels, [p_value] + alpha_levels,
                                 color=[self.error_color if p_value < 0.05 else self.blue_color] + [self.success_color] * 3,
                                 alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Statistical Significance Test')
            axes[0, 0].set_ylabel('P-value')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add significance annotation
            significance_text = 'Significant' if p_value < 0.05 else 'Not Significant'
            axes[0, 0].text(0.5, 0.8, significance_text, ha='center', va='center',
                           transform=axes[0, 0].transAxes, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='lightgreen' if p_value < 0.05 else 'lightcoral'))
        
        # Plot 2: Effect Size Interpretation
        if statistical_tests:
            effect_categories = ['Negligible\n(< 0.2)', 'Small\n(0.2-0.5)', 'Medium\n(0.5-0.8)', 'Large\n(> 0.8)']
            effect_thresholds = [0.2, 0.5, 0.8, 2.0]
            
            abs_effect = abs(effect_size)
            colors = ['lightgray'] * 4
            
            for i, threshold in enumerate(effect_thresholds):
                if abs_effect <= threshold:
                    colors[i] = self.warning_color
                    break
            
            bars = axes[0, 1].bar(effect_categories, effect_thresholds, color=colors, alpha=0.7)
            axes[0, 1].axhline(y=abs_effect, color=self.error_color, linestyle='--', linewidth=3,
                              label=f'Actual: {abs_effect:.3f}')
            axes[0, 1].set_title('Effect Size Analysis')
            axes[0, 1].set_ylabel('Effect Size (Cohen\'s d)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confidence Intervals
        if team_performance:
            red_stats = team_performance.get('red_team', {})
            blue_stats = team_performance.get('blue_team', {})
            
            teams = ['Red Team', 'Blue Team']
            means = [red_stats.get('mean_win_rate', 0), blue_stats.get('mean_win_rate', 0)]
            red_ci = red_stats.get('confidence_interval', (0, 0))
            blue_ci = blue_stats.get('confidence_interval', (0, 0))
            
            # Calculate error bars
            red_error = [[means[0] - red_ci[0]], [red_ci[1] - means[0]]]
            blue_error = [[means[1] - blue_ci[0]], [blue_ci[1] - means[1]]]
            
            # Plot red team confidence interval
            axes[0, 2].errorbar([teams[0]], [means[0]], yerr=[[red_error[0][0]], [red_error[1][0]]],
                               fmt='o', capsize=5, capthick=2, markersize=8,
                               color=self.red_color, label='Red Team')
            # Plot blue team confidence interval
            axes[0, 2].errorbar([teams[1]], [means[1]], yerr=[[blue_error[0][0]], [blue_error[1][0]]],
                               fmt='o', capsize=5, capthick=2, markersize=8,
                               color=self.blue_color, label='Blue Team')
            axes[0, 2].set_title('95% Confidence Intervals')
            axes[0, 2].set_ylabel('Win Rate (%)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Sample Size Analysis
        sample_size_analysis = stats_analysis.get('sample_size_analysis', {})
        if sample_size_analysis:
            current_size = sample_size_analysis.get('current_sample_size', 0)
            recommended_size = sample_size_analysis.get('recommended_sample_size', 0)
            
            bars = axes[1, 0].bar(['Current', 'Recommended'], [current_size, recommended_size],
                                 color=[self.warning_color, self.success_color], alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Sample Size Analysis')
            axes[1, 0].set_ylabel('Number of Trials')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add adequacy annotation
            adequate = sample_size_analysis.get('sample_size_adequate', False)
            axes[1, 0].text(0.5, 0.8, 'Adequate' if adequate else 'Insufficient',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                           facecolor='lightgreen' if adequate else 'lightcoral'))
        
        # Plot 5: Power Analysis
        if statistical_tests:
            power = statistical_tests.get('statistical_power', 0.0)
            
            # Create power visualization
            power_levels = [0.5, 0.8, 0.95]
            power_labels = ['Weak\n(0.5)', 'Adequate\n(0.8)', 'Strong\n(0.95)']
            
            bars = axes[1, 1].bar(power_labels, power_levels, 
                                 color=[self.error_color, self.warning_color, self.success_color], alpha=0.7)
            axes[1, 1].axhline(y=power, color=self.blue_color, linestyle='--', linewidth=3,
                              label=f'Actual: {power:.3f}')
            axes[1, 1].set_title('Statistical Power Analysis')
            axes[1, 1].set_ylabel('Statistical Power')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Distribution Comparison
        if team_performance:
            red_stats = team_performance.get('red_team', {})
            blue_stats = team_performance.get('blue_team', {})
            
            # Create normal distributions based on means and stds
            red_mean = red_stats.get('mean_win_rate', 0)
            red_std = red_stats.get('std_win_rate', 1)
            blue_mean = blue_stats.get('mean_win_rate', 0)
            blue_std = blue_stats.get('std_win_rate', 1)
            
            x = np.linspace(min(red_mean - 3*red_std, blue_mean - 3*blue_std),
                           max(red_mean + 3*red_std, blue_mean + 3*blue_std), 100)
            
            red_dist = (1/(red_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - red_mean) / red_std) ** 2)
            blue_dist = (1/(blue_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - blue_mean) / blue_std) ** 2)
            
            axes[1, 2].plot(x, red_dist, color=self.red_color, linewidth=2, label='Red Team')
            axes[1, 2].plot(x, blue_dist, color=self.blue_color, linewidth=2, label='Blue Team')
            axes[1, 2].fill_between(x, red_dist, alpha=0.3, color=self.red_color)
            axes[1, 2].fill_between(x, blue_dist, alpha=0.3, color=self.blue_color)
            axes[1, 2].set_title('Win Rate Distribution Comparison')
            axes[1, 2].set_xlabel('Win Rate (%)')
            axes[1, 2].set_ylabel('Probability Density')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'statistical_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success("Statistical dashboard visualization created")
    
    def _create_interactive_dashboard(self, results: Dict[str, Any]) -> None:
        """Create interactive Plotly dashboard."""
        logger.info("Creating interactive dashboard")
        
        # Create main dashboard with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Progress', 'Win Rate Analysis', 
                          'Performance Metrics', 'Balance Validation',
                          'Statistical Tests', 'Episode Analysis'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Extract data
        trials = results.get('trials', {})
        red_rewards = []
        blue_rewards = []
        win_rates = {'red': [], 'blue': []}
        
        for seed, trial_data in trials.items():
            if trial_data.get('red', {}).get('success', False):
                red_eval = trial_data['red']['evaluation_results']
                red_rewards.extend(red_eval.get('statistical_data', {}).get('episode_rewards', []))
                win_rates['red'].append(red_eval.get('win_rate', 0))
            
            if trial_data.get('blue', {}).get('success', False):
                blue_eval = trial_data['blue']['evaluation_results']
                blue_rewards.extend(blue_eval.get('statistical_data', {}).get('episode_rewards', []))
                win_rates['blue'].append(blue_eval.get('win_rate', 0))
        
        # Add training progress
        if red_rewards and blue_rewards:
            episodes = list(range(1, min(len(red_rewards), len(blue_rewards)) + 1))
            fig.add_trace(
                go.Scatter(x=episodes, y=red_rewards[:len(episodes)], 
                          mode='lines', name='Red Team', line=dict(color=self.red_color)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=episodes, y=blue_rewards[:len(episodes)], 
                          mode='lines', name='Blue Team', line=dict(color=self.blue_color)),
                row=1, col=1
            )
        
        # Add win rate comparison
        if win_rates['red'] and win_rates['blue']:
            fig.add_trace(
                go.Bar(x=['Red Team', 'Blue Team'], 
                      y=[np.mean(win_rates['red']), np.mean(win_rates['blue'])],
                      marker_color=[self.red_color, self.blue_color], name='Win Rates'),
                row=1, col=2
            )
        
        # Add performance scatter
        if win_rates['red'] and win_rates['blue']:
            fig.add_trace(
                go.Scatter(x=win_rates['red'], y=win_rates['blue'], 
                          mode='markers', name='Performance Correlation',
                          marker=dict(size=10, color=self.accent_color)),
                row=2, col=1
            )
        
        # Add balance indicator
        balance_validation = results.get('balance_validation', {})
        balance_metrics = balance_validation.get('balance_metrics', {})
        if balance_metrics:
            target_rate = balance_metrics.get('target_red_win_rate', 0) * 100
            actual_rate = balance_metrics.get('red_win_rate', 0) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=actual_rate,
                    delta={'reference': target_rate},
                    gauge={'axis': {'range': [None, 100]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 100], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': target_rate}},
                    title={'text': "Balance Score"}),
                row=2, col=2
            )
        
        # Add statistical tests
        stats_analysis = results.get('statistical_analysis', {})
        statistical_tests = stats_analysis.get('statistical_tests', {})
        if statistical_tests:
            p_value = statistical_tests.get('p_value', 1.0)
            effect_size = statistical_tests.get('effect_size', 0.0)
            
            fig.add_trace(
                go.Bar(x=['P-value', 'Effect Size'], y=[p_value, abs(effect_size)],
                      marker_color=[self.error_color if p_value < 0.05 else self.blue_color, self.warning_color],
                      name='Statistical Tests'),
                row=3, col=1
            )
        
        # Add episode length histogram
        episode_lengths = []
        for seed, trial_data in trials.items():
            for team in ['red', 'blue']:
                if trial_data.get(team, {}).get('success', False):
                    eval_results = trial_data[team]['evaluation_results']
                    lengths = eval_results.get('statistical_data', {}).get('episode_lengths', [])
                    episode_lengths.extend(lengths)
        
        if episode_lengths:
            fig.add_trace(
                go.Histogram(x=episode_lengths, name='Episode Lengths',
                           marker_color=self.success_color, opacity=0.7),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="ThreatSim Interactive Dashboard",
            showlegend=True,
            height=1000,
            template="plotly_white"
        )
        
        # Save interactive plot
        pyo.plot(fig, filename=str(self.interactive_dir / 'dashboard.html'), auto_open=False)
        
        logger.success("Interactive dashboard created")
    
    def _create_training_animation(self, results: Dict[str, Any]) -> None:
        """Create animated training progress visualization."""
        logger.info("Creating training animation")
        
        # Extract training data
        trials = results.get('trials', {})
        if not trials:
            return
        
        # Get the longest reward sequence
        max_length = 0
        best_red_rewards = []
        best_blue_rewards = []
        
        for seed, trial_data in trials.items():
            if trial_data.get('red', {}).get('success', False):
                red_eval = trial_data['red']['evaluation_results']
                red_rewards = red_eval.get('statistical_data', {}).get('episode_rewards', [])
                if len(red_rewards) > max_length:
                    max_length = len(red_rewards)
                    best_red_rewards = red_rewards
            
            if trial_data.get('blue', {}).get('success', False):
                blue_eval = trial_data['blue']['evaluation_results']
                blue_rewards = blue_eval.get('statistical_data', {}).get('episode_rewards', [])
                if len(blue_rewards) > len(best_blue_rewards):
                    best_blue_rewards = blue_rewards
        
        if not best_red_rewards or not best_blue_rewards:
            return
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            ax.clear()
            
            # Plot up to current frame
            episodes = range(1, frame + 2)
            red_data = best_red_rewards[:frame + 1]
            blue_data = best_blue_rewards[:frame + 1]
            
            ax.plot(episodes, red_data, color=self.red_color, 
                   linewidth=3, label='Red Team', marker='o', markersize=4)
            ax.plot(episodes, blue_data, color=self.blue_color, 
                   linewidth=3, label='Blue Team', marker='s', markersize=4)
            
            # Add moving averages
            if len(red_data) > 10:
                red_ma = pd.Series(red_data).rolling(window=10).mean()
                blue_ma = pd.Series(blue_data).rolling(window=10).mean()
                ax.plot(episodes, red_ma, color=self.red_color, 
                       linestyle='--', alpha=0.7, linewidth=2)
                ax.plot(episodes, blue_ma, color=self.blue_color, 
                       linestyle='--', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'Training Progress Animation - Episode {frame + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-limits
            all_rewards = best_red_rewards + best_blue_rewards
            ax.set_ylim(min(all_rewards) * 1.1, max(all_rewards) * 1.1)
            ax.set_xlim(0, len(best_red_rewards) + 10)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=min(len(best_red_rewards), 200),
                                     interval=100, blit=False)
        
        # Save animation
        anim.save(str(self.plots_dir / 'training_animation.gif'), writer='pillow', fps=10)
        plt.close()
        
        logger.success("Training animation created")
    
    def create_publication_figures(self, results: Dict[str, Any]) -> None:
        """Create publication-quality figures."""
        logger.info("Creating publication-quality figures")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # Create main publication figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ThreatSim Training Results', fontsize=18, fontweight='bold')
        
        # Extract data
        trials = results.get('trials', {})
        stats_analysis = results.get('statistical_analysis', {})
        team_performance = stats_analysis.get('team_performance', {})
        
        # Publication Figure 1: Training Convergence
        red_rewards = []
        blue_rewards = []
        
        for seed, trial_data in trials.items():
            if trial_data.get('red', {}).get('success', False):
                red_eval = trial_data['red']['evaluation_results']
                red_rewards.extend(red_eval.get('statistical_data', {}).get('episode_rewards', []))
            
            if trial_data.get('blue', {}).get('success', False):
                blue_eval = trial_data['blue']['evaluation_results']
                blue_rewards.extend(blue_eval.get('statistical_data', {}).get('episode_rewards', []))
        
        if red_rewards and blue_rewards:
            window = 50
            red_smooth = pd.Series(red_rewards).rolling(window=window).mean()
            blue_smooth = pd.Series(blue_rewards).rolling(window=window).mean()
            
            axes[0, 0].plot(red_smooth, color=self.red_color, linewidth=2, label='Red Team')
            axes[0, 0].plot(blue_smooth, color=self.blue_color, linewidth=2, label='Blue Team')
            axes[0, 0].fill_between(range(len(red_smooth)), red_smooth, alpha=0.3, color=self.red_color)
            axes[0, 0].fill_between(range(len(blue_smooth)), blue_smooth, alpha=0.3, color=self.blue_color)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Training Convergence')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Publication Figure 2: Performance Comparison
        if team_performance:
            red_stats = team_performance.get('red_team', {})
            blue_stats = team_performance.get('blue_team', {})
            
            teams = ['Red Team', 'Blue Team']
            win_rates = [red_stats.get('mean_win_rate', 0), blue_stats.get('mean_win_rate', 0)]
            errors = [red_stats.get('std_win_rate', 0), blue_stats.get('std_win_rate', 0)]
            
            bars = axes[0, 1].bar(teams, win_rates, yerr=errors, capsize=5,
                                 alpha=0.7, edgecolor='black')
            # Set colors manually for each bar
            bars[0].set_facecolor(self.red_color)
            bars[1].set_facecolor(self.blue_color)
            axes[0, 1].set_ylabel('Win Rate (%)')
            axes[0, 1].set_title('Performance Comparison')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add significance test result
            statistical_tests = stats_analysis.get('statistical_tests', {})
            p_value = statistical_tests.get('p_value', 1.0)
            if p_value < 0.05:
                axes[0, 1].text(0.5, max(win_rates) * 1.1, '*', ha='center', va='center',
                               fontsize=20, fontweight='bold')
        
        # Publication Figure 3: Balance Analysis
        balance_validation = results.get('balance_validation', {})
        balance_metrics = balance_validation.get('balance_metrics', {})
        
        if balance_metrics:
            target_rate = balance_metrics.get('target_red_win_rate', 0) * 100
            actual_rate = balance_metrics.get('red_win_rate', 0) * 100
            
            axes[1, 0].bar(['Target', 'Actual'], [target_rate, actual_rate],
                          color=[self.success_color, self.warning_color], alpha=0.7, edgecolor='black')
            axes[1, 0].set_ylabel('Red Team Win Rate (%)')
            axes[1, 0].set_title('Balance Validation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add deviation annotation
            deviation = abs(actual_rate - target_rate)
            axes[1, 0].text(0.5, max(target_rate, actual_rate) * 0.8,
                           f'Deviation: {deviation:.1f}%', ha='center', va='center',
                           fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
                           facecolor='lightgreen' if deviation < 10 else 'lightcoral'))
        
        # Publication Figure 4: Statistical Summary
        if statistical_tests:
            p_value = statistical_tests.get('p_value', 1.0)
            effect_size = statistical_tests.get('effect_size', 0.0)
            power = statistical_tests.get('statistical_power', 0.0)
            
            metrics = ['P-value', 'Effect Size', 'Power']
            values = [p_value, abs(effect_size), power]
            
            bars = axes[1, 1].bar(metrics, values, color=[self.error_color, self.warning_color, self.success_color],
                                 alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Statistical Summary')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add significance lines
            axes[1, 1].axhline(y=0.05, color=self.error_color, linestyle='--', alpha=0.5, label='α = 0.05')
            axes[1, 1].axhline(y=0.8, color=self.success_color, linestyle='--', alpha=0.5, label='Power = 0.8')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'publication_figure.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / 'publication_figure.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.success("Publication-quality figures created")
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate text summary of visualization results."""
        summary = []
        summary.append("ThreatSim Visualization Summary")
        summary.append("=" * 50)
        
        # Training overview
        trials = results.get('trials', {})
        if trials:
            summary.append(f"Training completed with {len(trials)} seeds")
            
            # Extract performance metrics
            red_wins = []
            blue_wins = []
            
            for seed, trial_data in trials.items():
                if trial_data.get('red', {}).get('success', False):
                    red_eval = trial_data['red']['evaluation_results']
                    red_wins.append(red_eval.get('win_rate', 0))
                
                if trial_data.get('blue', {}).get('success', False):
                    blue_eval = trial_data['blue']['evaluation_results']
                    blue_wins.append(blue_eval.get('win_rate', 0))
            
            if red_wins and blue_wins:
                summary.append(f"Red Team: {np.mean(red_wins):.1f}% ± {np.std(red_wins):.1f}% win rate")
                summary.append(f"Blue Team: {np.mean(blue_wins):.1f}% ± {np.std(blue_wins):.1f}% win rate")
        
        # Balance analysis
        balance_validation = results.get('balance_validation', {})
        balance_metrics = balance_validation.get('balance_metrics', {})
        
        if balance_metrics:
            target_rate = balance_metrics.get('target_red_win_rate', 0) * 100
            actual_rate = balance_metrics.get('red_win_rate', 0) * 100
            deviation = abs(actual_rate - target_rate)
            
            summary.append(f"Balance Analysis:")
            summary.append(f"  Target: {target_rate:.1f}%, Actual: {actual_rate:.1f}%")
            summary.append(f"  Deviation: {deviation:.1f}%")
            summary.append(f"  Status: {'Balanced' if deviation < 10 else 'Needs Adjustment'}")
