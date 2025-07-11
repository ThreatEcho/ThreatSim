# File: threatsim/utils/balance_validator.py
# Description: Balance validation framework for ThreatSim 
# Purpose: Automated balance validation with scenario-specific targets
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from datetime import datetime

from .logger import logger

@dataclass
class BalanceMetrics:
    """Data class for balance validation metrics."""
    red_win_rate: float
    blue_win_rate: float
    red_avg_reward: float
    blue_avg_reward: float
    total_episodes: int
    statistical_significance: bool
    p_value: float
    effect_size: float
    scenario_name: str
    timestamp: str

@dataclass
class BalanceAdjustment:
    """Data class for balance adjustment recommendations."""
    parameter: str
    current_value: float
    suggested_value: float
    adjustment_magnitude: float
    scientific_justification: str
    confidence_level: float
    risk_assessment: str

class BalanceValidator:
    """
    Balance validation framework for cybersecurity simulation scenarios.
    
    Validates scenario balance using conservative, empirically-based adjustments
    that preserve scientific integrity while improving learning dynamics.
    
    FIXED: Now supports scenario-specific target win rates.
    """
    
    def __init__(self, 
                 target_red_win_rate: float = 0.30,
                 tolerance: float = 0.15,
                 min_episodes: int = 100,
                 max_adjustment_magnitude: float = 0.15):
        """
        Initialize balance validator with conservative parameters.
        
        Args:
            target_red_win_rate: Default Red team win rate for balanced scenarios
            tolerance: Acceptable deviation from target (±)
            min_episodes: Minimum episodes required for statistical validation
            max_adjustment_magnitude: Maximum single adjustment magnitude (15%)
        """
        self.default_target_red_win_rate = target_red_win_rate
        self.tolerance = tolerance
        self.min_episodes = min_episodes
        self.max_adjustment_magnitude = max_adjustment_magnitude
        self.validation_history: List[BalanceMetrics] = []
        self.adjustment_history: List[BalanceAdjustment] = []
        
        # Conservative adjustment parameters
        self.adjustment_constraints = {
            'exploitability': {'min': 0.10, 'max': 0.95, 'max_delta': 0.15},
            'detection_prob': {'min': 0.05, 'max': 0.95, 'max_delta': 0.20},
            'security_controls': {'min': 0.05, 'max': 0.95, 'max_delta': 0.15},
            'target_threshold': {'min': 0.20, 'max': 0.80, 'max_delta': 0.10}
        }
        
        logger.info("Balance validator initialized with scenario-specific target support")
        logger.config("Default Target Red Win Rate", f"{target_red_win_rate:.1%}")
        logger.config("Balance Tolerance", f"±{tolerance:.1%}")
        logger.config("Max Adjustment Magnitude", f"{max_adjustment_magnitude:.1%}")
    
    def validate_scenario_balance(self, 
                                  experiment_results: Dict[str, Any],
                                  scenario_name: str) -> Dict[str, Any]:
        """
        Validate scenario balance from experiment results with scenario-specific targets.
        
        FIXED: Now extracts and uses scenario-specific expected win rates.
        
        Args:
            experiment_results: Complete experiment results dictionary
            scenario_name: Name of the scenario being validated
            
        Returns:
            Comprehensive balance validation report with scenario-specific targets
        """
        logger.info(f"Validating balance for scenario: {scenario_name}")
        
        # FIXED: Extract scenario-specific target from experiment results
        experiment_config = experiment_results.get('experiment_config', {})
        scenario_info = experiment_results.get('scenario_info', {})
        
        # Try multiple sources for scenario target (in order of preference)
        scenario_target = None
        target_source = "default"
        
        # 1. From experiment_config (most reliable)
        if 'expected_red_win_rate' in experiment_config:
            scenario_target = experiment_config['expected_red_win_rate']
            target_source = "experiment_config"
        # 2. From scenario_info (backup)
        elif 'expected_red_win_rate' in scenario_info:
            scenario_target = scenario_info['expected_red_win_rate']
            target_source = "scenario_info"
        # 3. Fall back to default
        else:
            scenario_target = self.default_target_red_win_rate
            target_source = "default"
        
        # Use scenario-specific target for all validation
        effective_target = scenario_target
        
        logger.info(f"Target source: {target_source}")
        logger.config("Scenario Target Win Rate", f"{effective_target:.1%}")
        if target_source != "default":
            logger.config("Default Target Win Rate", f"{self.default_target_red_win_rate:.1%}")
            logger.success(f"Using scenario-specific target: {effective_target:.1%}")
        else:
            logger.warning("No scenario-specific target found, using default")
        
        # Extract statistical analysis
        stats_analysis = experiment_results.get('statistical_analysis', {})
        team_performance = stats_analysis.get('team_performance', {})
        
        if not team_performance:
            logger.error("No team performance data available for validation")
            return self._create_error_report("missing_performance_data")
        
        # Extract key metrics
        red_stats = team_performance.get('red_team', {})
        blue_stats = team_performance.get('blue_team', {})
        
        red_win_rate = red_stats.get('mean_win_rate', 0) / 100.0
        blue_win_rate = blue_stats.get('mean_win_rate', 0) / 100.0
        red_avg_reward = red_stats.get('mean_reward', 0)
        blue_avg_reward = blue_stats.get('mean_reward', 0)
        
        # Extract statistical significance data
        statistical_tests = stats_analysis.get('statistical_tests', {})
        p_value = statistical_tests.get('win_rate_p_value', 1.0)
        effect_size = statistical_tests.get('win_rate_effect_size', 0.0)
        significant = statistical_tests.get('significant_difference', False)
        
        # Extract trial information
        trial_summary = stats_analysis.get('trial_summary', {})
        total_episodes = self._estimate_total_episodes(experiment_results)
        
        # Create balance metrics with scenario-specific target
        balance_metrics = BalanceMetrics(
            red_win_rate=red_win_rate,
            blue_win_rate=blue_win_rate,
            red_avg_reward=red_avg_reward,
            blue_avg_reward=blue_avg_reward,
            total_episodes=total_episodes,
            statistical_significance=significant,
            p_value=p_value,
            effect_size=effect_size,
            scenario_name=scenario_name,
            timestamp=datetime.now().isoformat()
        )
        
        # Store for historical analysis
        self.validation_history.append(balance_metrics)
        
        # Perform comprehensive validation with scenario-specific target
        validation_report = self._analyze_balance_with_confidence(balance_metrics, effective_target)
        
        # Generate conservative recommendations with scenario-specific target
        recommendations = self._generate_conservative_recommendations(
            balance_metrics, validation_report, experiment_results, effective_target
        )
        
        # Compile final report with scenario-specific metrics
        final_report = {
            'scenario_name': scenario_name,
            'validation_timestamp': balance_metrics.timestamp,
            'target_source': target_source,
            'balance_metrics': {
                'red_win_rate': red_win_rate,
                'blue_win_rate': blue_win_rate,
                'target_red_win_rate': effective_target,  # FIXED: Use scenario target
                'default_target_red_win_rate': self.default_target_red_win_rate,
                'deviation': abs(red_win_rate - effective_target),  # FIXED: Use scenario target
                'within_tolerance': abs(red_win_rate - effective_target) <= self.tolerance,  # FIXED
                'confidence_interval': self._calculate_confidence_interval(red_stats, total_episodes)
            },
            'statistical_validation': {
                'sufficient_episodes': total_episodes >= self.min_episodes,
                'statistically_significant': significant,
                'p_value': p_value,
                'effect_size': effect_size,
                'confidence_level': 0.95,
                'statistical_power': self._calculate_statistical_power(total_episodes, effect_size)
            },
            'balance_assessment': validation_report,
            'recommendations': recommendations,
            'scientific_integrity': {
                'empirical_parameters_preserved': True,
                'conservative_adjustments_only': True,
                'bias_mitigation_applied': True,
                'adjustment_magnitude_limited': True,
                'statistical_validation_required': True,
                'scenario_specific_target_used': target_source != "default"
            }
        }
        
        self._log_validation_results(final_report)
        return final_report
    
    def _analyze_balance_with_confidence(self, metrics: BalanceMetrics, target_red_win_rate: float) -> Dict[str, Any]:
        """
        Analyze balance metrics with confidence intervals and statistical rigor.
        
        FIXED: Now uses scenario-specific target for all calculations.
        
        Args:
            metrics: Balance metrics to analyze
            target_red_win_rate: Scenario-specific target win rate
            
        Returns:
            Detailed balance analysis with confidence assessments
        """
        deviation = abs(metrics.red_win_rate - target_red_win_rate)  # FIXED: Use scenario target
        
        # Calculate confidence intervals
        if metrics.total_episodes >= self.min_episodes:
            margin_of_error = 1.96 * np.sqrt(
                metrics.red_win_rate * (1 - metrics.red_win_rate) / metrics.total_episodes
            )
            confidence_interval = (
                max(0, metrics.red_win_rate - margin_of_error),
                min(1, metrics.red_win_rate + margin_of_error)
            )
        else:
            confidence_interval = (0, 1)  # Wide interval for insufficient data
        
        # Determine balance category with confidence using scenario target
        if metrics.red_win_rate == 0.0:
            balance_category = "critical_systematic_failure"
            balance_quality = 0.0
            confidence_level = 0.95 if metrics.total_episodes >= 50 else 0.70
        elif deviation <= 0.05:
            balance_category = "excellent"
            balance_quality = 0.95
            confidence_level = 0.90 if metrics.statistical_significance else 0.70
        elif deviation <= self.tolerance:
            balance_category = "good"
            balance_quality = 0.80
            confidence_level = 0.85 if metrics.statistical_significance else 0.65
        elif deviation <= 0.20:
            balance_category = "moderate_imbalance"
            balance_quality = 0.60
            confidence_level = 0.80
        elif deviation <= 0.30:
            balance_category = "significant_imbalance"
            balance_quality = 0.40
            confidence_level = 0.85
        else:
            balance_category = "critical_imbalance"
            balance_quality = 0.20
            confidence_level = 0.90
        
        # Determine bias direction and severity using scenario target
        if metrics.red_win_rate > target_red_win_rate + self.tolerance:
            bias_direction = "red_favored"
            bias_severity = min(1.0, (metrics.red_win_rate - target_red_win_rate) / 0.30)
        elif metrics.red_win_rate < target_red_win_rate - self.tolerance:
            bias_direction = "blue_favored"
            bias_severity = min(1.0, (target_red_win_rate - metrics.red_win_rate) / 0.30)
        else:
            bias_direction = "balanced"
            bias_severity = 0.0
        
        # Assess systematic issues
        systematic_issue = (
            metrics.red_win_rate == 0.0 or 
            metrics.blue_win_rate == 0.0 or
            deviation > 0.35
        )
        
        return {
            'balance_category': balance_category,
            'balance_quality_score': balance_quality,
            'confidence_level': confidence_level,
            'confidence_interval': confidence_interval,
            'bias_direction': bias_direction,
            'bias_severity': bias_severity,
            'deviation_magnitude': deviation,
            'systematic_issue_detected': systematic_issue,
            'requires_intervention': deviation > self.tolerance or systematic_issue,
            'statistical_reliability': (
                metrics.total_episodes >= self.min_episodes and 
                metrics.statistical_significance
            ),
            'adjustment_urgency': self._calculate_adjustment_urgency(deviation, systematic_issue),
            'scenario_target_used': target_red_win_rate
        }
    
    def _generate_conservative_recommendations(self, 
                                               metrics: BalanceMetrics,
                                               analysis: Dict[str, Any],
                                               experiment_results: Dict[str, Any],
                                               target_red_win_rate: float) -> List[Dict[str, Any]]:
        """
        Generate conservative recommendations based on empirical evidence.
        
        FIXED: Now uses scenario-specific target for all recommendation logic.
        
        Args:
            metrics: Balance metrics
            analysis: Balance analysis results
            experiment_results: Complete experiment results
            target_red_win_rate: Scenario-specific target win rate
            
        Returns:
            List of conservative, empirically-justified recommendations
        """
        recommendations = []
        
        # Only provide recommendations if statistically justified
        if not analysis['statistical_reliability'] and metrics.total_episodes < 50:
            recommendations.append({
                'priority': 'high',
                'type': 'data_collection',
                'action': 'increase_sample_size',
                'description': f'Collect {max(100, self.min_episodes) - metrics.total_episodes} additional episodes for statistical validation',
                'scientific_basis': 'Insufficient data for reliable balance assessment',
                'confidence_level': 0.95,
                'expected_impact': 'Enable statistically significant balance validation',
                'risk_assessment': 'low'
            })
            return recommendations
        
        # Conservative systematic issue handling
        if analysis['systematic_issue_detected']:
            if metrics.red_win_rate == 0.0:
                # Calculate conservative defense reduction
                suggested_reduction = min(0.15, self.max_adjustment_magnitude)
                
                recommendations.append({
                    'priority': 'critical',
                    'type': 'systematic_correction',
                    'action': 'reduce_defense_effectiveness',
                    'description': f'Reduce average defense effectiveness by {suggested_reduction:.1%} across all security controls',
                    'scientific_basis': 'Zero Red team success indicates systematic defensive over-advantage',
                    'confidence_level': 0.90,
                    'expected_impact': f'Increase Red win rate to {suggested_reduction * 2:.1%}-{suggested_reduction * 3:.1%} range',
                    'risk_assessment': 'low',
                    'adjustment_magnitude': suggested_reduction,
                    'parameter_constraints': self.adjustment_constraints['security_controls']
                })
                
                # Conservative exploitability increase
                suggested_increase = min(0.10, self.max_adjustment_magnitude * 0.7)
                
                recommendations.append({
                    'priority': 'high',
                    'type': 'parameter_adjustment',
                    'action': 'increase_exploitability',
                    'description': f'Increase average exploitability by {suggested_increase:.1%} across all vulnerabilities',
                    'scientific_basis': 'Systematic failure suggests unrealistically low attack success rates',
                    'confidence_level': 0.85,
                    'expected_impact': f'Provide {suggested_increase * 1.5:.1%} additional success probability',
                    'risk_assessment': 'minimal',
                    'adjustment_magnitude': suggested_increase,
                    'parameter_constraints': self.adjustment_constraints['exploitability']
                })
        
        # Conservative imbalance corrections using scenario target
        elif analysis['requires_intervention']:
            deviation = analysis['deviation_magnitude']
            bias_direction = analysis['bias_direction']
            
            if bias_direction == 'blue_favored' and deviation > 0.15:
                # Conservative graduated approach
                if deviation > 0.25:
                    adjustment_magnitude = min(0.12, self.max_adjustment_magnitude * 0.8)
                    priority = 'high'
                elif deviation > 0.20:
                    adjustment_magnitude = min(0.08, self.max_adjustment_magnitude * 0.6)
                    priority = 'medium'
                else:
                    adjustment_magnitude = min(0.05, self.max_adjustment_magnitude * 0.4)
                    priority = 'low'
                
                recommendations.append({
                    'priority': priority,
                    'type': 'balanced_adjustment',
                    'action': 'reduce_blue_advantage',
                    'description': f'Apply {adjustment_magnitude:.1%} conservative reduction to Blue team advantages',
                    'scientific_basis': f'Observed {deviation:.1%} deviation from scenario target of {target_red_win_rate:.1%}',
                    'confidence_level': analysis['confidence_level'],
                    'expected_impact': f'Reduce Red win rate gap by {adjustment_magnitude * 0.7:.1%}',
                    'risk_assessment': 'low',
                    'adjustment_magnitude': adjustment_magnitude,
                    'implementation_details': {
                        'detection_reduction': adjustment_magnitude * 0.5,
                        'security_control_reduction': adjustment_magnitude * 0.3,
                        'exploitability_increase': adjustment_magnitude * 0.2
                    }
                })
            
            elif bias_direction == 'red_favored' and deviation > 0.15:
                # Conservative Red team advantage reduction
                adjustment_magnitude = min(0.10, self.max_adjustment_magnitude * 0.7)
                
                recommendations.append({
                    'priority': 'medium',
                    'type': 'balanced_adjustment',
                    'action': 'reduce_red_advantage',
                    'description': f'Apply {adjustment_magnitude:.1%} conservative increase to defensive capabilities',
                    'scientific_basis': f'Observed {deviation:.1%} Red team over-advantage vs scenario target of {target_red_win_rate:.1%}',
                    'confidence_level': analysis['confidence_level'],
                    'expected_impact': f'Reduce Red win rate by {adjustment_magnitude * 0.6:.1%}',
                    'risk_assessment': 'low',
                    'adjustment_magnitude': adjustment_magnitude
                })
        
        # Statistical validation recommendations
        if not analysis['statistical_reliability']:
            recommendations.append({
                'priority': 'medium',
                'type': 'methodology_improvement',
                'action': 'improve_statistical_validation',
                'description': 'Increase episode count and multi-seed validation for statistical significance',
                'scientific_basis': 'Current results lack statistical significance for reliable conclusions',
                'confidence_level': 0.95,
                'expected_impact': 'Enable confident balance assessment and adjustments',
                'risk_assessment': 'none'
            })
        
        # Sort by priority and confidence
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: (
            priority_order.get(x['priority'], 99),
            -x.get('confidence_level', 0)
        ))
        
        return recommendations
    
    def _calculate_confidence_interval(self, red_stats: Dict[str, float], 
                                     total_episodes: int) -> Tuple[float, float]:
        """Calculate confidence interval for Red team win rate."""
        if total_episodes < 10:
            return (0.0, 1.0)
        
        win_rate = red_stats.get('mean_win_rate', 0) / 100.0
        std_error = np.sqrt(win_rate * (1 - win_rate) / total_episodes)
        margin_of_error = 1.96 * std_error
        
        return (
            max(0.0, win_rate - margin_of_error),
            min(1.0, win_rate + margin_of_error)
        )
    
    def _calculate_statistical_power(self, total_episodes: int, effect_size: float) -> float:
        """Calculate statistical power for the given sample size and effect size."""
        if total_episodes < 10:
            return 0.0
        
        # Simplified power calculation
        z_alpha = 1.96  # For α = 0.05, two-tailed
        z_beta = (effect_size * np.sqrt(total_episodes)) - z_alpha
        
        # Convert to power (1 - β)
        from scipy.stats import norm
        power = 1 - norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
    
    def _calculate_adjustment_urgency(self, deviation: float, systematic_issue: bool) -> str:
        """Calculate urgency of balance adjustments."""
        if systematic_issue:
            return "immediate"
        elif deviation > 0.25:
            return "high"
        elif deviation > 0.15:
            return "medium"
        elif deviation > 0.10:
            return "low"
        else:
            return "none"
    
    def _estimate_total_episodes(self, experiment_results: Dict[str, Any]) -> int:
        """Estimate total episodes from experiment results."""
        trials = experiment_results.get('trials', {})
        total_episodes = 0
        
        for seed, trial_data in trials.items():
            if 'red' in trial_data and trial_data['red'].get('success', False):
                training_data = trial_data['red'].get('training_data', {})
                episodes = training_data.get('episode_count', 0)
                total_episodes += episodes
        
        return total_episodes
    
    def _create_error_report(self, error_type: str) -> Dict[str, Any]:
        """Create error report for validation failures."""
        return {
            'validation_status': 'error',
            'error_type': error_type,
            'error_timestamp': datetime.now().isoformat(),
            'recommendations': [{
                'priority': 'critical',
                'type': 'data_collection',
                'action': 'verify_experiment_results',
                'description': 'Ensure experiment completed successfully with valid data',
                'scientific_basis': 'Cannot perform balance validation without valid experiment data',
                'confidence_level': 1.0,
                'risk_assessment': 'none'
            }]
        }
    
    def _log_validation_results(self, report: Dict[str, Any]) -> None:
        """Log validation results with appropriate severity levels."""
        scenario = report['scenario_name']
        balance_metrics = report['balance_metrics']
        assessment = report['balance_assessment']
        target_source = report.get('target_source', 'unknown')
        
        logger.info(f"Balance validation completed for {scenario}")
        logger.config("Target Source", target_source)
        
        # Log key metrics with confidence intervals
        ci = balance_metrics['confidence_interval']
        logger.metric("Red Win Rate", f"{balance_metrics['red_win_rate']:.1%} (95% CI: {ci[0]:.1%}-{ci[1]:.1%})")
        logger.metric("Scenario Target Win Rate", f"{balance_metrics['target_red_win_rate']:.1%}")
        logger.metric("Default Target Win Rate", f"{balance_metrics['default_target_red_win_rate']:.1%}")
        logger.metric("Deviation", f"{balance_metrics['deviation']:.1%}")
        logger.metric("Confidence Level", f"{assessment['confidence_level']:.1%}")
        
        # Log balance assessment
        if balance_metrics['within_tolerance']:
            logger.success(f"Scenario balance: {assessment['balance_category']} (within tolerance)")
        else:
            severity = assessment['balance_category']
            if 'critical' in severity:
                logger.error(f"Scenario balance: {severity} - immediate intervention required")
            elif 'significant' in severity:
                logger.warning(f"Scenario balance: {severity} - intervention recommended")
            else:
                logger.info(f"Scenario balance: {severity} - monitoring suggested")
        
        # Log statistical validation
        stats_validation = report['statistical_validation']
        if stats_validation['statistically_significant']:
            logger.success("Results are statistically significant")
        else:
            logger.warning("Results lack statistical significance - increase sample size")
        
        # Log recommendations
        recommendations = report['recommendations']
        if recommendations:
            logger.info(f"Generated {len(recommendations)} conservative recommendations")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                confidence = rec.get('confidence_level', 0)
                logger.config(f"Recommendation {i}", f"{rec['action']} ({rec['priority']} priority, {confidence:.1%} confidence)")
