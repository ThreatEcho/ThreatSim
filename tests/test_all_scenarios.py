# File: tests/test_all_scenarios.py
# Description: Scenario compatibility test 
# Purpose: Verify all scenarios load and function correctly
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from threatsim.envs.wrappers import RedTeamEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from threatsim.utils.logger import logger

def test_scenario(scenario_path: str, quick_test: bool = True):
    """Test a single scenario with detailed logging"""
    scenario_name = Path(scenario_path).stem
    logger.info(f"Testing scenario: {scenario_name}")
    
    try:
        # Test environment creation
        logger.debug("Creating environment")
        env = RedTeamEnv(
            scenario_path=scenario_path,
            max_steps=10 if quick_test else 50,
            blue_policy="passive"
        )
        
        # Test reset and step
        logger.debug("Testing basic environment operations")
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Test vectorization and basic training if not quick test
        if not quick_test:
            logger.debug("Testing advanced functionality")
            def make_env():
                return RedTeamEnv(
                    scenario_path=scenario_path,
                    max_steps=20,
                    blue_policy="passive"
                )
            
            vec_env = DummyVecEnv([make_env])
            model = PPO("MlpPolicy", vec_env, verbose=0, seed=42)
            model.learn(total_timesteps=64, progress_bar=False)
        
        logger.success(f"{scenario_name}: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"{scenario_name}: FAILED - {e}")
        return False

def analyze_scenario_metadata(scenario_files):
    """Analyze and display scenario metadata"""
    logger.header("Scenario Analysis")
    
    difficulties = {}
    attack_types = {}
    
    for scenario_file in scenario_files:
        try:
            import yaml
            with open(scenario_file, 'r') as f:
                scenario_data = yaml.safe_load(f)
            
            training_config = scenario_data.get('training_config', {})
            difficulty = training_config.get('difficulty', 'unknown')
            attack_type = training_config.get('attack_type', 'general')
            
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            
        except Exception as e:
            logger.warning(f"Could not analyze {scenario_file.name}: {e}")
    
    # Display analysis
    logger.info("Difficulty distribution:")
    for difficulty, count in sorted(difficulties.items()):
        logger.metric(difficulty.capitalize(), count)
    
    logger.info("Attack type distribution:")
    for attack_type, count in sorted(attack_types.items()):
        logger.metric(attack_type.replace('_', ' ').title(), count)

def main():
    """Test all scenarios with comprehensive reporting"""
    logger.header("ThreatSim Scenario Compatibility Test")
    
    # Find all scenario files
    scenarios_dir = project_root / "scenarios"
    scenario_files = list(scenarios_dir.glob("*.yaml"))
    
    if not scenario_files:
        logger.error("No scenario files found in scenarios/")
        return
    
    logger.config("Scenarios directory", scenarios_dir)
    logger.config("Total scenarios found", len(scenario_files))
    
    # Analyze scenario metadata
    analyze_scenario_metadata(scenario_files)
    
    # Test each scenario
    logger.header("Running Compatibility Tests")
    passed = 0
    failed = 0
    failed_scenarios = []
    
    for scenario_file in sorted(scenario_files):
        if test_scenario(str(scenario_file), quick_test=True):
            passed += 1
        else:
            failed += 1
            failed_scenarios.append(scenario_file.stem)
    
    # Comprehensive summary
    logger.header("Test Results Summary")
    logger.metric("Total scenarios", len(scenario_files))
    logger.metric("Passed", passed)
    logger.metric("Failed", failed)
    logger.metric("Success rate", f"{passed/(passed+failed)*100:.1f}%")
    
    if failed_scenarios:
        logger.warning("Failed scenarios:")
        for scenario in failed_scenarios:
            logger.error(f"  - {scenario}")
    
    # Final assessment
    if failed == 0:
        logger.success("All scenarios are compatible and functional")
        logger.info("The scenario suite is ready for production use")
    elif passed > failed:
        logger.warning(f"{failed} scenarios need attention, but majority functional")
        logger.info("Core system is stable with some scenario-specific issues")
    else:
        logger.error("Critical compatibility issues detected")
        logger.warning("System requires investigation before production use")

if __name__ == "__main__":
    main()
