# File: tests/minimal_test.py
# Description: Minimal working test 
# Purpose: Basic functionality verification
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import os
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from threatsim.envs.wrappers import RedTeamEnv
from threatsim.utils.logger import logger

def minimal_test():
    """Absolute minimal test with symbolic logging"""
    logger.header("Minimal ThreatSim Test")
    logger.config("Working directory", os.getcwd())
    logger.config("Project root", project_root)
    
    try:
        # Create environment with minimal settings
        logger.info("Creating Red team environment")
        env = RedTeamEnv(
            scenario_path="scenarios/simple_scenario.yaml",
            max_steps=10,
            blue_policy="passive"
        )
        logger.success("Environment created")
        
        # Reset environment
        logger.info("Resetting environment")
        obs, info = env.reset(seed=42)
        logger.success(f"Environment reset - observation shape: {obs.shape}")
        
        # Take a few steps
        logger.info("Testing environment stepping")
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.debug(f"Step {i+1}: reward={reward:.2f}, done={terminated or truncated}")
            
            if terminated or truncated:
                obs, info = env.reset()
                logger.debug("Episode ended, environment reset")
        
        logger.success("Environment stepping verified")
        
        # Create vectorized environment
        logger.info("Creating vectorized environment")
        def make_env():
            return RedTeamEnv(
                scenario_path="scenarios/simple_scenario.yaml",
                max_steps=10,
                blue_policy="passive"
            )
        
        vec_env = DummyVecEnv([make_env])
        logger.success("Vectorized environment created")
        
        # Create PPO model
        logger.info("Creating PPO model")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-3,
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            verbose=0,  # Suppress PPO's own logging
            seed=42
        )
        logger.success("PPO model created")
        
        # Train for minimal timesteps
        logger.info("Starting minimal training (64 timesteps)")
        model.learn(total_timesteps=64, progress_bar=False)
        logger.success("Training completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.debug("Full traceback:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    
    if success:
        logger.success("All basic functionality verified")
        logger.info("The core ThreatSim components are working correctly")
    else:
        logger.error("Basic components have critical issues")
        logger.warning("Check the error details above for troubleshooting")
