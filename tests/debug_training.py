# File: tests/debug_training.py
# Description: Debug script for systematic testing
# Purpose: Comprehensive environment and training validation
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import os
import sys
import traceback
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from threatsim.envs.wrappers import make_red_env, make_blue_env
from threatsim.utils.logger import logger

def test_environment_creation():
    """Test environment creation with detailed logging"""
    logger.info("Testing environment creation")
    logger.config("Working directory", os.getcwd())
    
    try:
        # Test Red environment
        logger.debug("Creating Red team environment")
        red_env = make_red_env(
            scenario_path="scenarios/simple_scenario.yaml",
            max_steps=20,
            blue_policy="heuristic"
        )
        logger.success("Red environment created successfully")
        
        # Test Blue environment  
        logger.debug("Creating Blue team environment")
        blue_env = make_blue_env(
            scenario_path="scenarios/simple_scenario.yaml",
            max_steps=20,
            red_strategy="apt"
        )
        logger.success("Blue environment created successfully")
        
        return red_env, blue_env
        
    except Exception as e:
        logger.error(f"Environment creation failed: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()
        return None, None

def test_environment_reset(env, env_name):
    """Test environment reset functionality"""
    logger.info(f"Testing {env_name} environment reset")
    
    try:
        obs, info = env.reset()
        logger.success(f"{env_name} environment reset successful")
        logger.config("Observation shape", obs.shape)
        logger.config("Observation space", env.observation_space)
        logger.config("Action space", env.action_space)
        return obs
    except Exception as e:
        logger.error(f"{env_name} environment reset failed: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()
        return None

def test_environment_step(env, env_name, obs):
    """Test environment step functionality"""
    logger.info(f"Testing {env_name} environment step")
    
    try:
        # Sample random action
        action = env.action_space.sample()
        logger.debug(f"Sampled action: {action}")
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        logger.success(f"{env_name} environment step successful")
        logger.metric("Reward", f"{reward:.3f}")
        logger.metric("Terminated", terminated)
        logger.config("Info keys", list(info.keys()))
        return True
    except Exception as e:
        logger.error(f"{env_name} environment step failed: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()
        return False

def test_vectorized_env(env, env_name):
    """Test vectorized environment wrapper"""
    logger.info(f"Testing {env_name} vectorized environment")
    
    try:
        def make_env():
            return env
        
        vec_env = DummyVecEnv([make_env])
        obs = vec_env.reset()
        logger.success(f"{env_name} vectorized environment created")
        logger.config("Vectorized observation shape", obs.shape)
        
        # Test step
        action = [env.action_space.sample()]
        next_obs, rewards, dones, infos = vec_env.step(action)
        logger.success(f"{env_name} vectorized environment step successful")
        
        return vec_env
    except Exception as e:
        logger.error(f"{env_name} vectorized environment failed: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()
        return None

def test_ppo_creation(vec_env, env_name):
    """Test PPO model creation"""
    logger.info(f"Testing PPO creation for {env_name}")
    
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=1,
            seed=42
        )
        logger.success(f"PPO model created for {env_name}")
        return model
    except Exception as e:
        logger.error(f"PPO creation failed for {env_name}: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()
        return None

def test_ppo_training(model, env_name, timesteps=100):
    """Test PPO training process"""
    logger.info(f"Testing PPO training for {env_name} ({timesteps} timesteps)")
    
    try:
        model.learn(total_timesteps=timesteps, progress_bar=False)
        logger.success(f"PPO training successful for {env_name}")
        return True
    except Exception as e:
        logger.error(f"PPO training failed for {env_name}: {e}")
        logger.debug("Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Run systematic debugging with comprehensive logging"""
    logger.header("ThreatSim Training Debug Script")
    
    # Test environment creation
    red_env, blue_env = test_environment_creation()
    if red_env is None or blue_env is None:
        logger.error("Cannot proceed - environment creation failed")
        return
    
    # Test both environments
    environments = [(red_env, "Red"), (blue_env, "Blue")]
    successful_envs = 0
    
    for env, env_name in environments:
        logger.header(f"Testing {env_name} Environment")
        
        # Test reset
        obs = test_environment_reset(env, env_name)
        if obs is None:
            logger.warning(f"Skipping further tests for {env_name} due to reset failure")
            continue
        
        # Test step
        if not test_environment_step(env, env_name, obs):
            logger.warning(f"Skipping further tests for {env_name} due to step failure")
            continue
        
        # Test vectorized environment
        vec_env = test_vectorized_env(env, env_name)
        if vec_env is None:
            logger.warning(f"Skipping PPO tests for {env_name} due to vectorization failure")
            continue
        
        # Test PPO creation
        model = test_ppo_creation(vec_env, env_name)
        if model is None:
            logger.warning(f"Skipping training test for {env_name} due to model creation failure")
            continue
        
        # Test minimal training
        if test_ppo_training(model, env_name, timesteps=100):
            logger.success(f"{env_name} environment fully functional")
            successful_envs += 1
        else:
            logger.error(f"{env_name} training verification failed")
    
    # Summary
    logger.header("Debug Summary")
    logger.metric("Environments tested", len(environments))
    logger.metric("Fully functional", successful_envs)
    logger.metric("Success rate", f"{successful_envs/len(environments)*100:.1f}%")
    
    if successful_envs == len(environments):
        logger.success("All environments are fully operational")
    elif successful_envs > 0:
        logger.warning("Some environments have issues but core functionality works")
    else:
        logger.error("Critical system failures detected")

if __name__ == "__main__":
    main()
