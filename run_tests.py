# File: run_tests.py
# Description: Test runner for ThreatSim
# Purpose: Execute all scripts in "test/" directory
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import subprocess
import sys
from pathlib import Path

# Add project root to path for logger import
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from threatsim.utils.logger import logger

def run_test(test_script, description):
    """Run a test script and report results with symbolic logging"""
    logger.info(f"Running: {description}")
    logger.debug(f"Executing: tests/{test_script}")
    
    try:
        result = subprocess.run([
            sys.executable, f"tests/{test_script}"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.success(f"{description}: PASSED")
            if result.stdout.strip():
                # Show output but filter out the subprocess's own logging
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():  # Skip empty lines
                        print(f"  {line}")
        else:
            logger.error(f"{description}: FAILED")
            if result.stderr.strip():
                logger.debug("Error output:")
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
            if result.stdout.strip():
                logger.debug("Standard output:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
                
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"{description}: TIMEOUT - Test exceeded 5 minutes")
        return False
    except Exception as e:
        logger.error(f"{description}: EXCEPTION - {e}")
        return False

def main():
    """Run all tests with comprehensive reporting"""
    logger.header("ThreatSim Test Suite")
    logger.info("Executing all tests from project root")
    logger.config("Python executable", sys.executable)
    logger.config("Working directory", Path.cwd())
    
    tests = [
        ("minimal_test.py", "Minimal Functionality Test"),
        ("debug_training.py", "Debug Training System"),
        ("test_all_scenarios.py", "Scenario Compatibility Test")
    ]
    
    logger.config("Total tests", len(tests))
    
    passed = 0
    failed_tests = []
    
    for test_script, description in tests:
        if run_test(test_script, description):
            passed += 1
        else:
            failed_tests.append(description)
    
    # Comprehensive summary
    logger.header("Test Suite Summary")
    logger.metric("Total tests", len(tests))
    logger.metric("Passed", passed)
    logger.metric("Failed", len(tests) - passed)
    logger.metric("Success rate", f"{passed/len(tests)*100:.1f}%")
    
    if failed_tests:
        logger.warning("Failed tests:")
        for test in failed_tests:
            logger.error(f"  - {test}")
    
    # Final assessment
    if passed == len(tests):
        logger.success("ALL TESTS PASSED")
        logger.info("ThreatSim system is fully operational")
        sys.exit(0)
    else:
        logger.error(f"{len(tests) - passed} test(s) failed")
        logger.warning("System requires attention before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()
