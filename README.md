# ThreatSim: Scientific Cybersecurity Simulation Framework

**A scientifically rigorous multi-agent reinforcement learning framework for cybersecurity training and research.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

ThreatSim provides empirically-derived cybersecurity simulations supporting Red team (attacker) vs Blue team (defender) training scenarios. Built on industry threat intelligence and peer-reviewed research, ThreatSim enables reproducible cybersecurity research with statistical validation and comprehensive analysis capabilities.

### Key Features

- **Multi-Agent RL Environment**: Simultaneous Red vs Blue team training with realistic adversarial dynamics
- **MITRE ATT&CK Integration**: Comprehensive Tactics, Techniques, and Procedures (TTP) implementation
- **Realistic Network Modeling**: Enterprise network topologies with security control simulation
- **Scientific Validation**: Statistical significance testing with confidence intervals and reproducibility requirements
- **Comprehensive Analytics**: Automated statistical analysis, balance validation, and publication-quality visualizations
- **Configurable Training**: Multiple training modes from quick validation to research-grade experiments

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/threatecho/threatsim.git
cd threatsim

# Create virtual environment
python -m venv threatsim-env
source threatsim-env/bin/activate  # On Windows: threatsim-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tests/minimal_test.py
```

### Basic Usage

```bash
# Run interactive training system
python train_threatsim.py

# Quick system verification
python run_tests.py

# System diagnostics
python run_system_check.py
```

## Scenario Library

ThreatSim includes 8 scientifically validated scenarios covering major threat vectors:

| Scenario | Difficulty | Attack Type | Expected Red Win Rate |
|----------|------------|-------------|----------------------|
| **Simple** | Beginner | Basic Training | 30% |
| **Enterprise** | Realistic | Enterprise Compromise | 32% |
| **Ransomware** | Intermediate | Modern Ransomware | 40% |
| **Financial Crime** | Intermediate | Business Email Compromise | 34% |
| **Nation State** | Advanced | APT Espionage | 35% |
| **Supply Chain** | Expert | Software Supply Chain | 25% |
| **Insider Threat** | Beginner | Malicious Insider | 30% |

## Scientific Methodology

### Parameter Derivation

All simulation parameters are derived from authoritative sources:
- **Verizon Data Breach Investigations Report**
- **MITRE ATT&CK Enterprise Matrix**
- **SANS SOC Survey and Analysis**
- **CrowdStrike Global Threat Report**
- **NIST Cybersecurity Framework**

### Balance Validation

ThreatSim employs rigorous balance validation:
- **Scenario-Specific Targets**: Each scenario has empirically-derived expected win rates
- **Statistical Testing**: Automated p-value and effect size analysis
- **Conservative Adjustments**: Parameter modifications limited to ±15% to preserve scientific integrity
- **Bias Mitigation**: Systematic prevention of cherry-picking and confirmation bias

### Reproducibility Standards

- **Multi-seed Validation**: Statistical significance across multiple random seeds
- **Configuration Management**: Centralized, version-controlled parameter storage
- **Comprehensive Logging**: Full methodology and source attribution

## Training Modes

| Mode | Timesteps | Seeds | Duration | Use Case |
|------|-----------|-------|----------|----------|
| **Quick** | 5,000 | 1 | ~5 min | Development & debugging |
| **Standard** | 20,000 | 3 | ~20 min | Regular experiments |
| **Research** | 100,000 | 5 | ~2 hours | Publication-quality results |

## Architecture

```
threatsim/
├── envs/                     # RL Environment implementations
│   ├── threatsim_env.py     # Core multi-agent environment
│   └── wrappers.py          # Single-agent training wrappers
├── utils/                   # Utility modules
│   ├── logger.py           # Structured logging system
│   ├── scenario_manager.py # Scenario loading and validation
│   ├── balance_validator.py # Scientific balance validation
│   ├── visualization.py    # Statistical analysis and plotting
│   └── output_generator.py # Comprehensive output generation
├── scenarios/              # Cybersecurity scenario library
├── data/                   # Configuration and framework data
└── tests/                  # Comprehensive test suite
```

## Configuration System

ThreatSim uses a sophisticated configuration management system:

- **`data/config.json`**: Unified system configuration with training parameters
- **`data/mitre_attack.json`**: MITRE ATT&CK framework data with TTP mappings
- **Scenario YAML files**: Individual network topologies and threat parameters

## Output & Analysis

ThreatSim generates comprehensive outputs:

- **📊 Statistical Analysis**: Win rates, confidence intervals, effect sizes
- **📉 Visualizations**: Training curves, performance analysis, balance validation
- **📄 Reports**: HTML experiment reports with publication-quality figures
- **💾 Data Export**: CSV and JSON formats for external analysis

## Testing & Validation

```
# Run complete test suite
python run_tests.py

# Individual test modules
python tests/minimal_test.py      # Basic functionality
python tests/debug_training.py   # System validation
python tests/test_all_scenarios.py # Scenario compatibility
```

## Research Applications

### Academic Research
- **Reproducible Experiments**: Multi-seed validation with statistical significance testing
- **Parameter Transparency**: Full documentation of empirical parameter derivation
- **Publication Support**: Professional visualization and comprehensive analysis reports

### Industry Applications
- **Security Team Training**: Realistic Red vs Blue team training scenarios
- **Security Control Evaluation**: Quantitative assessment of defense effectiveness
- **Risk Assessment**: Probabilistic modeling of cybersecurity threats

## Usage Examples

### Single-Agent Training

```python
from threatsim.envs.wrappers import make_red_env
from stable_baselines3 import PPO

# Create Red team training environment
env = make_red_env(
    scenario_path="scenarios/enterprise_scenario.yaml",
    max_steps=50,
    blue_policy="heuristic"
)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Multi-Agent Environment

```python
from threatsim.envs.threatsim_env import ThreatSimEnv

# Create multi-agent environment
env = ThreatSimEnv(
    yaml_path="scenarios/ransomware_scenario.yaml",
    max_steps=100
)

# Execute simultaneous actions
actions = {
    "red": [node_idx, tactic_idx, technique_idx],
    "blue": [target_node, focus_tactic, response_type]
}

obs, rewards, terminated, truncated, info = env.step(actions)
```

## Contributing

We welcome contributions that advance the scientific rigor and practical applicability of cybersecurity simulation:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/new-capability`)
3. **Implement with comprehensive tests**
4. **Update documentation including scientific methodology**
5. **Submit pull request** with detailed description

### Contribution Guidelines

- **Scientific Rigor**: All parameters must be empirically derived with documented sources
- **Code Quality**: Comprehensive documentation, type hints, and error handling required
- **Testing**: All new features must include corresponding test coverage
- **Configuration**: Use centralized configuration system for all parameters

## License

Copyright (c) 2025 ThreatEcho

Licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) file for details.

## Citation

If you use ThreatSim in your research, please cite:

```bibtex
@software{threatsim2025,
  title={ThreatSim: Scientific Cybersecurity Simulation Framework},
  author={ThreatEcho},
  year={2025},
  url={https://github.com/threatecho/threatsim},
  version={0.3.2}
}
```

## Support

- **Documentation**: Comprehensive documentation included in repository
- **Issues**: Report bugs and feature requests through GitHub Issues
- **Research Collaboration**: Contact maintainers for academic collaboration opportunities

## Acknowledgments

ThreatSim development is supported by empirical research from:
- MITRE Corporation (ATT&CK framework)
- TheDFIRReport (The DFIR Report)
- Verizon (Data Breach Investigations Report)
- SANS Institute (SOC effectiveness surveys)
- NIST (Cybersecurity framework and standards)
- Academic and industry cybersecurity researchers

Built upon the excellent work of the stable-baselines3, gymnasium, and pettingzoo communities.
