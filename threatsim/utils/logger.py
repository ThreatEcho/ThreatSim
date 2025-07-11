# File: threatsim/utils/logger.py
# Description: Logging utility with symbolic formatting for ThreatSim
# Purpose: Logging system with symbols and colors
#
# Information:
# Symbols: [+] success, [-] error, [*] info, [!] warning, [*] debug
# Colors: RED for errors, GREEN for success, YELLOW for warnings, BLUE for info
#
# Copyright (c) 2025 ThreatEcho
# Licensed under the GNU General Public License v3.0
# See LICENSE file for details.

import sys
import logging
from enum import Enum
from typing import Optional, List, Any

class LogLevel(Enum):
    """Enumeration of log levels"""
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # Research-specific log levels
    EXPERIMENT_START = 60
    EXPERIMENT_RESULT = 61
    METRIC = 62
    CONFIGURATION = 63
    VALIDATION = 64

class ColorFormatter:
    """Provides color formatting for terminal output"""
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'ENDC': '\033[0m'
    }
    
    # Symbol mapping for log levels
    SYMBOLS = {
        'SUCCESS': '[+]',
        'ERROR': '[-]',
        'INFO': '[*]',
        'WARNING': '[!]',
        'DEBUG': '[*]',
        'HEADER': '[=]',
        'CONFIG': '[>]',
        'METRIC': '[#]'
    }

    @classmethod
    def format(cls, message: str, symbol: str = None, color: str = None) -> str:
        """Apply symbol and color formatting to log messages"""
        if not sys.stdout.isatty():
            # No color/formatting for non-terminal output
            return f"{symbol} {message}" if symbol else message
        
        color_code = cls.COLORS.get(color, '')
        reset_code = cls.COLORS['ENDC']
        
        if symbol and color_code:
            return f"{color_code}{symbol}{reset_code} {message}"
        elif symbol:
            return f"{symbol} {message}"
        elif color_code:
            return f"{color_code}{message}{reset_code}"
        else:
            return message

class ResearchLogger:
    """
    Professional logging system with symbolic formatting for scientific research
    """
    
    def __init__(self, name: str = 'ThreatSimLogger', 
                 level: LogLevel = LogLevel.INFO):
        self._logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with minimal configuration"""
        if not self._logger.handlers:
            # Simple console output - we'll handle formatting ourselves
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            
            # Minimal formatter - just the message
            formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(formatter)
            
            self._logger.addHandler(console_handler)
        
        self._logger.setLevel(logging.DEBUG)
        # Prevent propagation to avoid duplicate messages
        self._logger.propagate = False
    
    def debug(self, message: str):
        """Log debug messages with [?] symbol in blue"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['DEBUG'], 
                                        'BLUE')
        print(formatted)
    
    def info(self, message: str):
        """Log informational messages with [*] symbol in blue"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['INFO'], 
                                        'BLUE')
        print(formatted)
    
    def success(self, message: str):
        """Log success messages with [+] symbol in green"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['SUCCESS'], 
                                        'GREEN')
        print(formatted)
    
    def warning(self, message: str):
        """Log warning messages with [!] symbol in yellow"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['WARNING'], 
                                        'YELLOW')
        print(formatted)
    
    def error(self, message: str):
        """Log error messages with [-] symbol in red"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['ERROR'], 
                                        'RED')
        print(formatted)
    
    def critical(self, message: str):
        """Log critical error messages with [-] symbol in red"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['ERROR'], 
                                        'RED')
        print(formatted)
    
    def header(self, message: str):
        """Log header messages with [=] symbol"""
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['HEADER'], 
                                        'PURPLE')
        print(f"\n{formatted}")
        # Add separator line
        separator = "=" * (len(message) + 4)  # +4 for symbol and spaces
        print(ColorFormatter.format(separator, color='PURPLE'))
    
    def config(self, config_name: str, value: Any):
        """Log configuration details with [>] symbol"""
        str_value = str(value) if not isinstance(value, str) else value
        message = f"{config_name}: {str_value}"
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['CONFIG'], 
                                        'CYAN')
        print(f"  {formatted}")
    
    def metric(self, metric_name: str, value: Any):
        """Log metrics with [#] symbol"""
        message = f"{metric_name}: {value}"
        formatted = ColorFormatter.format(message, 
                                        ColorFormatter.SYMBOLS['METRIC'], 
                                        'GREEN')
        print(f"  {formatted}")
    
    def table_header(self, headers: List[str], widths: List[int]):
        """Print formatted table header"""
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        separator = "-+-".join("-" * w for w in widths)
        
        if sys.stdout.isatty():
            header_row = ColorFormatter.format(header_row, color='BOLD')
        
        print(f"  {header_row}")
        print(f"  {separator}")
    
    def table_row(self, values: List[Any], widths: List[int]):
        """Print formatted table row"""
        str_values = [str(v) for v in values]
        row = " | ".join(v.ljust(w) for v, w in zip(str_values, widths))
        print(f"  {row}")
    
    def validation(self, validation_message: str, status: bool = True):
        """Log validation results"""
        status_text = 'PASSED' if status else 'FAILED'
        message = f"{validation_message}: {status_text}"
        
        if status:
            self.success(message)
        else:
            self.error(message)
    
    def experiment_start(self, experiment_name: str, parameters: Optional[dict] = None):
        """Log experiment start"""
        self.header(f"EXPERIMENT START: {experiment_name}")
        if parameters:
            for key, value in parameters.items():
                self.config(key, value)
    
    def experiment_result(self, metric_name: str, value: Any, 
                          unit: str = '', expected: Optional[float] = None):
        """Log experimental results"""
        if expected is not None:
            deviation = abs(float(value) - expected)
            within_tolerance = deviation < (expected * 0.1)
            status = "VALIDATED" if within_tolerance else "DEVIATION"
            message = f"{metric_name}: {value}{unit} (expected: {expected}{unit}, {status})"
            
            if within_tolerance:
                self.success(message)
            else:
                self.warning(message)
        else:
            message = f"{metric_name}: {value}{unit}"
            self.success(message)

# Global logger instance
logger = ResearchLogger()

# Convenience functions
def log_experiment_start(name: str, parameters: Optional[dict] = None):
    """Shorthand for logging experiment start"""
    logger.experiment_start(name, parameters)

def log_experiment_result(metric_name: str, value: Any, 
                          unit: str = '', expected: Optional[float] = None):
    """Shorthand for logging experiment results"""
    logger.experiment_result(metric_name, value, unit, expected)

def log_metric(metric_name: str, value: Any):
    """Shorthand for logging metrics"""
    logger.metric(metric_name, value)

# Demonstration if run directly
if __name__ == "__main__":
    logger.header("ThreatSim Logging System Demo")
    
    logger.info("System initialization starting")
    logger.debug("Loading configuration files")
    logger.config("Version", "0.3.1")
    logger.config("Random Seed", 42)
    
    logger.success("Environment created successfully")
    logger.metric("Win Rate", "35.2%")
    logger.warning("Parameter outside expected range")
    logger.error("Training failed due to invalid configuration")
    
    logger.validation("Model Performance Check", True)
    logger.validation("Parameter Validation", False)
    
    logger.experiment_result("Accuracy", 0.85, "%", 0.80)
