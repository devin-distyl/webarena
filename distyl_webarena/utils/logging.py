"""
DistylLogger: Enhanced logging for Distyl-WebArena

Provides structured logging with component-specific formatting and levels.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


class DistylLogger:
    """Enhanced logger for Distyl-WebArena components"""
    
    def __init__(self, component_name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        self.component_name = component_name
        self.logger = logging.getLogger(f"distyl_webarena.{component_name}")
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_logger(log_level, log_file)
    
    def _setup_logger(self, log_level: str, log_file: Optional[str]):
        """Setup logger with appropriate handlers and formatting"""
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context"""
        
        if not kwargs:
            return message
        
        # Add context information
        context_parts = []
        for key, value in kwargs.items():
            context_parts.append(f"{key}={value}")
        
        context_str = ", ".join(context_parts)
        return f"{message} [{context_str}]"
    
    def log_action(self, action: str, element_id: str = "", success: bool = True):
        """Log action execution"""
        status = "SUCCESS" if success else "FAILED"
        if element_id:
            self.info(f"Action {status}: {action}", element_id=element_id)
        else:
            self.info(f"Action {status}: {action}")
    
    def log_grounding(self, description: str, element_id: str, success: bool = True):
        """Log element grounding results"""
        status = "SUCCESS" if success else "FAILED"
        self.debug(f"Grounding {status}: '{description}' -> {element_id}")
    
    def log_planning(self, task: str, subtask_count: int):
        """Log planning results"""
        self.info(f"Planning completed: {subtask_count} subtasks for '{task}'")
    
    def log_trajectory(self, step: int, action: str, observation_summary: str = ""):
        """Log trajectory step"""
        if observation_summary:
            self.debug(f"Step {step}: {action} | {observation_summary}")
        else:
            self.debug(f"Step {step}: {action}")
    
    def log_performance(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics"""
        unit_str = f" {unit}" if unit else ""
        self.info(f"Performance: {metric_name} = {value:.3f}{unit_str}")
    
    def log_error_with_context(self, error: Exception, context: dict):
        """Log error with detailed context"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.error(f"Error: {str(error)} | Context: {context_str}")


class DistylLogManager:
    """Global log manager for Distyl-WebArena"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.log_level = "INFO"
        self.log_directory = "logs"
        self.enable_file_logging = True
    
    def configure(self, log_level: str = "INFO", log_directory: str = "logs", enable_file_logging: bool = True):
        """Configure global logging settings"""
        self.log_level = log_level
        self.log_directory = log_directory
        self.enable_file_logging = enable_file_logging
        
        # Ensure log directory exists
        if enable_file_logging and not os.path.exists(log_directory):
            os.makedirs(log_directory)
    
    def get_logger(self, component_name: str) -> DistylLogger:
        """Get or create logger for component"""
        
        if component_name not in self._loggers:
            log_file = None
            if self.enable_file_logging:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(self.log_directory, f"distyl_webarena_{timestamp}.log")
            
            self._loggers[component_name] = DistylLogger(
                component_name, 
                self.log_level, 
                log_file
            )
        
        return self._loggers[component_name]
    
    def set_log_level(self, level: str):
        """Set log level for all existing loggers"""
        self.log_level = level
        
        for logger in self._loggers.values():
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.logger.setLevel(log_level)
            
            for handler in logger.logger.handlers:
                handler.setLevel(log_level)
    
    def log_system_info(self):
        """Log system information at startup"""
        logger = self.get_logger("System")
        
        logger.info("Distyl-WebArena System Starting")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Log level: {self.log_level}")
        logger.info(f"File logging: {self.enable_file_logging}")
        
        if self.enable_file_logging:
            logger.info(f"Log directory: {self.log_directory}")


# Global instance
log_manager = DistylLogManager()


def get_logger(component_name: str) -> DistylLogger:
    """Convenience function to get a logger"""
    return log_manager.get_logger(component_name)


def configure_logging(log_level: str = "INFO", log_directory: str = "logs", enable_file_logging: bool = True):
    """Convenience function to configure logging"""
    log_manager.configure(log_level, log_directory, enable_file_logging)