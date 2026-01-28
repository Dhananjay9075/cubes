# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Configuration management for Cubes monitoring system
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    collection_interval: int = 30  # seconds
    retention_hours: int = 24
    max_history_size: int = 1000
    export_formats: list = field(default_factory=lambda: ['json', 'prometheus'])
    export_interval: int = 300  # seconds
    export_directory: str = "monitoring"
    
    def __post_init__(self):
        """Validate metrics configuration"""
        if self.collection_interval <= 0:
            raise ValueError("collection_interval must be positive")
        if self.retention_hours <= 0:
            raise ValueError("retention_hours must be positive")
        if self.max_history_size <= 0:
            raise ValueError("max_history_size must be positive")
        if self.export_interval <= 0:
            raise ValueError("export_interval must be positive")


@dataclass
class QueryConfig:
    """Configuration for query monitoring"""
    enabled: bool = True
    track_slow_queries: bool = True
    slow_query_threshold: float = 1.0  # seconds
    track_query_patterns: bool = True
    max_query_history: int = 1000
    analyze_query_complexity: bool = True
    
    def __post_init__(self):
        """Validate query configuration"""
        if self.slow_query_threshold <= 0:
            raise ValueError("slow_query_threshold must be positive")
        if self.max_query_history <= 0:
            raise ValueError("max_query_history must be positive")


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    enabled: bool = True
    tracking_interval: int = 60
    track_memory_usage: bool = True
    track_cpu_usage: bool = True
    track_database_connections: bool = True
    anomaly_detection: bool = True
    anomaly_threshold_multiplier: float = 2.0


@dataclass
class SystemConfig:
    """Configuration for system monitoring"""
    enabled: bool = True
    check_interval: int = 60
    max_history_size: int = 100
    
    # Resource thresholds
    cpu_warning_threshold: float = 70.0
    cpu_critical_threshold: float = 90.0
    memory_warning_threshold: float = 75.0
    memory_critical_threshold: float = 90.0
    disk_warning_threshold: float = 80.0
    disk_critical_threshold: float = 95.0
    
    # Database-specific thresholds
    db_connection_warning: int = 50
    db_connection_critical: int = 100
    db_query_time_warning: float = 2.0
    db_query_time_critical: float = 5.0
    
    def __post_init__(self):
        """Validate system configuration"""
        if self.check_interval <= 0:
            raise ValueError("check_interval must be positive")
        if self.max_history_size <= 0:
            raise ValueError("max_history_size must be positive")
        if self.cpu_warning_threshold >= self.cpu_critical_threshold:
            raise ValueError("cpu_warning_threshold must be less than cpu_critical_threshold")
        if self.memory_warning_threshold >= self.memory_critical_threshold:
            raise ValueError("memory_warning_threshold must be less than memory_critical_threshold")
        if self.disk_warning_threshold >= self.disk_critical_threshold:
            raise ValueError("disk_warning_threshold must be less than disk_critical_threshold")


@dataclass
class AlertsConfig:
    """Configuration for alerts"""
    enabled: bool = True
    email_notifications: bool = False
    webhook_url: Optional[str] = None
    alert_cooldown: int = 300  # seconds
    
    # Alert thresholds
    slow_query_alert: bool = True
    performance_alert: bool = True
    system_alert: bool = True
    error_rate_alert: bool = True


@dataclass
class MonitoringConfig:
    """Main monitoring configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    
    # Component configurations
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.metrics.collection_interval <= 0:
            raise ValueError("metrics.collection_interval must be positive")
        
        if self.query.slow_query_threshold <= 0:
            raise ValueError("query.slow_query_threshold must be positive")
        
        if self.system.cpu_warning_threshold >= self.system.cpu_critical_threshold:
            raise ValueError("cpu_warning_threshold must be less than cpu_critical_threshold")
        
        if self.system.memory_warning_threshold >= self.system.memory_critical_threshold:
            raise ValueError("memory_warning_threshold must be less than memory_critical_threshold")


class MonitoringConfigManager:
    """Manager for monitoring configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config = self._load_configuration()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations"""
        possible_paths = [
            "monitoring.yaml",
            "monitoring.yml", 
            "monitoring.json",
            "config/monitoring.yaml",
            "config/monitoring.yml",
            "config/monitoring.json",
            os.path.expanduser("~/.cubes/monitoring.yaml"),
            "/etc/cubes/monitoring.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_configuration(self) -> MonitoringConfig:
        """Load configuration from file or environment"""
        config_data = {}
        
        # Load from file if exists
        if self.config_file:
            config_data = self._load_config_file(self.config_file)
        
        # Override with environment variables
        config_data = self._apply_environment_overrides(config_data)
        
        # Create configuration object
        return self._create_config_from_dict(config_data)
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f) or {}
            elif file_path.endswith('.json'):
                return json.load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {file_path}")
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_mappings = {
            'CUBES_MONITORING_ENABLED': ('enabled', bool),
            'CUBES_MONITORING_LOG_LEVEL': ('log_level', str),
            'CUBES_MONITORING_METRICS_ENABLED': ('metrics.enabled', bool),
            'CUBES_MONITORING_METRICS_INTERVAL': ('metrics.collection_interval', int),
            'CUBES_MONITORING_QUERY_ENABLED': ('query.enabled', bool),
            'CUBES_MONITORING_QUERY_SLOW_THRESHOLD': ('query.slow_query_threshold', float),
            'CUBES_MONITORING_PERFORMANCE_ENABLED': ('performance.enabled', bool),
            'CUBES_MONITORING_SYSTEM_ENABLED': ('system.enabled', bool),
            'CUBES_MONITORING_SYSTEM_CPU_WARNING': ('system.cpu_warning_threshold', float),
            'CUBES_MONITORING_SYSTEM_CPU_CRITICAL': ('system.cpu_critical_threshold', float),
            'CUBES_MONITORING_SYSTEM_MEMORY_WARNING': ('system.memory_warning_threshold', float),
            'CUBES_MONITORING_SYSTEM_MEMORY_CRITICAL': ('system.memory_critical_threshold', float),
            'CUBES_MONITORING_ALERTS_ENABLED': ('alerts.enabled', bool),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert value to appropriate type
                if value_type == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif value_type == int:
                    value = int(value)
                elif value_type == float:
                    value = float(value)
                
                # Set nested configuration value
                self._set_nested_value(config_data, config_path, value)
        
        return config_data
    
    def _set_nested_value(self, config_data: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        current = config_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> MonitoringConfig:
        """Create MonitoringConfig from dictionary"""
        # Extract component configurations
        metrics_data = config_data.get('metrics', {})
        query_data = config_data.get('query', {})
        performance_data = config_data.get('performance', {})
        system_data = config_data.get('system', {})
        alerts_data = config_data.get('alerts', {})
        
        return MonitoringConfig(
            enabled=config_data.get('enabled', True),
            log_level=config_data.get('log_level', 'INFO'),
            metrics=MetricsConfig(**metrics_data),
            query=QueryConfig(**query_data),
            performance=PerformanceConfig(**performance_data),
            system=SystemConfig(**system_data),
            alerts=AlertsConfig(**alerts_data)
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            'enabled': self.config.enabled,
            'log_level': self.config.log_level,
            'metrics': {
                'enabled': self.config.metrics.enabled,
                'collection_interval': self.config.metrics.collection_interval,
                'retention_hours': self.config.metrics.retention_hours,
                'max_history_size': self.config.metrics.max_history_size,
                'export_formats': self.config.metrics.export_formats,
                'export_interval': self.config.metrics.export_interval,
                'export_directory': self.config.metrics.export_directory,
            },
            'query': {
                'enabled': self.config.query.enabled,
                'track_slow_queries': self.config.query.track_slow_queries,
                'slow_query_threshold': self.config.query.slow_query_threshold,
                'track_query_patterns': self.config.query.track_query_patterns,
                'max_query_history': self.config.query.max_query_history,
                'analyze_query_complexity': self.config.query.analyze_query_complexity,
            },
            'performance': {
                'enabled': self.config.performance.enabled,
                'tracking_interval': self.config.performance.tracking_interval,
                'track_memory_usage': self.config.performance.track_memory_usage,
                'track_cpu_usage': self.config.performance.track_cpu_usage,
                'track_database_connections': self.config.performance.track_database_connections,
                'anomaly_detection': self.config.performance.anomaly_detection,
                'anomaly_threshold_multiplier': self.config.performance.anomaly_threshold_multiplier,
            },
            'system': {
                'enabled': self.config.system.enabled,
                'check_interval': self.config.system.check_interval,
                'max_history_size': self.config.system.max_history_size,
                'cpu_warning_threshold': self.config.system.cpu_warning_threshold,
                'cpu_critical_threshold': self.config.system.cpu_critical_threshold,
                'memory_warning_threshold': self.config.system.memory_warning_threshold,
                'memory_critical_threshold': self.config.system.memory_critical_threshold,
                'disk_warning_threshold': self.config.system.disk_warning_threshold,
                'disk_critical_threshold': self.config.system.disk_critical_threshold,
                'db_connection_warning': self.config.system.db_connection_warning,
                'db_connection_critical': self.config.system.db_connection_critical,
                'db_query_time_warning': self.config.system.db_query_time_warning,
                'db_query_time_critical': self.config.system.db_query_time_critical,
            },
            'alerts': {
                'enabled': self.config.alerts.enabled,
                'email_notifications': self.config.alerts.email_notifications,
                'webhook_url': self.config.alerts.webhook_url,
                'alert_cooldown': self.config.alerts.alert_cooldown,
                'slow_query_alert': self.config.alerts.slow_query_alert,
                'performance_alert': self.config.alerts.performance_alert,
                'system_alert': self.config.alerts.system_alert,
                'error_rate_alert': self.config.alerts.error_rate_alert,
            }
        }
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        current_config = self.get_config_dict()
        self._deep_update(current_config, updates)
        self.config = self._create_config_from_dict(current_config)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = file_path or self.config_file or 'monitoring.yaml'
        
        config_dict = self.get_config_dict()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif save_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {save_path}")
    
    def is_component_enabled(self, component: str) -> bool:
        """Check if a monitoring component is enabled"""
        component_map = {
            'metrics': self.config.metrics.enabled,
            'query': self.config.query.enabled,
            'performance': self.config.performance.enabled,
            'system': self.config.system.enabled,
            'alerts': self.config.alerts.enabled,
        }
        
        return component_map.get(component, False) and self.config.enabled
