# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Monitoring integration for Cubes framework
"""

import time
import logging
from typing import Dict, List, Any, Optional

from .monitoring import CubesMonitoringManager, MonitoringConfigManager


class CubesMonitoringIntegration:
    """Integration layer for Cubes monitoring system"""
    
    def __init__(self, workspace=None, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.workspace = workspace
        
        # Initialize monitoring
        try:
            config_manager = MonitoringConfigManager(config_file)
            if config_manager.config.enabled:
                self.monitoring_manager = CubesMonitoringManager(
                    config_manager.config, 
                    config_file, 
                    self.logger
                )
                self.monitoring_manager.set_workspace(workspace)
                self.monitoring_manager.start_monitoring()
                self.logger.info("Cubes monitoring system initialized and started")
            else:
                self.monitoring_manager = None
                self.logger.info("Cubes monitoring system disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            self.monitoring_manager = None
    
    def record_query_execution(self, cube_name: str, query_type: str, 
                             cuts: List[str], drills: List[str],
                             dimensions: List[str], measures: List[str],
                             execution_time: float, result_size: int,
                             sql_query: Optional[str] = None) -> Dict[str, Any]:
        """Record query execution with monitoring"""
        if not self.monitoring_manager:
            return {}
        
        try:
            return self.monitoring_manager.record_query_execution(
                cube_name, query_type, cuts, drills, dimensions, measures,
                execution_time, result_size, sql_query
            )
        except Exception as e:
            self.logger.error(f"Failed to record query execution: {e}")
            return {}
    
    def record_aggregation(self, cube_name: str, cuts: List[str], 
                         execution_time: float, cell_count: int):
        """Record aggregation operation"""
        if not self.monitoring_manager:
            return
        
        try:
            self.monitoring_manager.metrics_collector.record_aggregate_metric(
                cube_name, cuts, execution_time, cell_count
            )
        except Exception as e:
            self.logger.error(f"Failed to record aggregation: {e}")
    
    def record_drilldown(self, cube_name: str, dimension: str, 
                        execution_time: float, result_count: int):
        """Record drilldown operation"""
        if not self.monitoring_manager:
            return
        
        try:
            self.monitoring_manager.metrics_collector.record_drilldown_metric(
                cube_name, dimension, execution_time, result_count
            )
        except Exception as e:
            self.logger.error(f"Failed to record drilldown: {e}")
    
    def record_facts_retrieval(self, cube_name: str, cuts: List[str], 
                             execution_time: float, result_count: int):
        """Record facts retrieval operation"""
        if not self.monitoring_manager:
            return
        
        try:
            self.monitoring_manager.metrics_collector.record_facts_metric(
                cube_name, cuts, execution_time, result_count
            )
        except Exception as e:
            self.logger.error(f"Failed to record facts retrieval: {e}")
    
    def get_monitoring_dashboard(self, hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        if not self.monitoring_manager:
            return {}
        
        try:
            return self.monitoring_manager.get_monitoring_dashboard(hours)
        except Exception as e:
            self.logger.error(f"Failed to get monitoring dashboard: {e}")
            return {}
    
    def get_cube_insights(self, cube_name: str, hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get insights for a specific cube"""
        if not self.monitoring_manager:
            return {}
        
        try:
            return self.monitoring_manager.get_cube_insights(cube_name, hours)
        except Exception as e:
            self.logger.error(f"Failed to get cube insights: {e}")
            return {}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        if not self.monitoring_manager:
            return {}
        
        try:
            return self.monitoring_manager.get_system_health()
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {}
    
    def stop(self):
        """Stop monitoring system"""
        if self.monitoring_manager:
            try:
                self.monitoring_manager.stop_monitoring()
                self.logger.info("Cubes monitoring system stopped")
            except Exception as e:
                self.logger.error(f"Error stopping monitoring system: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Global monitoring instance
_monitoring_integration = None


def initialize_monitoring(workspace=None, config_file: Optional[str] = None) -> CubesMonitoringIntegration:
    """Initialize global monitoring integration"""
    global _monitoring_integration
    _monitoring_integration = CubesMonitoringIntegration(workspace, config_file)
    return _monitoring_integration


def get_monitoring() -> Optional[CubesMonitoringIntegration]:
    """Get global monitoring integration instance"""
    return _monitoring_integration


def record_query(cube_name: str, query_type: str, cuts: List[str], drills: List[str],
                dimensions: List[str], measures: List[str], execution_time: float,
                result_size: int, sql_query: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to record query execution"""
    if _monitoring_integration:
        return _monitoring_integration.record_query_execution(
            cube_name, query_type, cuts, drills, dimensions, measures,
            execution_time, result_size, sql_query
        )
    return {}


def record_aggregation(cube_name: str, cuts: List[str], execution_time: float, cell_count: int):
    """Convenience function to record aggregation"""
    if _monitoring_integration:
        _monitoring_integration.record_aggregation(cube_name, cuts, execution_time, cell_count)


def record_drilldown(cube_name: str, dimension: str, execution_time: float, result_count: int):
    """Convenience function to record drilldown"""
    if _monitoring_integration:
        _monitoring_integration.record_drilldown(cube_name, dimension, execution_time, result_count)


def record_facts(cube_name: str, cuts: List[str], execution_time: float, result_count: int):
    """Convenience function to record facts retrieval"""
    if _monitoring_integration:
        _monitoring_integration.record_facts_retrieval(cube_name, cuts, execution_time, result_count)
