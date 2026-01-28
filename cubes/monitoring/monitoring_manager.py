# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Cubes Monitoring Manager
Central coordinator for all monitoring components
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import asdict
import logging

from .config import MonitoringConfig, MonitoringConfigManager
from .metrics_collector import OLAPMetricsCollector, QueryMetric
from .query_analyzer import QueryAnalyzer, QueryPattern, QueryComplexity
from .performance_tracker import PerformanceTracker
from .system_monitor import SystemMonitor, HealthCheck


class CubesMonitoringManager:
    """Central manager for Cubes OLAP monitoring system"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None, 
                 config_file: Optional[str] = None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        if config:
            self.config = config
        else:
            config_manager = MonitoringConfigManager(config_file)
            self.config = config_manager.config
        
        # Initialize components
        self.metrics_collector = OLAPMetricsCollector(self.config, self.logger)
        self.query_analyzer = QueryAnalyzer(self.config, self.logger)
        self.performance_tracker = PerformanceTracker(self.config, self.logger)
        self.system_monitor = SystemMonitor(self.config, self.logger)
        
        # State management
        self._running = False
        self._startup_time = time.time()
        
        # Callbacks and integrations
        self._alert_callbacks = []
        self._workspace = None
        
        # Setup cross-component callbacks
        self._setup_component_callbacks()
    
    def start_monitoring(self):
        """Start all monitoring components"""
        if self._running:
            self.logger.warning("Monitoring is already running")
            return
        
        try:
            self.logger.info("Starting Cubes monitoring system")
            
            # Start individual components
            if self.config.metrics.enabled:
                self.metrics_collector.start_collection()
                self.logger.info("Metrics collection started")
            
            if self.config.performance.enabled:
                self.performance_tracker.start_tracking()
                self.logger.info("Performance tracking started")
            
            if self.config.system.enabled:
                self.system_monitor.start_monitoring()
                self.logger.info("System monitoring started")
            
            self._running = True
            self.logger.info("Cubes monitoring system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        if not self._running:
            return
        
        self.logger.info("Stopping Cubes monitoring system")
        
        try:
            # Stop individual components
            self.metrics_collector.stop_collection()
            self.performance_tracker.stop_tracking()
            self.system_monitor.stop_monitoring()
            
            self._running = False
            self.logger.info("Cubes monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def set_workspace(self, workspace):
        """Set the Cubes workspace for integration"""
        self._workspace = workspace
        self.logger.info("Workspace set for monitoring integration")
    
    def record_query_execution(self, cube_name: str, query_type: str, 
                             cuts: List[str], drills: List[str],
                             dimensions: List[str], measures: List[str],
                             execution_time: float, result_size: int,
                             sql_query: Optional[str] = None) -> Dict[str, Any]:
        """Record a query execution across all monitoring components"""
        
        query_id = f"{cube_name}_{query_type}_{int(time.time() * 1000)}"
        
        # Create query metric
        query_metric = QueryMetric(
            query_id=query_id,
            cube_name=cube_name,
            query_type=query_type,
            execution_time=execution_time,
            result_size=result_size,
            rows_returned=result_size,  # Approximation
            timestamp=time.time(),
            cuts=cuts,
            drills=drills,
            dimensions=dimensions,
            measures=measures,
            sql_query=sql_query
        )
        
        # Record in metrics collector
        self.metrics_collector.record_query_execution(query_metric)
        
        # Analyze query pattern
        query_pattern = self.query_analyzer.analyze_query(
            cube_name, query_type, cuts, drills, dimensions, measures, execution_time, result_size
        )
        
        # Analyze query complexity
        query_complexity = self.query_analyzer.analyze_complexity(
            cube_name, query_type, cuts, drills, dimensions, measures, execution_time, result_size
        )
        
        # Record performance metrics
        self.performance_tracker.record_query_performance(cube_name, query_type, execution_time, result_size)
        
        # Record specific aggregation metrics
        if query_type == 'aggregate':
            self.performance_tracker.record_aggregation_performance(
                cube_name, len(cuts), execution_time, result_size
            )
        
        # Return analysis results
        return {
            'query_id': query_id,
            'pattern': asdict(query_pattern),
            'complexity': asdict(query_complexity),
            'performance_score': self._calculate_query_performance_score(query_metric, query_complexity)
        }
    
    def record_cube_operation(self, cube_name: str, operation: str, 
                            duration: float, metadata: Dict[str, Any] = None):
        """Record a cube-specific operation"""
        self.metrics_collector.record_cube_metric(cube_name, f"{operation}_duration", duration, metadata)
        self.performance_tracker.record_metric(f"cube_{operation}", duration, 'seconds', metadata)
    
    def get_monitoring_dashboard(self, hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        dashboard = {
            'timestamp': time.time(),
            'period_hours': hours or 24,
            'system_status': asdict(self.system_monitor.get_system_status()),
            'performance_summary': self.performance_tracker.get_performance_summary(hours),
            'query_metrics': self._get_query_metrics_summary(hours),
            'cube_metrics': self._get_cube_metrics_summary(hours),
            'health_trends': self.system_monitor.get_status_trends(hours),
            'query_patterns': self.query_analyzer.get_query_patterns(hours=hours),
            'complexity_trends': self.query_analyzer.get_complexity_trends(hours=hours),
            'monitoring_uptime': time.time() - self._startup_time
        }
        
        return dashboard
    
    def get_cube_insights(self, cube_name: str, hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get comprehensive insights for a specific cube"""
        
        insights = {
            'cube_name': cube_name,
            'timestamp': time.time(),
            'period_hours': hours or 24,
            'query_statistics': self.metrics_collector.get_query_statistics(cube_name, hours),
            'cube_summary': self.metrics_collector.get_cube_summary(cube_name, hours),
            'performance_insights': self.query_analyzer.get_performance_insights(cube_name),
            'slow_queries': self.metrics_collector.get_slow_queries(hours=hours),
            'optimization_suggestions': self.query_analyzer.get_optimization_suggestions(cube_name)
        }
        
        return insights
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return {
            'status': asdict(self.system_monitor.get_system_status()),
            'health_history': self.system_monitor.get_health_history(hours=1),
            'component_status': dict(self.system_monitor.component_status),
            'active_issues': self.system_monitor.get_system_status().active_issues,
            'recommendations': self.system_monitor.get_system_status().recommendations
        }
    
    def get_performance_report(self, hours: Optional[int] = 24) -> Dict[str, Any]:
        """Get detailed performance report"""
        
        return {
            'timestamp': time.time(),
            'period_hours': hours or 24,
            'system_performance': self.performance_tracker.get_performance_summary(hours),
            'resource_usage': self.performance_tracker.get_resource_usage(hours),
            'database_metrics': self.performance_tracker.get_database_metrics(hours),
            'performance_score': self.performance_tracker._calculate_overall_performance_score(),
            'anomalies': self.performance_tracker._get_recent_anomalies(hours)
        }
    
    def export_monitoring_data(self, format: str = 'json', 
                              components: Optional[List[str]] = None) -> str:
        """Export monitoring data in specified format"""
        
        if not components:
            components = ['metrics', 'performance', 'system', 'query']
        
        data = {
            'timestamp': time.time(),
            'export_format': format,
            'components': components
        }
        
        if 'metrics' in components:
            data['metrics'] = json.loads(self.metrics_collector.export_metrics('json'))
        
        if 'performance' in components:
            data['performance'] = json.loads(self.performance_tracker.export_performance_data('json'))
        
        if 'system' in components:
            data['system'] = json.loads(self.system_monitor.export_health_data('json'))
        
        if 'query' in components:
            data['query_patterns'] = self.query_analyzer.get_query_patterns()
            data['complexity_trends'] = self.query_analyzer.get_complexity_trends()
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == 'prometheus':
            return self._export_prometheus_format(data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export data in Prometheus format"""
        lines = []
        
        # System metrics
        if 'system' in data.get('components', []):
            system_status = data['system']['current_status']
            lines.append(f"cubes_health_score {system_status['health_score']}")
            lines.append(f"cubes_uptime_percentage {system_status.get('uptime_percentage', 0)}")
        
        # Performance metrics
        if 'performance' in data.get('components', []):
            perf_summary = data['performance']['performance_summary']
            if 'system_performance' in perf_summary:
                sys_perf = perf_summary['system_performance']
                if 'cpu' in sys_perf:
                    lines.append(f"cubes_cpu_percent {sys_perf['cpu']['current']}")
                if 'memory' in sys_perf:
                    lines.append(f"cubes_memory_percent {sys_perf['memory']['current']}")
        
        # Query metrics
        if 'metrics' in data.get('components', []):
            query_stats = data['metrics']['query_statistics']
            for cube_name, stats in query_stats.items():
                lines.append(f'cubes_queries_total{{cube="{cube_name}"}} {stats["total_queries"]}')
                lines.append(f'cubes_query_duration_avg{{cube="{cube_name}"}} {stats["avg_execution_time"]}')
        
        return '\n'.join(lines)
    
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register an alert callback"""
        self._alert_callbacks.append(callback)
    
    def trigger_health_check(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Trigger manual health check"""
        if component:
            health_check = self.system_monitor.perform_health_check(component)
            return asdict(health_check)
        else:
            system_status = self.system_monitor.trigger_manual_check()
            return asdict(system_status)
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """Update monitoring configuration"""
        # This would update the configuration and restart components if needed
        self.logger.info(f"Configuration update requested: {config_updates}")
        # Implementation would depend on specific configuration changes
    
    def _setup_component_callbacks(self):
        """Setup cross-component callbacks"""
        
        # Metrics collector callbacks
        def metric_callback(metric_type: str, metric_data):
            if metric_type == 'query' and hasattr(metric_data, 'execution_time'):
                if metric_data.execution_time > self.config.query.slow_query_threshold:
                    self._trigger_alert('slow_query', {
                        'query_id': metric_data.query_id,
                        'cube_name': metric_data.cube_name,
                        'execution_time': metric_data.execution_time
                    })
        
        self.metrics_collector.register_callback(metric_callback)
        
        # Performance tracker callbacks
        def performance_callback(metric_type: str, performance_data):
            if metric_type == 'resource':
                # Check for resource alerts
                if hasattr(performance_data, 'cpu_percent'):
                    if performance_data.cpu_percent > self.config.system.cpu_warning_threshold:
                        self._trigger_alert('high_cpu', {
                            'cpu_percent': performance_data.cpu_percent,
                            'threshold': self.config.system.cpu_warning_threshold
                        })
        
        self.performance_tracker.register_callback(performance_callback)
        
        # System monitor callbacks
        def health_callback(health_check: HealthCheck):
            if health_check.status in ['critical', 'warning']:
                self._trigger_alert('health_issue', {
                    'component': health_check.name,
                    'status': health_check.status,
                    'message': health_check.message
                })
        
        self.system_monitor.register_callback(health_callback)
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger alert to all registered callbacks"""
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _calculate_query_performance_score(self, query_metric: QueryMetric, 
                                         query_complexity: QueryComplexity) -> float:
        """Calculate performance score for a query"""
        # Base score starts at 100
        score = 100.0
        
        # Deduct for execution time
        if query_metric.execution_time > 1.0:
            score -= min(query_metric.execution_time * 10, 50)
        
        # Deduct for complexity
        score -= query_complexity.score * 20
        
        # Deduct for large result sets
        if query_metric.result_size > 100000:
            score -= min(query_metric.result_size / 10000, 20)
        
        return max(0, min(100, score))
    
    def _get_query_metrics_summary(self, hours: Optional[int]) -> Dict[str, Any]:
        """Get query metrics summary"""
        query_stats = self.metrics_collector.get_query_statistics(hours=hours)
        
        summary = {
            'total_queries': sum(stats['total_queries'] for stats in query_stats.values()),
            'avg_execution_time': 0,
            'slow_queries': len(self.metrics_collector.get_slow_queries(hours=hours)),
            'query_types': {},
            'cube_distribution': {}
        }
        
        if query_stats:
            execution_times = [stats['avg_execution_time'] for stats in query_stats.values()]
            summary['avg_execution_time'] = sum(execution_times) / len(execution_times)
            
            # Aggregate query types
            for stats in query_stats.values():
                for query_type in stats['query_types']:
                    summary['query_types'][query_type] = summary['query_types'].get(query_type, 0) + stats['total_queries']
            
            # Cube distribution
            for cube_name, stats in query_stats.items():
                summary['cube_distribution'][cube_name] = stats['total_queries']
        
        return summary
    
    def _get_cube_metrics_summary(self, hours: Optional[int]) -> Dict[str, Any]:
        """Get cube metrics summary"""
        # This would aggregate cube-specific metrics
        return {
            'active_cubes': 0,  # Would be calculated from actual data
            'total_operations': 0,
            'avg_operation_time': 0
        }
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        return {
            'running': self._running,
            'uptime': time.time() - self._startup_time,
            'components': {
                'metrics': {
                    'enabled': self.config.metrics.enabled,
                    'running': self.metrics_collector._running
                },
                'query': {
                    'enabled': self.config.query.enabled
                },
                'performance': {
                    'enabled': self.config.performance.enabled,
                    'running': self.performance_tracker._running
                },
                'system': {
                    'enabled': self.config.system.enabled,
                    'running': self.system_monitor._running
                },
                'alerts': {
                    'enabled': self.config.alerts.enabled
                }
            },
            'configuration': {
                'log_level': self.config.log_level,
                'collection_interval': self.config.metrics.collection_interval,
                'tracking_interval': self.config.performance.tracking_interval,
                'check_interval': self.config.system.check_interval
            }
        }
    
    def cleanup_old_data(self, retention_hours: Optional[int] = None):
        """Clean up old monitoring data"""
        retention = retention_hours or self.config.metrics.retention_hours
        
        self.metrics_collector._cleanup_old_metrics(retention)
        self.performance_tracker.cleanup_old_data(retention)
        
        self.logger.info(f"Cleaned up monitoring data older than {retention} hours")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()
