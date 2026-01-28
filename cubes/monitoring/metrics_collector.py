# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
OLAP metrics collector for Cubes framework
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryMetric:
    """OLAP query metric"""
    query_id: str
    cube_name: str
    query_type: str  # aggregate, drilldown, facts, etc.
    execution_time: float
    result_size: int
    rows_returned: int
    timestamp: float
    cuts: List[str]
    drills: List[str]
    dimensions: List[str]
    measures: List[str]
    sql_query: Optional[str] = None
    complexity_score: Optional[float] = None


@dataclass
class CubeMetric:
    """Cube-specific metric"""
    cube_name: str
    metric_type: str
    value: float
    timestamp: float
    metadata: Dict[str, Any]
    labels: Dict[str, Any] = field(default_factory=dict)


class OLAPMetricsCollector:
    """Collects and manages OLAP-specific metrics"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Storage for metrics
        self.query_metrics = deque(maxlen=config.query.max_query_history)
        self.cube_metrics = defaultdict(lambda: deque(maxlen=config.metrics.max_history_size))
        self.system_metrics = defaultdict(lambda: deque(maxlen=config.metrics.max_history_size))
        
        # Aggregated metrics
        self.query_stats = defaultdict(list)
        self.cube_stats = defaultdict(dict)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background collection
        self._running = False
        self._collection_thread = None
        
        # Callbacks for custom metrics
        self._metric_callbacks = []
    
    def start_collection(self):
        """Start background metrics collection"""
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        self.logger.info("OLAP metrics collection started")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        self.logger.info("OLAP metrics collection stopped")
    
    def _collection_loop(self):
        """Background collection loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                self._cleanup_old_metrics()
                time.sleep(self.config.metrics.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
    
    def record_query_execution(self, query_metric: QueryMetric):
        """Record a query execution metric"""
        with self._lock:
            self.query_metrics.append(query_metric)
            
            # Update query statistics
            self.query_stats[query_metric.cube_name].append({
                'execution_time': query_metric.execution_time,
                'result_size': query_metric.result_size,
                'rows_returned': query_metric.rows_returned,
                'timestamp': query_metric.timestamp,
                'query_type': query_metric.query_type
            })
            
            # Trigger callbacks
            self._trigger_callbacks('query', query_metric)
        
        # Check for slow query alert
        if (self.config.query.track_slow_queries and 
            query_metric.execution_time > self.config.query.slow_query_threshold):
            self._handle_slow_query(query_metric)
    
    def record_cube_metric(self, cube_name: str, metric_type: str, value: float, metadata: Dict[str, Any] = None):
        """Record a cube-specific metric"""
        metric = CubeMetric(
            cube_name=cube_name,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
            labels=metadata or {}  # Use metadata as labels for compatibility
        )
        
        with self._lock:
            self.cube_metrics[cube_name].append(metric)
            
            # Update cube statistics
            if metric_type not in self.cube_stats[cube_name]:
                self.cube_stats[cube_name][metric_type] = []
            
            self.cube_stats[cube_name][metric_type].append({
                'value': value,
                'timestamp': metric.timestamp
            })
            
            # Trigger callbacks
            self._trigger_callbacks('cube', metric)
    
    def record_aggregate_metric(self, cube_name: str, cuts: List[str], 
                             execution_time: float, result_count: int):
        """Record aggregation-specific metric"""
        self.record_cube_metric(cube_name, 'aggregate_execution_time', execution_time, {
            'cuts_count': len(cuts),
            'cuts': cuts
        })
        self.record_cube_metric(cube_name, 'aggregate_result_count', result_count, {
            'cuts_count': len(cuts)
        })
    
    def record_drilldown_metric(self, cube_name: str, dimension: str, 
                              execution_time: float, result_count: int):
        """Record drilldown-specific metric"""
        self.record_cube_metric(cube_name, 'drilldown_execution_time', execution_time, {
            'dimension': dimension
        })
        self.record_cube_metric(cube_name, 'drilldown_result_count', result_count, {
            'dimension': dimension
        })
    
    def record_facts_metric(self, cube_name: str, cuts: List[str], 
                          execution_time: float, result_count: int):
        """Record facts retrieval metric"""
        self.record_cube_metric(cube_name, 'facts_execution_time', execution_time, {
            'cuts_count': len(cuts)
        })
        self.record_cube_metric(cube_name, 'facts_result_count', result_count, {
            'cuts_count': len(cuts)
        })
    
    def get_query_metrics(self, cube_name: Optional[str] = None, 
                         hours: Optional[int] = None) -> List[QueryMetric]:
        """Get query metrics with optional filtering"""
        with self._lock:
            metrics = list(self.query_metrics)
        
        # Filter by cube name
        if cube_name:
            metrics = [m for m in metrics if m.cube_name == cube_name]
        
        # Filter by time
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_cube_metrics(self, cube_name: str, metric_type: Optional[str] = None,
                        hours: Optional[int] = None) -> List[CubeMetric]:
        """Get cube metrics with optional filtering"""
        with self._lock:
            metrics = list(self.cube_metrics.get(cube_name, []))
        
        # Filter by metric type
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        
        # Filter by time
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_query_statistics(self, cube_name: Optional[str] = None, 
                           hours: Optional[int] = None) -> Dict[str, Any]:
        """Get query statistics"""
        with self._lock:
            stats = dict(self.query_stats)
        
        # Filter by cube name
        if cube_name:
            stats = {cube_name: stats.get(cube_name, [])}
        
        # Filter by time
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            for cn in stats:
                stats[cn] = [s for s in stats[cn] if s['timestamp'] >= cutoff_time]
        
        # Calculate statistics
        result = {}
        for cn, cn_stats in stats.items():
            if not cn_stats:
                continue
            
            execution_times = [s['execution_time'] for s in cn_stats]
            result_sizes = [s['result_size'] for s in cn_stats]
            rows_returned = [s['rows_returned'] for s in cn_stats]
            
            result[cn] = {
                'total_queries': len(cn_stats),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'avg_result_size': sum(result_sizes) / len(result_sizes),
                'total_rows_returned': sum(rows_returned),
                'query_types': list(set(s['query_type'] for s in cn_stats))
            }
        
        return result
    
    def get_slow_queries(self, threshold: Optional[float] = None, 
                         hours: Optional[int] = None) -> List[QueryMetric]:
        """Get slow queries"""
        threshold = threshold or self.config.query.slow_query_threshold
        
        metrics = self.get_query_metrics(hours=hours)
        return [m for m in metrics if m.execution_time > threshold]
    
    def get_cube_summary(self, cube_name: str, hours: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive cube metrics summary"""
        query_stats = self.get_query_statistics(cube_name, hours=hours)
        cube_metrics = self.get_cube_metrics(cube_name, hours=hours)
        
        # Group cube metrics by type
        metrics_by_type = defaultdict(list)
        for metric in cube_metrics:
            metrics_by_type[metric.metric_type].append(metric.value)
        
        # Calculate summary for each metric type
        summary = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                summary[metric_type] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        # Add query statistics
        if cube_name in query_stats:
            summary['queries'] = query_stats[cube_name]
        
        return summary
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_system_metric('cpu_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._record_system_metric('memory_percent', memory.percent)
            self._record_system_metric('memory_used_mb', memory.used / (1024 * 1024))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self._record_system_metric('disk_percent', disk.percent)
            self._record_system_metric('disk_free_gb', disk.free / (1024 * 1024 * 1024))
            
        except ImportError:
            self.logger.warning("psutil not available for system metrics")
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _record_system_metric(self, metric_name: str, value: float):
        """Record a system metric"""
        metric = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels={'source': 'system'}
        )
        
        with self._lock:
            self.system_metrics[metric_name].append(metric)
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        cutoff_time = time.time() - (self.config.metrics.retention_hours * 3600)
        
        with self._lock:
            # Clean up system metrics
            for metric_name in self.system_metrics:
                self.system_metrics[metric_name] = deque(
                    [m for m in self.system_metrics[metric_name] if m.timestamp >= cutoff_time],
                    maxlen=self.config.metrics.max_history_size
                )
            
            # Clean up cube metrics
            for cube_name in self.cube_metrics:
                self.cube_metrics[cube_name] = deque(
                    [m for m in self.cube_metrics[cube_name] if m.timestamp >= cutoff_time],
                    maxlen=self.config.metrics.max_history_size
                )
    
    def _handle_slow_query(self, query_metric: QueryMetric):
        """Handle slow query alert"""
        self.logger.warning(
            f"Slow query detected: {query_metric.query_id} on cube {query_metric.cube_name} "
            f"took {query_metric.execution_time:.3f}s"
        )
        
        # Trigger alert callback if configured
        if self.config.alerts.enabled and self.config.alerts.slow_query_alert:
            self._trigger_alert('slow_query', {
                'query_id': query_metric.query_id,
                'cube_name': query_metric.cube_name,
                'execution_time': query_metric.execution_time,
                'threshold': self.config.query.slow_query_threshold
            })
    
    def _trigger_callbacks(self, metric_type: str, metric_data):
        """Trigger registered metric callbacks"""
        for callback in self._metric_callbacks:
            try:
                callback(metric_type, metric_data)
            except Exception as e:
                self.logger.error(f"Error in metric callback: {e}")
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger alert notification"""
        # This would integrate with the alerts system
        self.logger.warning(f"Alert triggered: {alert_type} - {alert_data}")
    
    def register_callback(self, callback: Callable[[str, Any], None]):
        """Register a callback for metric events"""
        self._metric_callbacks.append(callback)
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        with self._lock:
            data = {
                'timestamp': time.time(),
                'query_metrics': [asdict(m) for m in self.query_metrics],
                'cube_metrics': {
                    cube: [asdict(m) for m in metrics]
                    for cube, metrics in self.cube_metrics.items()
                },
                'system_metrics': {
                    name: [asdict(m) for m in metrics]
                    for name, metrics in self.system_metrics.items()
                },
                'query_statistics': self.get_query_statistics(),
                'cube_summaries': {
                    cube: self.get_cube_summary(cube)
                    for cube in self.cube_metrics.keys()
                }
            }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == 'prometheus':
            return self._export_prometheus_format(data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # System metrics
        for metric_name, metrics in data['system_metrics'].items():
            if metrics:
                latest = metrics[-1]
                lines.append(f"cubes_{metric_name} {latest['value']}")
        
        # Query metrics
        query_stats = data['query_statistics']
        for cube_name, stats in query_stats.items():
            lines.append(f'cubes_queries_total{{cube="{cube_name}"}} {stats["total_queries"]}')
            lines.append(f'cubes_query_duration_avg{{cube="{cube_name}"}} {stats["avg_execution_time"]}')
            lines.append(f'cubes_query_duration_max{{cube="{cube_name}"}} {stats["max_execution_time"]}')
        
        return '\n'.join(lines)
