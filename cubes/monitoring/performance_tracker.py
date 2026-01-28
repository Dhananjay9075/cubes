# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Performance tracker for Cubes OLAP framework
Tracks system performance and resource usage
"""

import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any]


@dataclass
class ResourceUsage:
    """Resource usage snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    open_files: int
    threads: int
    process_count: int


@dataclass
class DatabaseMetric:
    """Database performance metric"""
    timestamp: float
    connection_count: int
    active_connections: int
    query_count: int
    avg_query_time: float
    slow_query_count: int
    cache_hit_ratio: float


class PerformanceTracker:
    """Tracks system and database performance"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance data storage
        self.performance_metrics = defaultdict(lambda: deque(maxlen=config.performance.max_history_size))
        self.resource_usage = deque(maxlen=config.performance.max_history_size)
        self.database_metrics = deque(maxlen=config.performance.max_history_size)
        
        # Performance baselines
        self.performance_baselines = {}
        self.anomaly_thresholds = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tracking
        self._running = False
        self._tracking_thread = None
        
        # Performance callbacks
        self._performance_callbacks = []
        
        # Process handle for system metrics
        self._process = psutil.Process()
    
    def start_tracking(self):
        """Start background performance tracking"""
        if self._running:
            return
        
        self._running = True
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()
        self.logger.info("Performance tracking started")
    
    def stop_tracking(self):
        """Stop background performance tracking"""
        self._running = False
        if self._tracking_thread:
            self._tracking_thread.join(timeout=5)
        self.logger.info("Performance tracking stopped")
    
    def _tracking_loop(self):
        """Background tracking loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                self._collect_performance_metrics()
                self._check_performance_anomalies()
                time.sleep(self.config.performance.tracking_interval)
            except Exception as e:
                self.logger.error(f"Error in performance tracking: {e}")
    
    def record_metric(self, metric_name: str, value: float, unit: str = '', metadata: Dict[str, Any] = None):
        """Record a custom performance metric"""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.performance_metrics[metric_name].append(metric)
            
            # Trigger callbacks
            self._trigger_callbacks('metric', metric)
    
    def record_query_performance(self, cube_name: str, query_type: str, 
                               execution_time: float, result_size: int):
        """Record query-specific performance metric"""
        self.record_metric(f'query_execution_time_{cube_name}', execution_time, 'seconds', {
            'query_type': query_type,
            'cube_name': cube_name
        })
        
        self.record_metric(f'query_result_size_{cube_name}', result_size, 'bytes', {
            'query_type': query_type,
            'cube_name': cube_name
        })
        
        # Track queries per second
        self.record_metric('queries_per_second', 1, 'count', {
            'cube_name': cube_name,
            'query_type': query_type
        })
    
    def record_aggregation_performance(self, cube_name: str, cuts_count: int,
                                    execution_time: float, cell_count: int):
        """Record aggregation-specific performance"""
        self.record_metric(f'aggregation_time_{cube_name}', execution_time, 'seconds', {
            'cuts_count': cuts_count,
            'cell_count': cell_count
        })
        
        # Calculate aggregation rate
        if execution_time > 0:
            rate = cell_count / execution_time
            self.record_metric(f'aggregation_rate_{cube_name}', rate, 'cells/sec', {
                'cuts_count': cuts_count
            })
    
    def record_database_performance(self, db_metric: DatabaseMetric):
        """Record database performance metric"""
        with self._lock:
            self.database_metrics.append(db_metric)
            
            # Trigger callbacks
            self._trigger_callbacks('database', db_metric)
    
    def get_performance_metrics(self, metric_name: Optional[str] = None,
                             hours: Optional[int] = None) -> Dict[str, List[PerformanceMetric]]:
        """Get performance metrics with optional filtering"""
        with self._lock:
            metrics = dict(self.performance_metrics)
        
        # Filter by metric name
        if metric_name:
            metrics = {metric_name: metrics.get(metric_name, [])}
        
        # Filter by time
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            for name in metrics:
                metrics[name] = [m for m in metrics[name] if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_resource_usage(self, hours: Optional[int] = None) -> List[ResourceUsage]:
        """Get resource usage history"""
        with self._lock:
            usage = list(self.resource_usage)
        
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            usage = [u for u in usage if u.timestamp >= cutoff_time]
        
        return usage
    
    def get_database_metrics(self, hours: Optional[int] = None) -> List[DatabaseMetric]:
        """Get database performance metrics"""
        with self._lock:
            metrics = list(self.database_metrics)
        
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def get_performance_summary(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        # Get recent metrics
        metrics = self.get_performance_metrics(hours=hours)
        resource_usage = self.get_resource_usage(hours=hours)
        db_metrics = self.get_database_metrics(hours=hours)
        
        summary = {
            'timestamp': time.time(),
            'period_hours': hours or 24,
            'system_performance': self._analyze_system_performance(resource_usage),
            'query_performance': self._analyze_query_performance(metrics),
            'database_performance': self._analyze_database_performance(db_metrics),
            'performance_score': self._calculate_overall_performance_score(),
            'anomalies': self._get_recent_anomalies(hours=hours)
        }
        
        return summary
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        if not self.config.performance.track_cpu_usage and not self.config.performance.track_memory_usage:
            return
        
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1) if self.config.performance.track_cpu_usage else 0
            memory = psutil.virtual_memory() if self.config.performance.track_memory_usage else None
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process_memory = self._process.memory_info() if memory else None
            open_files = len(self._process.open_files())
            threads = self._process.num_threads()
            
            # System-wide process count
            process_count = len(psutil.pids())
            
            resource = ResourceUsage(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent if memory else 0,
                memory_used_mb=memory.used / (1024 * 1024) if memory else 0,
                memory_available_mb=memory.available / (1024 * 1024) if memory else 0,
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                open_files=open_files,
                threads=threads,
                process_count=process_count
            )
            
            with self._lock:
                self.resource_usage.append(resource)
                
                # Trigger callbacks
                self._trigger_callbacks('resource', resource)
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_performance_metrics(self):
        """Collect application performance metrics"""
        try:
            # Garbage collection metrics
            gc_stats = gc.get_stats()
            total_collected = sum(stat['collected'] for stat in gc_stats)
            total_uncollectable = sum(stat['uncollectable'] for stat in gc_stats)
            
            self.record_metric('gc_objects_collected', total_collected, 'objects')
            self.record_metric('gc_objects_uncollectable', total_uncollectable, 'objects')
            
            # Process memory (if not collected in system metrics)
            if not self.config.performance.track_memory_usage:
                process_memory = self._process.memory_info()
                self.record_metric('process_memory_mb', process_memory.rss / (1024 * 1024), 'MB')
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
    
    def _check_performance_anomalies(self):
        """Check for performance anomalies"""
        if not self.config.performance.anomaly_detection:
            return
        
        try:
            # Check resource usage anomalies
            if self.resource_usage:
                recent_usage = list(self.resource_usage)[-10:]  # Last 10 measurements
                
                if len(recent_usage) >= 5:
                    # CPU anomaly detection
                    cpu_values = [u.cpu_percent for u in recent_usage]
                    if self._detect_anomaly(cpu_values, 'cpu'):
                        self._handle_performance_anomaly('cpu', cpu_values[-1])
                    
                    # Memory anomaly detection
                    memory_values = [u.memory_percent for u in recent_usage]
                    if self._detect_anomaly(memory_values, 'memory'):
                        self._handle_performance_anomaly('memory', memory_values[-1])
            
            # Check query performance anomalies
            for metric_name, metric_list in self.performance_metrics.items():
                if 'query_execution_time' in metric_name and len(metric_list) >= 10:
                    recent_times = [m.value for m in list(metric_list)[-10:]]
                    if self._detect_anomaly(recent_times, f'query_time_{metric_name}'):
                        self._handle_performance_anomaly('query_time', recent_times[-1], metric_name)
        
        except Exception as e:
            self.logger.error(f"Error checking performance anomalies: {e}")
    
    def _detect_anomaly(self, values: List[float], metric_type: str) -> bool:
        """Detect anomaly in metric values using statistical methods"""
        if len(values) < 5:
            return False
        
        # Calculate baseline from historical data
        baseline_values = values[:-1]  # Exclude latest value
        latest_value = values[-1]
        
        # Calculate mean and standard deviation
        mean = sum(baseline_values) / len(baseline_values)
        variance = sum((x - mean) ** 2 for x in baseline_values) / len(baseline_values)
        std_dev = variance ** 0.5
        
        # Detect anomaly (using threshold multiplier)
        threshold = self.config.performance.anomaly_threshold_multiplier
        
        # Handle zero standard deviation
        if std_dev == 0:
            return abs(latest_value - mean) > (mean * 0.5)  # 50% change threshold
        
        z_score = abs(latest_value - mean) / std_dev
        return z_score > threshold
    
    def _handle_performance_anomaly(self, anomaly_type: str, value: float, metric_name: str = None):
        """Handle detected performance anomaly"""
        self.logger.warning(f"Performance anomaly detected: {anomaly_type} = {value}")
        
        # Trigger alert if configured
        if self.config.alerts.enabled and self.config.alerts.performance_alert:
            alert_data = {
                'anomaly_type': anomaly_type,
                'value': value,
                'metric_name': metric_name,
                'timestamp': time.time()
            }
            self._trigger_alert('performance_anomaly', alert_data)
    
    def _analyze_system_performance(self, resource_usage: List[ResourceUsage]) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        if not resource_usage:
            return {}
        
        cpu_values = [u.cpu_percent for u in resource_usage]
        memory_values = [u.memory_percent for u in resource_usage]
        disk_values = [u.disk_usage_percent for u in resource_usage]
        
        return {
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'current': memory_values[-1] if memory_values else 0
            },
            'disk': {
                'avg': sum(disk_values) / len(disk_values),
                'max': max(disk_values),
                'min': min(disk_values),
                'current': disk_values[-1] if disk_values else 0
            },
            'sample_count': len(resource_usage)
        }
    
    def _analyze_query_performance(self, metrics: Dict[str, List[PerformanceMetric]]) -> Dict[str, Any]:
        """Analyze query performance metrics"""
        query_metrics = {k: v for k, v in metrics.items() if 'query' in k}
        
        if not query_metrics:
            return {}
        
        analysis = {}
        for metric_name, metric_list in query_metrics.items():
            if not metric_list:
                continue
            
            values = [m.value for m in metric_list]
            analysis[metric_name] = {
                'avg': sum(values) / len(values),
                'max': max(values),
                'min': min(values),
                'count': len(values),
                'latest': values[-1]
            }
        
        return analysis
    
    def _analyze_database_performance(self, db_metrics: List[DatabaseMetric]) -> Dict[str, Any]:
        """Analyze database performance metrics"""
        if not db_metrics:
            return {}
        
        connection_counts = [m.connection_count for m in db_metrics]
        query_times = [m.avg_query_time for m in db_metrics if m.avg_query_time > 0]
        slow_query_counts = [m.slow_query_count for m in db_metrics]
        
        return {
            'connections': {
                'avg': sum(connection_counts) / len(connection_counts),
                'max': max(connection_counts),
                'current': connection_counts[-1] if connection_counts else 0
            },
            'query_time': {
                'avg': sum(query_times) / len(query_times) if query_times else 0,
                'max': max(query_times) if query_times else 0,
                'current': query_times[-1] if query_times else 0
            },
            'slow_queries': {
                'total': sum(slow_query_counts),
                'rate': sum(slow_query_counts) / len(db_metrics)
            }
        }
    
    def _calculate_overall_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            # Get recent resource usage
            recent_usage = list(self.resource_usage)[-10:] if self.resource_usage else []
            
            if not recent_usage:
                return 100.0  # Perfect score if no data
            
            # Calculate resource scores
            cpu_score = max(0, 100 - max(u.cpu_percent for u in recent_usage))
            memory_score = max(0, 100 - max(u.memory_percent for u in recent_usage))
            
            # Calculate query performance score
            query_metrics = [m for metric_list in self.performance_metrics.values() 
                           for m in metric_list if 'query_execution_time' in m.metric_name]
            
            if query_metrics:
                avg_query_time = sum(m.value for m in query_metrics[-100:]) / len(query_metrics[-100:])
                query_score = max(0, 100 - (avg_query_time * 20))  # 5 seconds = 0 score
            else:
                query_score = 100.0
            
            # Weighted overall score
            overall_score = (cpu_score * 0.3 + memory_score * 0.3 + query_score * 0.4)
            return round(overall_score, 2)
        
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return 50.0  # Neutral score on error
    
    def _get_recent_anomalies(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent performance anomalies"""
        # This would integrate with an anomaly tracking system
        # For now, return empty list
        return []
    
    def _trigger_callbacks(self, metric_type: str, data: Any):
        """Trigger performance callbacks"""
        for callback in self._performance_callbacks:
            try:
                callback(metric_type, data)
            except Exception as e:
                self.logger.error(f"Error in performance callback: {e}")
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger performance alert"""
        self.logger.warning(f"Performance alert: {alert_type} - {alert_data}")
    
    def register_callback(self, callback: Callable[[str, Any], None]):
        """Register a performance callback"""
        self._performance_callbacks.append(callback)
    
    def set_performance_baseline(self, metric_name: str, baseline_value: float):
        """Set performance baseline for a metric"""
        self.performance_baselines[metric_name] = baseline_value
    
    def get_performance_baselines(self) -> Dict[str, float]:
        """Get all performance baselines"""
        return dict(self.performance_baselines)
    
    def export_performance_data(self, format: str = 'json') -> str:
        """Export performance data in specified format"""
        with self._lock:
            data = {
                'timestamp': time.time(),
                'performance_metrics': {
                    name: [asdict(m) for m in metrics]
                    for name, metrics in self.performance_metrics.items()
                },
                'resource_usage': [asdict(u) for u in self.resource_usage],
                'database_metrics': [asdict(m) for m in self.database_metrics],
                'performance_baselines': self.performance_baselines,
                'performance_summary': self.get_performance_summary()
            }
        
        if format.lower() == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_data(self, retention_hours: int = 24):
        """Clean up old performance data"""
        cutoff_time = time.time() - (retention_hours * 3600)
        
        with self._lock:
            # Clean up performance metrics
            for metric_name in self.performance_metrics:
                self.performance_metrics[metric_name] = deque(
                    [m for m in self.performance_metrics[metric_name] if m.timestamp >= cutoff_time],
                    maxlen=self.config.performance.max_history_size
                )
            
            # Clean up resource usage
            self.resource_usage = deque(
                [u for u in self.resource_usage if u.timestamp >= cutoff_time],
                maxlen=self.config.performance.max_history_size
            )
            
            # Clean up database metrics
            self.database_metrics = deque(
                [m for m in self.database_metrics if m.timestamp >= cutoff_time],
                maxlen=self.config.performance.max_history_size
            )
        
        self.logger.info(f"Cleaned up performance data older than {retention_hours} hours")
