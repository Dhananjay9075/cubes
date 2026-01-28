# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
System monitor for Cubes OLAP framework
Monitors system health and resource availability
"""

import time
import threading
import psutil
import socket
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: str  # healthy, warning, critical, unknown
    message: str
    timestamp: float
    response_time: float
    metadata: Dict[str, Any]


@dataclass
class SystemStatus:
    """Overall system status"""
    overall_status: str
    health_score: float
    timestamp: float
    component_status: Dict[str, str]
    active_issues: List[Dict[str, Any]]
    recommendations: List[str]


class SystemMonitor:
    """Monitors system health and resource availability"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Health check storage
        self.health_checks = defaultdict(lambda: deque(maxlen=config.system.max_history_size))
        self.component_status = {}
        self.last_check_time = 0
        
        # System status history
        self.status_history = deque(maxlen=config.system.max_history_size)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._running = False
        self._monitoring_thread = None
        
        # Health check callbacks
        self._health_callbacks = []
        
        # Custom health check functions
        self._custom_checks = {}
    
    def start_monitoring(self):
        """Start background system monitoring"""
        if self._running:
            return
        
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background system monitoring"""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                self._perform_health_checks()
                self._update_system_status()
                time.sleep(self.config.system.check_interval)
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
    
    def register_health_check(self, name: str, check_function: Callable[[], Dict[str, Any]]):
        """Register a custom health check function"""
        self._custom_checks[name] = check_function
        self.logger.info(f"Registered custom health check: {name}")
    
    def perform_health_check(self, check_name: str) -> HealthCheck:
        """Perform a specific health check"""
        start_time = time.time()
        
        try:
            if check_name in self._custom_checks:
                # Custom health check
                result = self._custom_checks[check_name]()
                status = result.get('status', 'unknown')
                message = result.get('message', 'Custom check completed')
                metadata = result.get('metadata', {})
            else:
                # Built-in health check
                result = self._perform_builtin_check(check_name)
                status, message, metadata = result
            
            response_time = time.time() - start_time
            
            health_check = HealthCheck(
                name=check_name,
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time,
                metadata=metadata
            )
            
            with self._lock:
                self.health_checks[check_name].append(health_check)
                self.component_status[check_name] = status
            
            return health_check
        
        except Exception as e:
            response_time = time.time() - start_time
            error_check = HealthCheck(
                name=check_name,
                status='critical',
                message=f"Health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=response_time,
                metadata={'error': str(e)}
            )
            
            with self._lock:
                self.health_checks[check_name].append(error_check)
                self.component_status[check_name] = 'critical'
            
            return error_check
    
    def _perform_builtin_check(self, check_name: str) -> tuple:
        """Perform built-in health checks"""
        if check_name == 'cpu':
            return self._check_cpu_usage()
        elif check_name == 'memory':
            return self._check_memory_usage()
        elif check_name == 'disk':
            return self._check_disk_usage()
        elif check_name == 'network':
            return self._check_network_connectivity()
        elif check_name == 'database':
            return self._check_database_connectivity()
        elif check_name == 'processes':
            return self._check_process_health()
        else:
            return 'unknown', f"Unknown health check: {check_name}", {}
    
    def _check_cpu_usage(self) -> tuple:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent >= self.config.system.cpu_critical_threshold:
                status = 'critical'
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent >= self.config.system.cpu_warning_threshold:
                status = 'warning'
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = 'healthy'
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            metadata = {
                'cpu_percent': cpu_percent,
                'warning_threshold': self.config.system.cpu_warning_threshold,
                'critical_threshold': self.config.system.cpu_critical_threshold
            }
            
            return status, message, metadata
        
        except Exception as e:
            return 'critical', f"Failed to check CPU usage: {str(e)}", {'error': str(e)}
    
    def _check_memory_usage(self) -> tuple:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent >= self.config.system.memory_critical_threshold:
                status = 'critical'
                message = f"Memory usage critical: {memory.percent:.1f}%"
            elif memory.percent >= self.config.system.memory_warning_threshold:
                status = 'warning'
                message = f"Memory usage high: {memory.percent:.1f}%"
            else:
                status = 'healthy'
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            metadata = {
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'warning_threshold': self.config.system.memory_warning_threshold,
                'critical_threshold': self.config.system.memory_critical_threshold
            }
            
            return status, message, metadata
        
        except Exception as e:
            return 'critical', f"Failed to check memory usage: {str(e)}", {'error': str(e)}
    
    def _check_disk_usage(self) -> tuple:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            
            if disk.percent >= self.config.system.disk_critical_threshold:
                status = 'critical'
                message = f"Disk usage critical: {disk.percent:.1f}%"
            elif disk.percent >= self.config.system.disk_warning_threshold:
                status = 'warning'
                message = f"Disk usage high: {disk.percent:.1f}%"
            else:
                status = 'healthy'
                message = f"Disk usage normal: {disk.percent:.1f}%"
            
            metadata = {
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'warning_threshold': self.config.system.disk_warning_threshold,
                'critical_threshold': self.config.system.disk_critical_threshold
            }
            
            return status, message, metadata
        
        except Exception as e:
            return 'critical', f"Failed to check disk usage: {str(e)}", {'error': str(e)}
    
    def _check_network_connectivity(self) -> tuple:
        """Check network connectivity"""
        try:
            # Test basic network connectivity
            test_hosts = ['8.8.8.8', '1.1.1.1']
            connected_hosts = 0
            
            for host in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((host, 53))
                    sock.close()
                    
                    if result == 0:
                        connected_hosts += 1
                except:
                    pass
            
            if connected_hosts == 0:
                status = 'critical'
                message = "No network connectivity"
            elif connected_hosts < len(test_hosts):
                status = 'warning'
                message = f"Limited network connectivity: {connected_hosts}/{len(test_hosts)} hosts reachable"
            else:
                status = 'healthy'
                message = "Network connectivity normal"
            
            metadata = {
                'connected_hosts': connected_hosts,
                'total_hosts': len(test_hosts),
                'test_hosts': test_hosts
            }
            
            return status, message, metadata
        
        except Exception as e:
            return 'critical', f"Failed to check network connectivity: {str(e)}", {'error': str(e)}
    
    def _check_database_connectivity(self) -> tuple:
        """Check database connectivity (placeholder)"""
        # This would be implemented based on the actual database configuration
        # For now, return a placeholder status
        status = 'healthy'
        message = "Database connectivity check not implemented"
        metadata = {'implementation': 'placeholder'}
        
        return status, message, metadata
    
    def _check_process_health(self) -> tuple:
        """Check process health"""
        try:
            current_process = psutil.Process()
            
            # Check if process is responsive
            status = 'healthy'
            issues = []
            
            # Check number of threads
            thread_count = current_process.num_threads()
            if thread_count > 1000:  # Arbitrary high number
                issues.append(f"High thread count: {thread_count}")
                status = 'warning'
            
            # Check file descriptors
            try:
                open_files = len(current_process.open_files())
                if open_files > 1000:  # Arbitrary high number
                    issues.append(f"High file descriptor count: {open_files}")
                    if status == 'healthy':
                        status = 'warning'
            except (psutil.AccessDenied, OSError):
                pass  # May not have permission to check open files
            
            # Check memory usage of this process
            process_memory = current_process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)
            
            if process_memory_mb > 2048:  # 2GB
                issues.append(f"High process memory usage: {process_memory_mb:.1f}MB")
                if status == 'healthy':
                    status = 'warning'
            
            message = "; ".join(issues) if issues else "Process health normal"
            
            metadata = {
                'pid': current_process.pid,
                'thread_count': thread_count,
                'memory_mb': process_memory_mb,
                'open_files': open_files if 'open_files' in locals() else 0,
                'issues': issues
            }
            
            return status, message, metadata
        
        except Exception as e:
            return 'critical', f"Failed to check process health: {str(e)}", {'error': str(e)}
    
    def _perform_health_checks(self):
        """Perform all registered health checks"""
        check_names = ['cpu', 'memory', 'disk', 'network', 'database', 'processes']
        check_names.extend(self._custom_checks.keys())
        
        for check_name in check_names:
            try:
                health_check = self.perform_health_check(check_name)
                self._trigger_health_callbacks(health_check)
            except Exception as e:
                self.logger.error(f"Error performing health check {check_name}: {e}")
    
    def _update_system_status(self):
        """Update overall system status"""
        try:
            overall_status, health_score, active_issues, recommendations = self._calculate_system_status()
            
            system_status = SystemStatus(
                overall_status=overall_status,
                health_score=health_score,
                timestamp=time.time(),
                component_status=dict(self.component_status),
                active_issues=active_issues,
                recommendations=recommendations
            )
            
            with self._lock:
                self.status_history.append(system_status)
                self.last_check_time = time.time()
        
        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")
    
    def _calculate_system_status(self) -> tuple:
        """Calculate overall system status"""
        if not self.component_status:
            return 'unknown', 0.0, [], ["No health checks performed"]
        
        # Count statuses
        status_counts = defaultdict(int)
        for status in self.component_status.values():
            status_counts[status] += 1
        
        # Determine overall status
        if status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        elif status_counts['healthy'] == len(self.component_status):
            overall_status = 'healthy'
        else:
            overall_status = 'unknown'
        
        # Calculate health score
        total_checks = len(self.component_status)
        healthy_checks = status_counts['healthy']
        warning_checks = status_counts['warning']
        
        health_score = (healthy_checks * 100 + warning_checks * 50) / total_checks
        
        # Identify active issues
        active_issues = []
        for check_name, status in self.component_status.items():
            if status in ['critical', 'warning']:
                latest_check = list(self.health_checks[check_name])[-1] if self.health_checks[check_name] else None
                if latest_check:
                    active_issues.append({
                        'component': check_name,
                        'status': status,
                        'message': latest_check.message,
                        'timestamp': latest_check.timestamp
                    })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(active_issues)
        
        return overall_status, health_score, active_issues, recommendations
    
    def _generate_recommendations(self, active_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on active issues"""
        recommendations = []
        
        for issue in active_issues:
            component = issue['component']
            status = issue['status']
            
            if component == 'cpu' and status == 'critical':
                recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive queries")
            elif component == 'memory' and status == 'critical':
                recommendations.append("Consider increasing memory or optimizing memory usage")
            elif component == 'disk' and status == 'critical':
                recommendations.append("Free up disk space or consider disk cleanup")
            elif component == 'network' and status != 'healthy':
                recommendations.append("Check network connectivity and firewall settings")
            elif component == 'database' and status != 'healthy':
                recommendations.append("Check database connection and configuration")
        
        return recommendations
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        if not self.status_history:
            # Perform immediate status check
            self._perform_health_checks()
            self._update_system_status()
        
        return self.status_history[-1] if self.status_history else SystemStatus(
            overall_status='unknown',
            health_score=0.0,
            timestamp=time.time(),
            component_status={},
            active_issues=[],
            recommendations=["System monitoring not initialized"]
        )
    
    def get_health_history(self, component: Optional[str] = None, 
                          hours: Optional[int] = None) -> Dict[str, List[HealthCheck]]:
        """Get health check history"""
        with self._lock:
            history = dict(self.health_checks)
        
        # Filter by component
        if component:
            history = {component: history.get(component, [])}
        
        # Filter by time
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            for check_name in history:
                history[check_name] = [h for h in history[check_name] if h.timestamp >= cutoff_time]
        
        return history
    
    def get_status_trends(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """Get system status trends"""
        with self._lock:
            status_list = list(self.status_history)
        
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            status_list = [s for s in status_list if s.timestamp >= cutoff_time]
        
        if not status_list:
            return {}
        
        # Calculate trends
        health_scores = [s.health_score for s in status_list]
        status_counts = defaultdict(list)
        
        for status in status_list:
            status_counts[status.overall_status].append(status.timestamp)
        
        return {
            'period_hours': hours or 24,
            'data_points': len(status_list),
            'health_score': {
                'current': health_scores[-1] if health_scores else 0,
                'average': sum(health_scores) / len(health_scores) if health_scores else 0,
                'min': min(health_scores) if health_scores else 0,
                'max': max(health_scores) if health_scores else 0,
                'trend': self._calculate_trend(health_scores)
            },
            'status_distribution': {
                status: len(timestamps) for status, timestamps in status_counts.items()
            },
            'uptime_percentage': self._calculate_uptime(status_list)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation
        recent_avg = sum(values[-5:]) / min(5, len(values))
        earlier_avg = sum(values[:5]) / min(5, len(values))
        
        change = recent_avg - earlier_avg
        
        if abs(change) < 5:
            return 'stable'
        elif change > 0:
            return 'improving'
        else:
            return 'degrading'
    
    def _calculate_uptime(self, status_list: List[SystemStatus]) -> float:
        """Calculate uptime percentage"""
        if not status_list:
            return 0.0
        
        healthy_count = sum(1 for s in status_list if s.overall_status == 'healthy')
        return (healthy_count / len(status_list)) * 100
    
    def _trigger_health_callbacks(self, health_check: HealthCheck):
        """Trigger health check callbacks"""
        for callback in self._health_callbacks:
            try:
                callback(health_check)
            except Exception as e:
                self.logger.error(f"Error in health callback: {e}")
    
    def register_callback(self, callback: Callable[[HealthCheck], None]):
        """Register a health check callback"""
        self._health_callbacks.append(callback)
    
    def export_health_data(self, format: str = 'json') -> str:
        """Export health data in specified format"""
        with self._lock:
            data = {
                'timestamp': time.time(),
                'current_status': asdict(self.get_system_status()),
                'health_history': {
                    name: [asdict(h) for h in checks]
                    for name, checks in self.health_checks.items()
                },
                'status_trends': self.get_status_trends(),
                'component_status': dict(self.component_status)
            }
        
        if format.lower() == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def trigger_manual_check(self, check_name: Optional[str] = None):
        """Trigger manual health check"""
        if check_name:
            return self.perform_health_check(check_name)
        else:
            self._perform_health_checks()
            self._update_system_status()
            return self.get_system_status()
