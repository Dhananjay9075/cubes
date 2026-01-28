# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Comprehensive test suite for Cubes monitoring system
"""

import unittest
import time
import tempfile
import os
import json
from unittest.mock import Mock, patch
from datetime import datetime

# Import monitoring modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cubes.monitoring.config import MonitoringConfig, MonitoringConfigManager
from cubes.monitoring.metrics_collector import OLAPMetricsCollector, QueryMetric
from cubes.monitoring.query_analyzer import QueryAnalyzer
from cubes.monitoring.performance_tracker import PerformanceTracker
from cubes.monitoring.system_monitor import SystemMonitor
from cubes.monitoring.monitoring_manager import CubesMonitoringManager


class TestMonitoringConfig(unittest.TestCase):
    """Test cases for monitoring configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up after tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = MonitoringConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.log_level, "INFO")
        self.assertTrue(config.metrics.enabled)
        self.assertTrue(config.query.enabled)
        self.assertTrue(config.performance.enabled)
        self.assertTrue(config.system.enabled)
        self.assertTrue(config.alerts.enabled)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = MonitoringConfig()
        
        # Test valid configuration
        config.metrics.collection_interval = 30
        config.query.slow_query_threshold = 1.0
        config.system.cpu_warning_threshold = 70.0
        config.system.cpu_critical_threshold = 90.0
        config.__post_init__()  # Should not raise
        
        # Test invalid configuration
        with self.assertRaises(ValueError):
            config.metrics.collection_interval = 0
            config.metrics.__post_init__()
        
        with self.assertRaises(ValueError):
            config.system.cpu_warning_threshold = 90.0
            config.system.cpu_critical_threshold = 80.0
            config.system.__post_init__()
    
    def test_config_manager_file_loading(self):
        """Test configuration manager file loading"""
        config_data = {
            'enabled': True,
            'metrics': {
                'collection_interval': 60,
                'retention_hours': 48
            },
            'query': {
                'slow_query_threshold': 2.0
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = MonitoringConfigManager(self.config_file)
        
        self.assertTrue(config_manager.config.enabled)
        self.assertEqual(config_manager.config.metrics.collection_interval, 60)
        self.assertEqual(config_manager.config.query.slow_query_threshold, 2.0)


class TestOLAPMetricsCollector(unittest.TestCase):
    """Test cases for OLAP metrics collector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MonitoringConfig()
        self.config.metrics.max_history_size = 100
        self.config.query.max_query_history = 100
        
        self.collector = OLAPMetricsCollector(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        self.collector.stop_collection()
    
    def test_query_metric_recording(self):
        """Test query metric recording"""
        query_metric = QueryMetric(
            query_id='test_query_1',
            cube_name='test_cube',
            query_type='aggregate',
            execution_time=0.5,
            result_size=1000,
            rows_returned=100,
            timestamp=time.time(),
            cuts=['date:2023'],
            drills=[],
            dimensions=['date', 'category'],
            measures=['amount'],
            sql_query='SELECT * FROM test'
        )
        
        self.collector.record_query_execution(query_metric)
        
        # Verify metric was recorded
        metrics = self.collector.get_query_metrics()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].query_id, 'test_query_1')
        self.assertEqual(metrics[0].cube_name, 'test_cube')
    
    def test_cube_metric_recording(self):
        """Test cube metric recording"""
        self.collector.record_cube_metric('test_cube', 'test_metric', 42.0, {'label': 'test'})
        
        metrics = self.collector.get_cube_metrics('test_cube')
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].metric_type, 'test_metric')
        self.assertEqual(metrics[0].value, 42.0)
        self.assertEqual(metrics[0].metadata['label'], 'test')
    
    def test_query_statistics(self):
        """Test query statistics calculation"""
        # Record multiple queries
        for i in range(5):
            query_metric = QueryMetric(
                query_id=f'test_query_{i}',
                cube_name='test_cube',
                query_type='aggregate',
                execution_time=0.1 + i * 0.1,
                result_size=100 * i,
                rows_returned=10 * i,
                timestamp=time.time(),
                cuts=[f'date:202{i}'],
                drills=[],
                dimensions=['date'],
                measures=['amount']
            )
            self.collector.record_query_execution(query_metric)
        
        stats = self.collector.get_query_statistics('test_cube')
        
        self.assertIn('test_cube', stats)
        self.assertEqual(stats['test_cube']['total_queries'], 5)
        self.assertEqual(stats['test_cube']['avg_execution_time'], 0.3)
        self.assertEqual(stats['test_cube']['min_execution_time'], 0.1)
        self.assertEqual(stats['test_cube']['max_execution_time'], 0.5)
    
    def test_slow_queries_detection(self):
        """Test slow queries detection"""
        # Record normal query
        normal_query = QueryMetric(
            query_id='normal_query',
            cube_name='test_cube',
            query_type='aggregate',
            execution_time=0.5,
            result_size=100,
            rows_returned=10,
            timestamp=time.time(),
            cuts=[],
            drills=[],
            dimensions=[],
            measures=[]
        )
        
        # Record slow query
        slow_query = QueryMetric(
            query_id='slow_query',
            cube_name='test_cube',
            query_type='aggregate',
            execution_time=2.0,  # Above default threshold of 1.0
            result_size=1000,
            rows_returned=100,
            timestamp=time.time(),
            cuts=[],
            drills=[],
            dimensions=[],
            measures=[]
        )
        
        self.collector.record_query_execution(normal_query)
        self.collector.record_query_execution(slow_query)
        
        slow_queries = self.collector.get_slow_queries()
        self.assertEqual(len(slow_queries), 1)
        self.assertEqual(slow_queries[0].query_id, 'slow_query')
    
    def test_metrics_export_json(self):
        """Test JSON export of metrics"""
        self.collector.record_cube_metric('test_cube', 'test_metric', 42.0)
        
        json_export = self.collector.export_metrics('json')
        data = json.loads(json_export)
        
        self.assertIn('timestamp', data)
        self.assertIn('cube_metrics', data)
        self.assertIn('test_cube', data['cube_metrics'])
        self.assertEqual(len(data['cube_metrics']['test_cube']), 1)
        self.assertEqual(data['cube_metrics']['test_cube'][0]['value'], 42.0)


class TestQueryAnalyzer(unittest.TestCase):
    """Test cases for query analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MonitoringConfig()
        self.analyzer = QueryAnalyzer(self.config)
    
    def test_query_pattern_analysis(self):
        """Test query pattern analysis"""
        pattern = self.analyzer.analyze_query(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023', 'category:electronics'],
            drills=['category'],
            dimensions=['date', 'category', 'product'],
            measures=['amount', 'quantity'],
            execution_time=0.5,
            result_size=1000
        )
        
        self.assertEqual(pattern.cuts_count, 2)
        self.assertEqual(pattern.drills_count, 1)
        self.assertEqual(len(pattern.dimensions_used), 3)
        self.assertEqual(len(pattern.measures_used), 2)
        self.assertGreater(pattern.complexity_score, 0)
    
    def test_query_complexity_analysis(self):
        """Test query complexity analysis"""
        complexity = self.analyzer.analyze_complexity(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023'] * 10,  # Many cuts
            drills=['category'] * 5,   # Many drills
            dimensions=['dim1'] * 10,   # Many dimensions
            measures=['measure1'] * 5,  # Many measures
            execution_time=2.0,
            result_size=10000
        )
        
        self.assertGreater(complexity.score, 0)
        self.assertIn(complexity.level, ['low', 'medium', 'high', 'very_high'])
        self.assertIsInstance(complexity.factors, dict)
    
    def test_pattern_type_determination(self):
        """Test pattern type determination"""
        # Simple query
        simple_pattern = self.analyzer._analyze_pattern_structure(
            'aggregate', [], [], ['date'], ['amount']
        )
        self.assertEqual(simple_pattern.pattern_type, 'simple')
        
        # Complex analytical query
        complex_pattern = self.analyzer._analyze_pattern_structure(
            'aggregate', ['cut1'] * 15, ['drill1'] * 6, ['dim1'] * 10, ['measure1'] * 6
        )
        self.assertEqual(complex_pattern.pattern_type, 'complex_analytical')
    
    def test_sql_query_analysis(self):
        """Test SQL query analysis"""
        sql_query = """
        SELECT d.date, c.category, SUM(f.amount) as total_amount
        FROM facts f
        JOIN dates d ON f.date_id = d.id
        JOIN categories c ON f.category_id = c.id
        WHERE d.year = 2023
        GROUP BY d.date, c.category
        ORDER BY total_amount DESC
        """
        
        analysis = self.analyzer.analyze_sql_query(sql_query)
        
        self.assertIn('query_length', analysis)
        self.assertIn('joins_count', analysis)
        self.assertIn('group_by_count', analysis)
        self.assertIn('sql_complexity', analysis)
        self.assertGreater(analysis['joins_count'], 0)
        self.assertEqual(analysis['group_by_count'], 1)


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for performance tracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MonitoringConfig()
        self.config.performance.max_history_size = 100
        self.tracker = PerformanceTracker(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        self.tracker.stop_tracking()
    
    def test_metric_recording(self):
        """Test custom metric recording"""
        self.tracker.record_metric('test_metric', 42.0, 'units', {'label': 'test'})
        
        metrics = self.tracker.get_performance_metrics('test_metric')
        self.assertIn('test_metric', metrics)
        self.assertEqual(len(metrics['test_metric']), 1)
        self.assertEqual(metrics['test_metric'][0].value, 42.0)
        self.assertEqual(metrics['test_metric'][0].unit, 'units')
    
    def test_query_performance_recording(self):
        """Test query performance recording"""
        self.tracker.record_query_performance('test_cube', 'aggregate', 0.5, 1000)
        
        metrics = self.tracker.get_performance_metrics()
        
        # Should have query execution time metric
        query_time_metrics = metrics.get('query_execution_time_test_cube', [])
        self.assertGreater(len(query_time_metrics), 0)
        self.assertEqual(query_time_metrics[0].value, 0.5)
        
        # Should have result size metric
        result_size_metrics = metrics.get('query_result_size_test_cube', [])
        self.assertGreater(len(result_size_metrics), 0)
        self.assertEqual(result_size_metrics[0].value, 1000)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Record some test metrics
        self.tracker.record_metric('test_metric', 10.0)
        self.tracker.record_metric('test_metric', 20.0)
        self.tracker.record_metric('test_metric', 15.0)
        
        summary = self.tracker.get_performance_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('system_performance', summary)
        self.assertIn('query_performance', summary)
        self.assertIn('performance_score', summary)
        self.assertGreaterEqual(summary['performance_score'], 0)
        self.assertLessEqual(summary['performance_score'], 100)


class TestSystemMonitor(unittest.TestCase):
    """Test cases for system monitor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MonitoringConfig()
        self.config.system.max_history_size = 100
        self.monitor = SystemMonitor(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        self.monitor.stop_monitoring()
    
    def test_cpu_health_check(self):
        """Test CPU health check"""
        status, message, metadata = self.monitor._check_cpu_usage()
        
        self.assertIn(status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(message, str)
        self.assertIsInstance(metadata, dict)
        self.assertIn('cpu_percent', metadata)
    
    def test_memory_health_check(self):
        """Test memory health check"""
        status, message, metadata = self.monitor._check_memory_usage()
        
        self.assertIn(status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(message, str)
        self.assertIsInstance(metadata, dict)
        self.assertIn('memory_percent', metadata)
    
    def test_disk_health_check(self):
        """Test disk health check"""
        status, message, metadata = self.monitor._check_disk_usage()
        
        self.assertIn(status, ['healthy', 'warning', 'critical'])
        self.assertIsInstance(message, str)
        self.assertIsInstance(metadata, dict)
        self.assertIn('disk_percent', metadata)
    
    def test_custom_health_check_registration(self):
        """Test custom health check registration"""
        def custom_check():
            return {
                'status': 'healthy',
                'message': 'Custom check passed',
                'metadata': {'custom': 'data'}
            }
        
        self.monitor.register_health_check('custom_check', custom_check)
        
        self.assertIn('custom_check', self.monitor._custom_checks)
        
        # Perform the custom check
        result = self.monitor.perform_health_check('custom_check')
        
        self.assertEqual(result.status, 'healthy')
        self.assertEqual(result.message, 'Custom check passed')
        self.assertEqual(result.metadata['custom'], 'data')
    
    def test_system_status_calculation(self):
        """Test overall system status calculation"""
        # Set up some component statuses
        self.monitor.component_status = {
            'cpu': 'healthy',
            'memory': 'warning',
            'disk': 'healthy'
        }
        
        overall_status, health_score, active_issues, recommendations = self.monitor._calculate_system_status()
        
        self.assertEqual(overall_status, 'warning')  # Due to memory warning
        self.assertGreaterEqual(health_score, 0)
        self.assertLessEqual(health_score, 100)
        self.assertIsInstance(active_issues, list)
        self.assertIsInstance(recommendations, list)


class TestCubesMonitoringManager(unittest.TestCase):
    """Test cases for Cubes monitoring manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MonitoringConfig()
        self.manager = CubesMonitoringManager(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        self.manager.stop_monitoring()
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        self.assertIsNotNone(self.manager.metrics_collector)
        self.assertIsNotNone(self.manager.query_analyzer)
        self.assertIsNotNone(self.manager.performance_tracker)
        self.assertIsNotNone(self.manager.system_monitor)
        self.assertFalse(self.manager._running)
    
    def test_query_execution_recording(self):
        """Test query execution recording across all components"""
        result = self.manager.record_query_execution(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023'],
            drills=[],
            dimensions=['date', 'category'],
            measures=['amount'],
            execution_time=0.5,
            result_size=1000
        )
        
        self.assertIn('query_id', result)
        self.assertIn('pattern', result)
        self.assertIn('complexity', result)
        self.assertIn('performance_score', result)
        
        # Verify metrics were recorded in collector
        query_metrics = self.manager.metrics_collector.get_query_metrics('test_cube')
        self.assertEqual(len(query_metrics), 1)
        self.assertEqual(query_metrics[0].cube_name, 'test_cube')
    
    def test_cube_operation_recording(self):
        """Test cube operation recording"""
        self.manager.record_cube_operation('test_cube', 'test_operation', 0.3, {'key': 'value'})
        
        # Verify metric was recorded
        cube_metrics = self.manager.metrics_collector.get_cube_metrics('test_cube', 'test_operation_duration')
        self.assertEqual(len(cube_metrics), 1)
        self.assertEqual(cube_metrics[0].value, 0.3)
    
    def test_monitoring_dashboard(self):
        """Test monitoring dashboard generation"""
        # Record some test data
        self.manager.record_query_execution(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023'],
            drills=[],
            dimensions=['date'],
            measures=['amount'],
            execution_time=0.5,
            result_size=1000
        )
        
        dashboard = self.manager.get_monitoring_dashboard()
        
        self.assertIn('timestamp', dashboard)
        self.assertIn('system_status', dashboard)
        self.assertIn('performance_summary', dashboard)
        self.assertIn('query_metrics', dashboard)
        self.assertIn('cube_metrics', dashboard)
        self.assertIn('health_trends', dashboard)
        self.assertIn('query_patterns', dashboard)
        self.assertIn('complexity_trends', dashboard)
        self.assertIn('monitoring_uptime', dashboard)
    
    def test_cube_insights(self):
        """Test cube insights generation"""
        # Record some test data
        self.manager.record_query_execution(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023'],
            drills=[],
            dimensions=['date'],
            measures=['amount'],
            execution_time=0.5,
            result_size=1000
        )
        
        insights = self.manager.get_cube_insights('test_cube')
        
        self.assertIn('cube_name', insights)
        self.assertIn('query_statistics', insights)
        self.assertIn('cube_summary', insights)
        self.assertIn('performance_insights', insights)
        self.assertIn('slow_queries', insights)
        self.assertIn('optimization_suggestions', insights)
        self.assertEqual(insights['cube_name'], 'test_cube')
    
    def test_monitoring_status(self):
        """Test monitoring system status"""
        status = self.manager.get_monitoring_status()
        
        self.assertIn('running', status)
        self.assertIn('uptime', status)
        self.assertIn('components', status)
        self.assertIn('configuration', status)
        
        # Check component status
        self.assertIn('metrics', status['components'])
        self.assertIn('query', status['components'])
        self.assertIn('performance', status['components'])
        self.assertIn('system', status['components'])
        self.assertIn('alerts', status['components'])
    
    def test_context_manager(self):
        """Test context manager functionality"""
        with self.manager as manager:
            self.assertTrue(manager._running)
        
        # Should be stopped after exiting context
        self.assertFalse(manager._running)


def run_all_tests():
    """Run all test suites"""
    test_classes = [
        TestMonitoringConfig,
        TestOLAPMetricsCollector,
        TestQueryAnalyzer,
        TestPerformanceTracker,
        TestSystemMonitor,
        TestCubesMonitoringManager
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
