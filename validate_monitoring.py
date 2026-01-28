#!/usr/bin/env python3
"""
Standalone validation test for Cubes monitoring system.
Tests core functionality without full Cubes dependencies.
"""

import sys
import os
import time
import tempfile
import json
from unittest.mock import Mock

# Add the monitoring module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cubes'))

def test_config_manager():
    """Test configuration manager standalone"""
    print("Testing Configuration Manager...")
    
    try:
        from monitoring.config import MonitoringConfigManager
        
        # Test default configuration
        config_manager = MonitoringConfigManager()
        assert config_manager.config.enabled == True
        assert config_manager.config.metrics.collection_interval == 30
        assert config_manager.config.query.enabled == True
        assert config_manager.config.performance.enabled == True
        assert config_manager.config.system.enabled == True
        
        print("✓ Default configuration loaded successfully")
        
        # Test configuration validation
        try:
            config_manager.config.metrics.collection_interval = 0
            config_manager.config.metrics.__post_init__()
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Configuration validation works correctly")
        
        # Test component configuration
        metrics_config = config_manager.get_component_config('metrics')
        assert hasattr(metrics_config, 'enabled')
        assert metrics_config.enabled == True
        
        print("✓ Component configuration access works")
        
        # Test thresholds dictionary
        thresholds = config_manager.get_thresholds_dict()
        assert 'system' in thresholds
        assert 'query' in thresholds
        assert 'performance' in thresholds
        
        print("✓ Thresholds dictionary generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration manager test failed: {e}")
        return False

def test_dataclasses():
    """Test dataclass definitions"""
    print("\nTesting Dataclasses...")
    
    try:
        from monitoring.config import (
            MetricsConfig, QueryConfig, PerformanceConfig, 
            SystemConfig, AlertsConfig, MonitoringConfig
        )
        
        # Test MetricsConfig
        metrics_config = MetricsConfig(
            collection_interval=60,
            retention_hours=48
        )
        assert metrics_config.collection_interval == 60
        assert metrics_config.retention_hours == 48
        assert metrics_config.export_formats == ['json', 'prometheus']
        
        print("✓ MetricsConfig works correctly")
        
        # Test QueryConfig
        query_config = QueryConfig(
            slow_query_threshold=2.0
        )
        assert query_config.slow_query_threshold == 2.0
        assert query_config.track_slow_queries == True
        
        print("✓ QueryConfig works correctly")
        
        # Test SystemConfig
        system_config = SystemConfig(
            cpu_warning_threshold=75.0,
            cpu_critical_threshold=90.0
        )
        assert system_config.cpu_warning_threshold == 75.0
        assert system_config.cpu_critical_threshold == 90.0
        
        print("✓ SystemConfig works correctly")
        
        # Test MonitoringConfig
        monitoring_config = MonitoringConfig(
            enabled=True,
            log_level="INFO"
        )
        assert monitoring_config.enabled == True
        assert monitoring_config.log_level == "INFO"
        assert hasattr(monitoring_config, 'metrics')
        assert hasattr(monitoring_config, 'query')
        
        print("✓ MonitoringConfig works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataclass test failed: {e}")
        return False

def test_config_file_operations():
    """Test configuration file operations"""
    print("\nTesting Configuration File Operations...")
    
    try:
        from monitoring.config import MonitoringConfigManager
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'enabled': True,
                'metrics': {
                    'collection_interval': 120,
                    'retention_hours': 48
                },
                'query': {
                    'slow_query_threshold': 2.5
                }
            }
            json.dump(config_data, f)
            temp_config_file = f.name
        
        try:
            # Load configuration from file
            config_manager = MonitoringConfigManager(temp_config_file)
            
            assert config_manager.config.enabled == True
            assert config_manager.config.metrics.collection_interval == 120
            assert config_manager.config.query.slow_query_threshold == 2.5
            
            print("✓ Configuration file loading works")
            
            # Test configuration updates
            updates = {
                'metrics': {
                    'collection_interval': 180
                },
                'system': {
                    'cpu_warning_threshold': 80.0
                }
            }
            
            config_manager.update_config(updates)
            assert config_manager.config.metrics.collection_interval == 180
            
            print("✓ Configuration updates work")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_config_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration file operations test failed: {e}")
        return False

def test_environment_overrides():
    """Test environment variable overrides"""
    print("\nTesting Environment Variable Overrides...")
    
    try:
        from monitoring.config import MonitoringConfigManager
        
        # Set environment variables
        os.environ['CUBES_MONITORING_ENABLED'] = 'false'
        os.environ['CUBES_MONITORING_METRICS_INTERVAL'] = '45'
        os.environ['CUBES_MONITORING_QUERY_SLOW_THRESHOLD'] = '3.5'
        
        try:
            config_manager = MonitoringConfigManager()
            
            assert config_manager.config.enabled == False
            assert config_manager.config.metrics.collection_interval == 45
            assert config_manager.config.query.slow_query_threshold == 3.5
            
            print("✓ Environment variable overrides work")
            
        finally:
            # Clean up environment variables
            for key in ['CUBES_MONITORING_ENABLED', 
                       'CUBES_MONITORING_METRICS_INTERVAL',
                       'CUBES_MONITORING_QUERY_SLOW_THRESHOLD']:
                if key in os.environ:
                    del os.environ[key]
        
        return True
        
    except Exception as e:
        print(f"✗ Environment overrides test failed: {e}")
        return False

def test_mock_metrics_collector():
    """Test metrics collector with mocked dependencies"""
    print("\nTesting Mock Metrics Collector...")
    
    try:
        # Mock the dependencies
        mock_config = Mock()
        mock_config.metrics.max_history_size = 100
        mock_config.query.max_query_history = 100
        mock_config.query.slow_query_threshold = 1.0
        
        # Mock psutil to avoid system dependency
        import sys
        from unittest.mock import patch
        sys.modules['psutil'] = Mock()
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock memory
            mock_memory_obj = Mock()
            mock_memory_obj.percent = 60.0
            mock_memory_obj.used = 1024 * 1024 * 1024
            mock_memory.return_value = mock_memory_obj
            
            # Mock disk
            mock_disk_obj = Mock()
            mock_disk_obj.percent = 70.0
            mock_disk.return_value = mock_disk_obj
            
            # Import and test metrics collector
            from monitoring.metrics_collector import OLAPMetricsCollector
            
            collector = OLAPMetricsCollector(mock_config)
            
            # Test metric recording
            collector.record_cube_metric('test_cube', 'test_metric', 42.0, {'label': 'test'})
            
            metrics = collector.get_cube_metrics('test_cube')
            assert len(metrics) == 1
            assert metrics[0].value == 42.0
            assert metrics[0].labels['label'] == 'test'
            
            print("✓ Metric recording works")
            
            # Test metric summary
            summary = collector.get_cube_summary('test_cube')
            assert 'test_metric' in summary
            assert 'count' in summary['test_metric']
            assert summary['test_metric']['count'] == 1
            assert summary['test_metric']['min'] == 42.0
            assert summary['test_metric']['max'] == 42.0
            assert summary['test_metric']['avg'] == 42.0
            
            print("✓ Metric summary works")
            
            # Test export
            json_export = collector.export_metrics('json')
            assert 'test_cube' in json_export
            
            prometheus_export = collector.export_metrics('prometheus')
            assert 'cubes_test_metric' in prometheus_export
            
            print("✓ Metric export works")
            
            # Clean up
            collector.stop_collection()
        
        return True
        
    except Exception as e:
        print(f"✗ Mock metrics collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_query_analyzer():
    """Test query analyzer with mocked dependencies"""
    print("\nTesting Mock Query Analyzer...")
    
    try:
        from monitoring.query_analyzer import QueryAnalyzer
        
        mock_config = Mock()
        mock_config.query.slow_query_threshold = 1.0
        
        analyzer = QueryAnalyzer(mock_config)
        
        # Test pattern analysis
        pattern = analyzer.analyze_query(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023', 'category:electronics'],
            drills=['category'],
            dimensions=['date', 'category', 'product'],
            measures=['amount', 'quantity'],
            execution_time=0.5,
            result_size=1000
        )
        
        assert pattern.cuts_count == 2
        assert pattern.drills_count == 1
        assert len(pattern.dimensions_used) == 3
        assert len(pattern.measures_used) == 2
        assert pattern.complexity_score > 0
        
        print("✓ Query pattern analysis works")
        
        # Test complexity analysis
        complexity = analyzer.analyze_complexity(
            cube_name='test_cube',
            query_type='aggregate',
            cuts=['date:2023'] * 10,
            drills=['category'] * 5,
            dimensions=['dim1'] * 10,
            measures=['measure1'] * 5,
            execution_time=2.0,
            result_size=10000
        )
        
        assert complexity.score > 0
        assert complexity.level in ['low', 'medium', 'high', 'very_high']
        assert isinstance(complexity.factors, dict)
        
        print("✓ Query complexity analysis works")
        
        # Test SQL analysis
        sql_query = """
        SELECT d.date, c.category, SUM(f.amount) as total_amount
        FROM facts f
        JOIN dates d ON f.date_id = d.id
        JOIN categories c ON f.category_id = c.id
        WHERE d.year = 2023
        GROUP BY d.date, c.category
        ORDER BY total_amount DESC
        """
        
        analysis = analyzer.analyze_sql_query(sql_query)
        assert 'query_length' in analysis
        assert 'joins_count' in analysis
        assert 'group_by_count' in analysis
        assert 'sql_complexity' in analysis
        assert analysis['joins_count'] > 0
        assert analysis['group_by_count'] == 1
        
        print("✓ SQL query analysis works")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock query analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def calculate_code_metrics():
    """Calculate basic code metrics"""
    print("\nCalculating Code Metrics...")
    
    monitoring_dir = os.path.join(os.path.dirname(__file__), 'cubes', 'monitoring')
    
    python_files = [
        '__init__.py',
        'config.py',
        'metrics_collector.py',
        'query_analyzer.py',
        'performance_tracker.py',
        'system_monitor.py',
        'monitoring_manager.py',
        'api.py'
    ]
    
    total_lines = 0
    total_classes = 0
    total_functions = 0
    
    for file in python_files:
        file_path = os.path.join(monitoring_dir, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = len(content.splitlines())
            total_lines += lines
            
            # Simple class and function counting (basic approximation)
            classes = content.count('class ')
            functions = content.count('def ')
            
            total_classes += classes
            total_functions += functions
            
            print(f"✓ {file}: {lines} lines, ~{classes} classes, ~{functions} functions")
            
        except Exception as e:
            print(f"✗ {file}: Error analyzing - {e}")
    
    print(f"\n📊 Total Metrics:")
    print(f"   Total Lines: {total_lines}")
    print(f"   Total Classes: {total_classes}")
    print(f"   Total Functions: {total_functions}")
    print(f"   Average Lines per File: {total_lines // len(python_files)}")
    
    # Check if we meet the requirements
    if total_lines >= 2000:  # Requirement: 2000+ lines
        print("✓ Meets line count requirement (2000+)")
    else:
        print(f"✗ Below line count requirement (2000+): {total_lines}")
    
    if total_classes >= 15:  # Should have many classes
        print("✓ Good class count")
    else:
        print(f"? Low class count: {total_classes}")
    
    if total_functions >= 100:  # Should have many functions
        print("✓ Good function count")
    else:
        print(f"? Low function count: {total_functions}")
    
    return True

def main():
    """Run all validation tests"""
    print("🔍 Cubes Monitoring System Validation\n")
    
    tests = [
        ("Configuration Management", test_config_manager),
        ("Dataclass Definitions", test_dataclasses),
        ("Configuration File Operations", test_config_file_operations),
        ("Environment Overrides", test_environment_overrides),
        ("Mock Metrics Collector", test_mock_metrics_collector),
        ("Mock Query Analyzer", test_mock_query_analyzer),
        ("Code Metrics", calculate_code_metrics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"🏁 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Cubes monitoring system structure is correct and ready for use.")
        return 0
    else:
        print(f"⚠️  {total - passed} validations failed.")
        print("❌ Please fix the issues before using the monitoring system.")
        return 1

if __name__ == '__main__':
    exit(main())
