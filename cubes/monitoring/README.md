# Cubes OLAP Monitoring System

Enterprise-grade monitoring and metrics system for Cubes OLAP framework that provides real-time observability, performance tracking, and operational intelligence.

## Features

### 🔍 **Comprehensive OLAP Monitoring**
- **Query Performance**: Execution time, complexity analysis, pattern recognition
- **Cube Operations**: Aggregation, drilldown, facts retrieval tracking
- **Resource Usage**: CPU, memory, disk, network monitoring
- **Database Metrics**: Connection pools, query optimization tracking
- **System Health**: Component health checks with configurable thresholds

### 📊 **Real-Time Analytics**
- **Query Pattern Analysis**: Identify slow queries and optimization opportunities
- **Performance Trends**: Historical analysis and anomaly detection
- **Cube Insights**: Per-cube performance analytics and recommendations
- **Health Monitoring**: System-wide health status with alerting

### 🚨 **Intelligent Alerting**
- **Threshold-Based Alerts**: Configurable thresholds for all metrics
- **Anomaly Detection**: Statistical anomaly detection for performance metrics
- **Health Monitoring**: Component health checks and status monitoring
- **Query Alerts**: Slow query detection and optimization suggestions

### 📈 **REST API**
- **Comprehensive Endpoints**: Full REST API for all monitoring data
- **Real-Time Data**: Live metrics and status information
- **Historical Data**: Time-series data with flexible querying
- **Export Capabilities**: JSON and Prometheus format exports

### ⚙️ **Advanced Configuration**
- **YAML Configuration**: Comprehensive configuration management
- **Environment Variables**: Override configuration via environment
- **Dynamic Updates**: Runtime configuration changes
- **Component Control**: Enable/disable individual monitoring components

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Cubes Monitoring Manager                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ OLAP Metrics    │ │ Query Analyzer  │ │ Performance │ │
│  │ Collector       │ │                 │ │ Tracker     │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ System Monitor  │ │ Config Manager  │ │ REST API     │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Cubes Core     │
                    │   OLAP Server    │
                    │   Flask App      │
                    └───────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- Cubes OLAP framework
- Redis server (optional, for distributed monitoring)

### Dependencies
```bash
# Install monitoring dependencies
pip install -r requirements-monitoring.txt
```

### Configuration
Create a `monitoring.yaml` configuration file:

```yaml
# Enable/disable monitoring system
enabled: true

# Metrics collection configuration
metrics:
  enabled: true
  collection_interval: 30  # seconds
  retention_hours: 24
  export_formats: ['json', 'prometheus']

# Query monitoring configuration
query:
  enabled: true
  track_slow_queries: true
  slow_query_threshold: 1.0  # seconds
  analyze_query_complexity: true

# Performance tracking configuration
performance:
  enabled: true
  tracking_interval: 60
  anomaly_detection: true
  anomaly_threshold_multiplier: 2.0

# System health monitoring configuration
system:
  enabled: true
  check_interval: 60
  cpu_warning_threshold: 70.0
  cpu_critical_threshold: 90.0
  memory_warning_threshold: 75.0
  memory_critical_threshold: 90.0

# Alert configuration
alerts:
  enabled: true
  slow_query_alert: true
  performance_alert: true
  system_alert: true
```

## Usage

### Basic Integration

```python
from cubes.monitoring_integration import initialize_monitoring, record_query

# Initialize monitoring
monitoring = initialize_monitoring(workspace, 'monitoring.yaml')

# Record query execution (automatically done by decorators)
record_query(
    cube_name='sales',
    query_type='aggregate',
    cuts=['date:2023', 'region:US'],
    drills=[],
    dimensions=['date', 'region', 'product'],
    measures=['amount', 'quantity'],
    execution_time=0.5,
    result_size=1000
)
```

### Using Decorators

```python
from cubes.server.monitoring_patch import monitor_aggregation, monitor_drilldown

@monitor_aggregation
def aggregate_cube(cube, cuts):
    # Your aggregation logic
    return result

@monitor_drilldown
def drilldown_cube(cube, dimension, cuts):
    # Your drilldown logic
    return result
```

### Manual Monitoring

```python
from cubes.monitoring_integration import QueryContext

# Use context manager for manual monitoring
with QueryContext('sales', cuts=['date:2023']):
    # Your query logic here
    result = perform_complex_query()
```

## API Endpoints

### System Status
```bash
# Get monitoring system status
curl http://localhost:5000/api/v1/monitoring/status

# Get system health
curl http://localhost:5000/api/v1/monitoring/health

# Trigger health check
curl -X POST http://localhost:5000/api/v1/monitoring/health/check
```

### Dashboard Data
```bash
# Get comprehensive dashboard
curl http://localhost:5000/api/v1/monitoring/dashboard

# Get overview
curl http://localhost:5000/api/v1/monitoring/overview
```

### Query Metrics
```bash
# Get query metrics
curl http://localhost:5000/api/v1/monitoring/metrics/query

# Get slow queries
curl http://localhost:5000/api/v1/monitoring/metrics/slow-queries

# Get cube-specific metrics
curl http://localhost:5000/api/v1/monitoring/metrics/cube/sales
```

### Performance Data
```bash
# Get performance report
curl http://localhost:5000/api/v1/monitoring/performance

# Get resource usage
curl http://localhost:5000/api/v1/monitoring/performance/resources
```

### Query Analysis
```bash
# Get query patterns
curl http://localhost:5000/api/v1/monitoring/queries/patterns

# Get query complexity trends
curl http://localhost:5000/api/v1/monitoring/queries/complexity

# Get cube insights
curl http://localhost:5000/api/v1/monitoring/queries/insights/sales
```

### Configuration
```bash
# Get configuration
curl http://localhost:5000/api/v1/monitoring/config

# Update configuration
curl -X PUT http://localhost:5000/api/v1/monitoring/config \
  -H "Content-Type: application/json" \
  -d '{"metrics": {"collection_interval": 60}}'
```

### Export Data
```bash
# Export as JSON
curl "http://localhost:5000/api/v1/monitoring/export?format=json" -o monitoring.json

# Export as Prometheus
curl "http://localhost:5000/api/v1/monitoring/export?format=prometheus" -o monitoring.prom
```

## Configuration

### Environment Variables
Override configuration using environment variables:

```bash
export CUBES_MONITORING_ENABLED=true
export CUBES_MONITORING_METRICS_INTERVAL=60
export CUBES_MONITORING_QUERY_SLOW_THRESHOLD=2.0
export CUBES_MONITORING_SYSTEM_CPU_WARNING=75.0
export CUBES_MONITORING_SYSTEM_MEMORY_CRITICAL=95.0
```

### Component Configuration
Enable/disable specific components:

```yaml
monitoring:
  metrics:
    enabled: true
  query:
    enabled: true
  performance:
    enabled: true
  system:
    enabled: true
  alerts:
    enabled: true
```

### Threshold Configuration
Configure alert thresholds:

```yaml
monitoring:
  query:
    slow_query_threshold: 2.0
  system:
    cpu_warning_threshold: 75.0
    cpu_critical_threshold: 90.0
    memory_warning_threshold: 80.0
    memory_critical_threshold: 95.0
```

## Advanced Features

### Query Pattern Analysis
The system automatically analyzes query patterns to identify:
- **Simple Queries**: Basic aggregations and filters
- **Complex Analytical**: Multi-dimensional analysis
- **Drilldown Analysis**: Hierarchical exploration
- **Performance Issues**: Slow queries and optimization opportunities

### Anomaly Detection
Statistical anomaly detection identifies:
- **Performance Anomalies**: Unusual execution times
- **Resource Anomalies**: Abnormal resource usage
- **Query Anomalies**: Unusual query patterns
- **System Anomalies**: Component health issues

### Optimization Suggestions
The system provides intelligent suggestions:
- **Query Optimization**: Index suggestions, query restructuring
- **Resource Management**: Memory and CPU optimization
- **Configuration Tuning**: Performance parameter adjustments
- **Architecture Improvements**: Scaling and caching recommendations

## Integration Examples

### Flask Application Integration
```python
from flask import Flask
from cubes.monitoring_integration import initialize_monitoring
from cubes.server.monitoring_patch import register_monitoring_api

app = Flask(__name__)

# Initialize monitoring
monitoring = initialize_monitoring()

# Register monitoring API
register_monitoring_api(app, monitoring.monitoring_manager, monitoring.config_manager)

@app.route('/api/cubes/aggregate')
def aggregate():
    with QueryContext('sales', cuts=['date:2023']):
        # Your aggregation logic
        return result
```

### Workspace Integration
```python
from cubes import Workspace
from cubes.monitoring_integration import initialize_monitoring

# Create workspace
workspace = Workspace()

# Initialize monitoring with workspace
monitoring = initialize_monitoring(workspace)

# Use workspace with automatic monitoring
browser = workspace.browser('sales')
result = browser.aggregate(cuts=['date:2023'])
```

## Performance Considerations

### Resource Usage
- **Memory**: ~50-100MB base + ~1MB per 1000 metrics
- **CPU**: ~1-2% during normal operation
- **Disk**: ~10MB/day for metrics storage

### Optimization Tips
- Increase collection intervals for less resource usage
- Reduce retention hours to save disk space
- Disable unused components
- Use export to external systems for long-term storage

## Troubleshooting

### Common Issues

**Monitoring not starting**
- Check if `enabled: true` in configuration
- Verify dependencies are installed
- Check logs for initialization errors

**Missing metrics**
- Ensure workspace is properly initialized
- Check collection intervals
- Verify component permissions

**High resource usage**
- Increase collection intervals
- Reduce retention hours
- Disable unused components

**API not responding**
- Check if Flask app is running
- Verify monitoring is enabled
- Check network connectivity

### Debug Mode
Enable debug logging:
```yaml
monitoring:
  log_level: DEBUG
```

Or via environment:
```bash
export CUBES_MONITORING_LOG_LEVEL=DEBUG
```

## Security

### Access Control
- API endpoints are accessible via Flask app port
- Configure firewall rules as needed
- Use HTTPS in production environments

### Data Sensitivity
- Metrics may contain query patterns and performance data
- Secure export files appropriately
- Consider data retention policies

## Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/Dhananjay9075/cubes.git

# Install dependencies
pip install -r requirements-monitoring.txt

# Run tests
python -m pytest tests/test_monitoring.py -v
```

### Adding New Metrics
1. Add metric collection to appropriate component
2. Update configuration schema if needed
3. Add API endpoints for new metrics
4. Write tests for new functionality
5. Update documentation

## License

This monitoring system is part of Cubes OLAP Framework and is licensed under MIT.

## Support

- **Issues**: Report bugs via GitHub issues
- **Documentation**: See Cubes documentation
- **Community**: Join our discussion forums
- **Email**: support@cubes.org
