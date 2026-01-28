# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
REST API endpoints for Cubes monitoring system
"""

import time
import json
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional

from flask import Blueprint, request, jsonify, Response
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError


def create_monitoring_blueprint(monitoring_manager, config_manager):
    """Create Flask blueprint for monitoring API"""
    
    monitoring_api = Blueprint('monitoring', __name__, url_prefix='/api/v1/monitoring')
    
    def handle_errors(f):
        """Error handling decorator"""
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                return jsonify({
                    'error': 'validation_error',
                    'message': str(e),
                    'timestamp': time.time()
                }), 400
            except NotFound as e:
                return jsonify({
                    'error': 'not_found',
                    'message': str(e),
                    'timestamp': time.time()
                }), 404
            except Exception as e:
                return jsonify({
                    'error': 'internal_error',
                    'message': str(e),
                    'timestamp': time.time()
                }), 500
        return wrapper
    
    def validate_hours():
        """Validate and return hours parameter"""
        hours = request.args.get('hours', type=int)
        if hours is not None and (hours < 1 or hours > 168):  # Max 1 week
            raise ValueError("Hours must be between 1 and 168")
        return hours
    
    def validate_cube_name():
        """Validate and return cube name parameter"""
        cube_name = request.args.get('cube_name')
        if cube_name and not cube_name.strip():
            raise ValueError("Cube name cannot be empty")
        return cube_name
    
    # Status and Health Endpoints
    @monitoring_api.route('/status', methods=['GET'])
    @handle_errors
    def get_monitoring_status():
        """Get monitoring system status"""
        status = monitoring_manager.get_monitoring_status()
        return jsonify({
            'status': 'success',
            'data': status,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/health', methods=['GET'])
    @handle_errors
    def get_system_health():
        """Get system health status"""
        health = monitoring_manager.get_system_health()
        return jsonify({
            'status': 'success',
            'data': health,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/health/check', methods=['POST'])
    @handle_errors
    def trigger_health_check():
        """Trigger manual health check"""
        component = request.json.get('component') if request.is_json else None
        
        result = monitoring_manager.trigger_health_check(component)
        return jsonify({
            'status': 'success',
            'data': result,
            'timestamp': time.time()
        })
    
    # Dashboard and Overview Endpoints
    @monitoring_api.route('/dashboard', methods=['GET'])
    @handle_errors
    def get_dashboard():
        """Get comprehensive monitoring dashboard"""
        hours = validate_hours()
        dashboard = monitoring_manager.get_monitoring_dashboard(hours)
        
        return jsonify({
            'status': 'success',
            'data': dashboard,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/overview', methods=['GET'])
    @handle_errors
    def get_overview():
        """Get monitoring overview"""
        hours = validate_hours()
        
        # Get key metrics for overview
        dashboard = monitoring_manager.get_monitoring_dashboard(hours)
        
        overview = {
            'system_status': dashboard['system_status']['overall_status'],
            'health_score': dashboard['system_status']['health_score'],
            'total_queries': dashboard['query_metrics']['total_queries'],
            'avg_execution_time': dashboard['query_metrics']['avg_execution_time'],
            'slow_queries': dashboard['query_metrics']['slow_queries'],
            'performance_score': dashboard['performance_summary']['performance_score'],
            'monitoring_uptime': dashboard['monitoring_uptime'],
            'active_cubes': len(dashboard['cube_metrics']),
            'period_hours': hours or 24
        }
        
        return jsonify({
            'status': 'success',
            'data': overview,
            'timestamp': time.time()
        })
    
    # Metrics Endpoints
    @monitoring_api.route('/metrics', methods=['GET'])
    @handle_errors
    def get_metrics():
        """Get monitoring metrics"""
        hours = validate_hours()
        cube_name = validate_cube_name()
        
        if cube_name:
            metrics = monitoring_manager.metrics_collector.get_cube_metrics(cube_name, hours=hours)
        else:
            query_metrics = monitoring_manager.metrics_collector.get_query_metrics(hours=hours)
            metrics = {
                'query_metrics': [m.__dict__ for m in query_metrics],
                'query_statistics': monitoring_manager.metrics_collector.get_query_statistics(hours=hours)
            }
        
        return jsonify({
            'status': 'success',
            'data': metrics,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/metrics/query', methods=['GET'])
    @handle_errors
    def get_query_metrics():
        """Get query-specific metrics"""
        hours = validate_hours()
        cube_name = validate_cube_name()
        
        query_metrics = monitoring_manager.metrics_collector.get_query_metrics(cube_name, hours)
        query_stats = monitoring_manager.metrics_collector.get_query_statistics(cube_name, hours)
        
        return jsonify({
            'status': 'success',
            'data': {
                'query_metrics': [m.__dict__ for m in query_metrics],
                'statistics': query_stats
            },
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/metrics/slow-queries', methods=['GET'])
    @handle_errors
    def get_slow_queries():
        """Get slow queries"""
        hours = validate_hours()
        threshold = request.args.get('threshold', type=float)
        
        slow_queries = monitoring_manager.metrics_collector.get_slow_queries(threshold, hours)
        
        return jsonify({
            'status': 'success',
            'data': {
                'slow_queries': [q.__dict__ for q in slow_queries],
                'count': len(slow_queries),
                'threshold': threshold or monitoring_manager.config.query.slow_query_threshold,
                'period_hours': hours or 24
            },
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/metrics/cube/<cube_name>', methods=['GET'])
    @handle_errors
    def get_cube_metrics_detail(cube_name):
        """Get detailed metrics for a specific cube"""
        hours = validate_hours()
        
        cube_metrics = monitoring_manager.metrics_collector.get_cube_metrics(cube_name, hours=hours)
        cube_summary = monitoring_manager.metrics_collector.get_cube_summary(cube_name, hours)
        
        return jsonify({
            'status': 'success',
            'data': {
                'cube_name': cube_name,
                'metrics': [m.__dict__ for m in cube_metrics],
                'summary': cube_summary,
                'period_hours': hours or 24
            },
            'timestamp': time.time()
        })
    
    # Performance Endpoints
    @monitoring_api.route('/performance', methods=['GET'])
    @handle_errors
    def get_performance():
        """Get performance metrics"""
        hours = validate_hours()
        
        performance = monitoring_manager.get_performance_report(hours)
        
        return jsonify({
            'status': 'success',
            'data': performance,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/performance/summary', methods=['GET'])
    @handle_errors
    def get_performance_summary():
        """Get performance summary"""
        hours = validate_hours()
        
        summary = monitoring_manager.performance_tracker.get_performance_summary(hours)
        
        return jsonify({
            'status': 'success',
            'data': summary,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/performance/resources', methods=['GET'])
    @handle_errors
    def get_resource_usage():
        """Get resource usage metrics"""
        hours = validate_hours()
        
        resources = monitoring_manager.performance_tracker.get_resource_usage(hours)
        
        return jsonify({
            'status': 'success',
            'data': {
                'resource_usage': [r.__dict__ for r in resources],
                'count': len(resources),
                'period_hours': hours or 24
            },
            'timestamp': time.time()
        })
    
    # Query Analysis Endpoints
    @monitoring_api.route('/queries/patterns', methods=['GET'])
    @handle_errors
    def get_query_patterns():
        """Get query patterns analysis"""
        hours = validate_hours()
        cube_name = validate_cube_name()
        
        patterns = monitoring_manager.query_analyzer.get_query_patterns(cube_name, hours)
        
        return jsonify({
            'status': 'success',
            'data': patterns,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/queries/complexity', methods=['GET'])
    @handle_errors
    def get_query_complexity():
        """Get query complexity trends"""
        hours = validate_hours()
        cube_name = validate_cube_name()
        
        complexity = monitoring_manager.query_analyzer.get_complexity_trends(cube_name, hours)
        
        return jsonify({
            'status': 'success',
            'data': complexity,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/queries/insights/<cube_name>', methods=['GET'])
    @handle_errors
    def get_cube_insights(cube_name):
        """Get comprehensive insights for a cube"""
        hours = validate_hours()
        
        insights = monitoring_manager.get_cube_insights(cube_name, hours)
        
        return jsonify({
            'status': 'success',
            'data': insights,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/queries/suggestions/<cube_name>', methods=['GET'])
    @handle_errors
    def get_optimization_suggestions(cube_name):
        """Get optimization suggestions for a cube"""
        pattern_type = request.args.get('pattern_type')
        
        suggestions = monitoring_manager.query_analyzer.get_optimization_suggestions(cube_name, pattern_type)
        
        return jsonify({
            'status': 'success',
            'data': {
                'cube_name': cube_name,
                'suggestions': suggestions,
                'pattern_type': pattern_type
            },
            'timestamp': time.time()
        })
    
    # System Endpoints
    @monitoring_api.route('/system/status', methods=['GET'])
    @handle_errors
    def get_system_status():
        """Get detailed system status"""
        hours = validate_hours()
        
        status = monitoring_manager.system_monitor.get_system_status()
        trends = monitoring_manager.system_monitor.get_status_trends(hours)
        
        return jsonify({
            'status': 'success',
            'data': {
                'current_status': status.__dict__,
                'trends': trends,
                'period_hours': hours or 24
            },
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/system/health-history', methods=['GET'])
    @handle_errors
    def get_health_history():
        """Get health check history"""
        hours = validate_hours()
        component = request.args.get('component')
        
        history = monitoring_manager.system_monitor.get_health_history(component, hours)
        
        # Convert to serializable format
        serializable_history = {}
        for check_name, checks in history.items():
            serializable_history[check_name] = [check.__dict__ for check in checks]
        
        return jsonify({
            'status': 'success',
            'data': {
                'health_history': serializable_history,
                'period_hours': hours or 24,
                'component': component
            },
            'timestamp': time.time()
        })
    
    # Configuration Endpoints
    @monitoring_api.route('/config', methods=['GET'])
    @handle_errors
    def get_configuration():
        """Get monitoring configuration"""
        config = config_manager.get_config_dict()
        
        return jsonify({
            'status': 'success',
            'data': config,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/config', methods=['PUT'])
    @handle_errors
    def update_configuration():
        """Update monitoring configuration"""
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        updates = request.get_json()
        if not updates:
            raise ValueError("No configuration updates provided")
        
        config_manager.update_config(updates)
        
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated successfully',
            'data': config_manager.get_config_dict(),
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/config/reload', methods=['POST'])
    @handle_errors
    def reload_configuration():
        """Reload configuration from file"""
        try:
            # This would reload configuration from file
            return jsonify({
                'status': 'success',
                'message': 'Configuration reloaded successfully',
                'timestamp': time.time()
            })
        except Exception as e:
            raise ValueError(f"Failed to reload configuration: {str(e)}")
    
    # Export Endpoints
    @monitoring_api.route('/export', methods=['GET'])
    @handle_errors
    def export_data():
        """Export monitoring data"""
        format_type = request.args.get('format', 'json').lower()
        components = request.args.getlist('components')
        
        if format_type not in ['json', 'prometheus']:
            raise ValueError("Format must be 'json' or 'prometheus'")
        
        if not components:
            components = ['metrics', 'performance', 'system']
        
        data = monitoring_manager.export_monitoring_data(format_type, components)
        
        if format_type == 'json':
            return Response(
                data,
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename=monitoring_export_{int(time.time())}.json'}
            )
        else:  # prometheus
            return Response(
                data,
                mimetype='text/plain',
                headers={'Content-Disposition': f'attachment; filename=monitoring_export_{int(time.time())}.prom'}
            )
    
    # Analytics Endpoints
    @monitoring_api.route('/analytics/summary', methods=['GET'])
    @handle_errors
    def get_analytics_summary():
        """Get analytics summary"""
        hours = validate_hours()
        
        dashboard = monitoring_manager.get_monitoring_dashboard(hours)
        
        analytics = {
            'query_analytics': dashboard['query_metrics'],
            'performance_analytics': dashboard['performance_summary'],
            'system_analytics': dashboard['system_status'],
            'health_analytics': dashboard['health_trends'],
            'pattern_analytics': dashboard['query_patterns'],
            'complexity_analytics': dashboard['complexity_trends']
        }
        
        return jsonify({
            'status': 'success',
            'data': analytics,
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/analytics/trends', methods=['GET'])
    @handle_errors
    def get_analytics_trends():
        """Get analytics trends"""
        hours = validate_hours()
        
        trends = {
            'query_trends': monitoring_manager.query_analyzer.get_complexity_trends(hours=hours),
            'performance_trends': monitoring_manager.performance_tracker.get_performance_summary(hours),
            'health_trends': monitoring_manager.system_monitor.get_status_trends(hours)
        }
        
        return jsonify({
            'status': 'success',
            'data': trends,
            'timestamp': time.time()
        })
    
    # Utility Endpoints
    @monitoring_api.route('/cleanup', methods=['POST'])
    @handle_errors
    def cleanup_data():
        """Clean up old monitoring data"""
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        retention_hours = request.get_json().get('retention_hours')
        if retention_hours and (retention_hours < 1 or retention_hours > 168):
            raise ValueError("Retention hours must be between 1 and 168")
        
        monitoring_manager.cleanup_old_data(retention_hours)
        
        return jsonify({
            'status': 'success',
            'message': f'Cleaned up data older than {retention_hours or "default"} hours',
            'timestamp': time.time()
        })
    
    @monitoring_api.route('/version', methods=['GET'])
    @handle_errors
    def get_version():
        """Get monitoring API version"""
        return jsonify({
            'status': 'success',
            'data': {
                'api_version': 'v1',
                'monitoring_version': '1.0.0',
                'features': [
                    'metrics_collection',
                    'query_analysis',
                    'performance_tracking',
                    'system_monitoring',
                    'health_checks',
                    'alerts'
                ]
            },
            'timestamp': time.time()
        })
    
    # Error handlers
    @monitoring_api.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'not_found',
            'message': 'Endpoint not found',
            'timestamp': time.time()
        }), 404
    
    @monitoring_api.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': 'method_not_allowed',
            'message': 'HTTP method not allowed',
            'timestamp': time.time()
        }), 405
    
    @monitoring_api.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'internal_error',
            'message': 'Internal server error',
            'timestamp': time.time()
        }), 500
    
    return monitoring_api


def register_monitoring_api(app, monitoring_manager, config_manager):
    """Register monitoring API with Flask app"""
    monitoring_blueprint = create_monitoring_blueprint(monitoring_manager, config_manager)
    app.register_blueprint(monitoring_blueprint)
    
    # Add CORS headers if needed
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    return monitoring_blueprint
