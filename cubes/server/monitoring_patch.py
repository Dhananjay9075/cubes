# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Monitoring integration patch for Cubes server blueprint
"""

import time
import functools
from typing import Dict, List, Any, Optional

try:
    from ..monitoring_integration import get_monitoring, record_query, record_aggregation, record_drilldown, record_facts
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def monitor_query_execution(func):
    """Decorator to monitor query execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not MONITORING_AVAILABLE:
            return func(*args, **kwargs)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract query information from result or context
            cube_name = getattr(g, 'current_cube', 'unknown')
            query_type = func.__name__.replace('aggregate_', '').replace('drilldown_', '').replace('facts_', '')
            
            # Try to extract query parameters
            cuts = getattr(g, 'current_cuts', [])
            drills = getattr(g, 'current_drills', [])
            dimensions = getattr(g, 'current_dimensions', [])
            measures = getattr(g, 'current_measures', [])
            
            # Estimate result size
            result_size = len(str(result)) if result else 0
            
            # Record the query
            record_query(
                cube_name=cube_name,
                query_type=query_type,
                cuts=cuts,
                drills=drills,
                dimensions=dimensions,
                measures=measures,
                execution_time=execution_time,
                result_size=result_size
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failed query
            record_query(
                cube_name=getattr(g, 'current_cube', 'unknown'),
                query_type=func.__name__,
                cuts=[],
                drills=[],
                dimensions=[],
                measures=[],
                execution_time=execution_time,
                result_size=0
            )
            
            raise
    
    return wrapper


def monitor_aggregation(func):
    """Decorator to monitor aggregation operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not MONITORING_AVAILABLE:
            return func(*args, **kwargs)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract information
            cube_name = getattr(g, 'current_cube', 'unknown')
            cuts = getattr(g, 'current_cuts', [])
            
            # Estimate cell count from result
            cell_count = len(result) if hasattr(result, '__len__') else 1
            
            record_aggregation(cube_name, cuts, execution_time, cell_count)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            record_aggregation(getattr(g, 'current_cube', 'unknown'), [], execution_time, 0)
            raise
    
    return wrapper


def monitor_drilldown(func):
    """Decorator to monitor drilldown operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not MONITORING_AVAILABLE:
            return func(*args, **kwargs)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract information
            cube_name = getattr(g, 'current_cube', 'unknown')
            dimension = kwargs.get('dimension') or getattr(g, 'current_dimension', 'unknown')
            
            # Estimate result count
            result_count = len(result) if hasattr(result, '__len__') else 1
            
            record_drilldown(cube_name, dimension, execution_time, result_count)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            record_drilldown(getattr(g, 'current_cube', 'unknown'), 'unknown', execution_time, 0)
            raise
    
    return wrapper


def monitor_facts_retrieval(func):
    """Decorator to monitor facts retrieval operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not MONITORING_AVAILABLE:
            return func(*args, **kwargs)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract information
            cube_name = getattr(g, 'current_cube', 'unknown')
            cuts = getattr(g, 'current_cuts', [])
            
            # Estimate result count
            result_count = len(result) if hasattr(result, '__len__') else 1
            
            record_facts(cube_name, cuts, execution_time, result_count)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            record_facts(getattr(g, 'current_cube', 'unknown'), [], execution_time, 0)
            raise
    
    return wrapper


def extract_query_context(cube_name: str, cuts: List[str] = None, drills: List[str] = None,
                        dimensions: List[str] = None, measures: List[str] = None):
    """Extract and store query context in Flask global context"""
    if not MONITORING_AVAILABLE:
        return
    
    g.current_cube = cube_name
    g.current_cuts = cuts or []
    g.current_drills = drills or []
    g.current_dimensions = dimensions or []
    g.current_measures = measures or []


def get_monitoring_data():
    """Get monitoring data for API responses"""
    if not MONITORING_AVAILABLE:
        return {}
    
    monitoring = get_monitoring()
    if not monitoring:
        return {}
    
    try:
        return monitoring.get_monitoring_dashboard(hours=1)
    except Exception:
        return {}


def get_cube_monitoring_insights(cube_name: str):
    """Get monitoring insights for a specific cube"""
    if not MONITORING_AVAILABLE:
        return {}
    
    monitoring = get_monitoring()
    if not monitoring:
        return {}
    
    try:
        return monitoring.get_cube_insights(cube_name, hours=24)
    except Exception:
        return {}


def get_system_health_status():
    """Get system health status"""
    if not MONITORING_AVAILABLE:
        return {}
    
    monitoring = get_monitoring()
    if not monitoring:
        return {}
    
    try:
        return monitoring.get_system_health()
    except Exception:
        return {}


# Context manager for query monitoring
class QueryContext:
    """Context manager for query monitoring"""
    
    def __init__(self, cube_name: str, cuts: List[str] = None, drills: List[str] = None,
                 dimensions: List[str] = None, measures: List[str] = None):
        self.cube_name = cube_name
        self.cuts = cuts or []
        self.drills = drills or []
        self.dimensions = dimensions or []
        self.measures = measures or []
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        extract_query_context(
            self.cube_name, self.cuts, self.drills, 
            self.dimensions, self.measures
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not MONITORING_AVAILABLE or not self.start_time:
            return
        
        execution_time = time.time() - self.start_time
        result_size = 0  # Would be set by the calling code
        
        record_query(
            cube_name=self.cube_name,
            query_type='manual',
            cuts=self.cuts,
            drills=self.drills,
            dimensions=self.dimensions,
            measures=self.measures,
            execution_time=execution_time,
            result_size=result_size
        )
