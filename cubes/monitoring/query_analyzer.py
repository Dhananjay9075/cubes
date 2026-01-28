# SPDX-FileCopyrightText: 2024 Cubes OLAP Framework
# SPDX-License-Identifier: MIT

"""
Query analyzer for Cubes OLAP framework
Analyzes query patterns, complexity, and performance
"""

import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging


@dataclass
class QueryPattern:
    """Query pattern analysis result"""
    pattern_id: str
    pattern_type: str  # simple, complex, analytical, etc.
    complexity_score: float
    dimensions_used: List[str]
    measures_used: List[str]
    cuts_count: int
    drills_count: int
    estimated_cost: float
    optimization_suggestions: List[str]


@dataclass
class QueryComplexity:
    """Query complexity analysis"""
    score: float
    factors: Dict[str, float]
    level: str  # low, medium, high, very_high


class QueryAnalyzer:
    """Analyzes OLAP queries for performance and patterns"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Query pattern storage
        self.query_patterns = defaultdict(list)
        self.complexity_history = defaultdict(list)
        
        # Analysis caches
        self._pattern_cache = {}
        self._complexity_cache = {}
        
        # Complexity weights
        self.complexity_weights = {
            'dimensions_count': 0.2,
            'measures_count': 0.1,
            'cuts_count': 0.3,
            'drills_count': 0.2,
            'result_size_factor': 0.1,
            'time_factor': 0.1
        }
    
    def analyze_query(self, cube_name: str, query_type: str, cuts: List[str], 
                     drills: List[str], dimensions: List[str], 
                     measures: List[str], execution_time: float,
                     result_size: int = 0) -> QueryPattern:
        """Analyze a query and return pattern information"""
        
        # Generate query pattern ID
        pattern_id = self._generate_pattern_id(query_type, cuts, drills, dimensions, measures)
        
        # Check cache first
        cache_key = f"{cube_name}:{pattern_id}"
        if cache_key in self._pattern_cache:
            base_pattern = self._pattern_cache[cache_key]
        else:
            base_pattern = self._analyze_pattern_structure(
                query_type, cuts, drills, dimensions, measures
            )
            self._pattern_cache[cache_key] = base_pattern
        
        # Update with execution data
        pattern = QueryPattern(
            pattern_id=pattern_id,
            pattern_type=base_pattern.pattern_type,
            complexity_score=base_pattern.complexity_score,
            dimensions_used=dimensions,
            measures_used=measures,
            cuts_count=len(cuts),
            drills_count=len(drills),
            estimated_cost=self._estimate_query_cost(cuts, drills, dimensions, measures),
            optimization_suggestions=base_pattern.optimization_suggestions
        )
        
        # Store pattern
        self.query_patterns[cube_name].append({
            'pattern': pattern,
            'timestamp': time.time(),
            'execution_time': execution_time,
            'result_size': result_size
        })
        
        return pattern
    
    def analyze_complexity(self, cube_name: str, query_type: str, cuts: List[str],
                          drills: List[str], dimensions: List[str], 
                          measures: List[str], execution_time: float,
                          result_size: int = 0) -> QueryComplexity:
        """Analyze query complexity"""
        
        # Calculate complexity factors
        factors = {
            'dimensions_count': min(len(dimensions) / 10.0, 1.0),
            'measures_count': min(len(measures) / 5.0, 1.0),
            'cuts_count': min(len(cuts) / 20.0, 1.0),
            'drills_count': min(len(drills) / 10.0, 1.0),
            'result_size_factor': min(result_size / 100000.0, 1.0),
            'time_factor': min(execution_time / 10.0, 1.0)
        }
        
        # Calculate weighted score
        score = sum(
            factors[factor] * weight 
            for factor, weight in self.complexity_weights.items()
        )
        
        # Determine complexity level
        if score < 0.3:
            level = 'low'
        elif score < 0.6:
            level = 'medium'
        elif score < 0.8:
            level = 'high'
        else:
            level = 'very_high'
        
        complexity = QueryComplexity(
            score=score,
            factors=factors,
            level=level
        )
        
        # Store complexity history
        self.complexity_history[cube_name].append({
            'complexity': complexity,
            'timestamp': time.time(),
            'execution_time': execution_time
        })
        
        return complexity
    
    def get_query_patterns(self, cube_name: Optional[str] = None, 
                          hours: Optional[int] = None) -> Dict[str, Any]:
        """Get analyzed query patterns"""
        patterns = {}
        
        for cn, pattern_list in self.query_patterns.items():
            if cube_name and cn != cube_name:
                continue
            
            # Filter by time
            if hours:
                cutoff_time = time.time() - (hours * 3600)
                pattern_list = [p for p in pattern_list if p['timestamp'] >= cutoff_time]
            
            # Analyze patterns
            pattern_counter = Counter()
            complexity_scores = []
            execution_times = []
            
            for p in pattern_list:
                pattern_counter[p['pattern'].pattern_type] += 1
                complexity_scores.append(p['pattern'].complexity_score)
                execution_times.append(p['execution_time'])
            
            patterns[cn] = {
                'total_queries': len(pattern_list),
                'pattern_distribution': dict(pattern_counter),
                'avg_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
                'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                'most_common_pattern': pattern_counter.most_common(1)[0][0] if pattern_counter else None
            }
        
        return patterns
    
    def get_complexity_trends(self, cube_name: Optional[str] = None,
                             hours: Optional[int] = None) -> Dict[str, Any]:
        """Get complexity trends over time"""
        trends = {}
        
        for cn, complexity_list in self.complexity_history.items():
            if cube_name and cn != cube_name:
                continue
            
            # Filter by time
            if hours:
                cutoff_time = time.time() - (hours * 3600)
                complexity_list = [c for c in complexity_list if c['timestamp'] >= cutoff_time]
            
            if not complexity_list:
                continue
            
            # Calculate trends
            scores = [c['complexity'].score for c in complexity_list]
            levels = [c['complexity'].level for c in complexity_list]
            
            trends[cn] = {
                'total_analyzed': len(complexity_list),
                'avg_complexity': sum(scores) / len(scores),
                'complexity_trend': self._calculate_trend(scores),
                'level_distribution': dict(Counter(levels)),
                'latest_complexity': scores[-1] if scores else 0,
                'complexity_change': scores[-1] - scores[0] if len(scores) > 1 else 0
            }
        
        return trends
    
    def get_optimization_suggestions(self, cube_name: str, 
                                   pattern_type: Optional[str] = None) -> List[str]:
        """Get optimization suggestions for queries"""
        suggestions = []
        
        pattern_list = self.query_patterns.get(cube_name, [])
        if not pattern_list:
            return suggestions
        
        # Filter by pattern type if specified
        if pattern_type:
            pattern_list = [p for p in pattern_list if p['pattern'].pattern_type == pattern_type]
        
        # Analyze common issues
        slow_queries = [p for p in pattern_list if p['execution_time'] > self.config.query.slow_query_threshold]
        complex_queries = [p for p in pattern_list if p['pattern'].complexity_score > 0.7]
        
        # Generate suggestions
        if slow_queries:
            suggestions.append(
                f"Found {len(slow_queries)} slow queries. Consider optimizing cube structure or adding indexes."
            )
        
        if complex_queries:
            suggestions.append(
                f"Found {len(complex_queries)} complex queries. Consider simplifying queries or pre-aggregating data."
            )
        
        # Pattern-specific suggestions
        pattern_types = [p['pattern'].pattern_type for p in pattern_list]
        most_common = Counter(pattern_types).most_common(1)
        if most_common:
            common_type = most_common[0][0]
            if common_type == 'complex_analytical':
                suggestions.append(
                    "Many complex analytical queries detected. Consider materialized views or summary tables."
                )
            elif common_type == 'heavy_aggregation':
                suggestions.append(
                    "Heavy aggregation queries detected. Consider pre-computing aggregations."
                )
        
        return suggestions
    
    def _generate_pattern_id(self, query_type: str, cuts: List[str], 
                           drills: List[str], dimensions: List[str], 
                           measures: List[str]) -> str:
        """Generate unique pattern ID"""
        pattern_data = {
            'type': query_type,
            'cuts': sorted(cuts),
            'drills': sorted(drills),
            'dimensions': sorted(dimensions),
            'measures': sorted(measures)
        }
        
        pattern_str = str(pattern_data)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
    
    def _analyze_pattern_structure(self, query_type: str, cuts: List[str],
                                 drills: List[str], dimensions: List[str],
                                 measures: List[str]) -> QueryPattern:
        """Analyze query pattern structure"""
        
        # Determine pattern type
        cuts_count = len(cuts)
        drills_count = len(drills)
        dimensions_count = len(dimensions)
        measures_count = len(measures)
        
        if cuts_count == 0 and drills_count == 0:
            pattern_type = 'simple'
        elif cuts_count > 10 or drills_count > 5:
            pattern_type = 'complex_analytical'
        elif cuts_count > 5:
            pattern_type = 'moderately_complex'
        elif drills_count > 0:
            pattern_type = 'drilldown_analysis'
        elif measures_count > 3:
            pattern_type = 'multi_measure_analysis'
        else:
            pattern_type = 'standard'
        
        # Calculate complexity score
        complexity_score = (
            min(cuts_count / 20.0, 1.0) * 0.4 +
            min(drills_count / 10.0, 1.0) * 0.3 +
            min(dimensions_count / 10.0, 1.0) * 0.2 +
            min(measures_count / 5.0, 1.0) * 0.1
        )
        
        # Generate optimization suggestions
        suggestions = []
        if cuts_count > 15:
            suggestions.append("Consider reducing number of cuts for better performance")
        if drills_count > 5:
            suggestions.append("Multiple drilldowns detected - consider batch processing")
        if dimensions_count > 8:
            suggestions.append("Many dimensions used - consider dimension optimization")
        if measures_count > 5:
            suggestions.append("Many measures - consider measure grouping")
        
        return QueryPattern(
            pattern_id="",  # Will be set by caller
            pattern_type=pattern_type,
            complexity_score=complexity_score,
            dimensions_used=dimensions,
            measures_used=measures,
            cuts_count=cuts_count,
            drills_count=drills_count,
            estimated_cost=0,  # Will be calculated by caller
            optimization_suggestions=suggestions
        )
    
    def _estimate_query_cost(self, cuts: List[str], drills: List[str],
                           dimensions: List[str], measures: List[str]) -> float:
        """Estimate query execution cost"""
        # Simple cost estimation based on query components
        base_cost = 1.0
        cut_cost = len(cuts) * 0.1
        drill_cost = len(drills) * 0.2
        dimension_cost = len(dimensions) * 0.05
        measure_cost = len(measures) * 0.02
        
        return base_cost + cut_cost + drill_cost + dimension_cost + measure_cost
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        # Determine trend
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def analyze_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Analyze SQL query for complexity and optimization opportunities"""
        if not sql_query:
            return {}
        
        analysis = {
            'query_length': len(sql_query),
            'joins_count': len(re.findall(r'\bJOIN\b', sql_query, re.IGNORECASE)),
            'where_conditions': len(re.findall(r'\bWHERE\b', sql_query, re.IGNORECASE)),
            'group_by_count': len(re.findall(r'\bGROUP BY\b', sql_query, re.IGNORECASE)),
            'order_by_count': len(re.findall(r'\bORDER BY\b', sql_query, re.IGNORECASE)),
            'subqueries': len(re.findall(r'\bSELECT\b.*\bFROM\b.*\bWHERE\b', sql_query, re.IGNORECASE | re.DOTALL)),
        }
        
        # Calculate SQL complexity
        sql_complexity = (
            min(analysis['joins_count'] / 5.0, 1.0) * 0.3 +
            min(analysis['where_conditions'] / 3.0, 1.0) * 0.2 +
            min(analysis['group_by_count'] / 2.0, 1.0) * 0.2 +
            min(analysis['subqueries'] / 3.0, 1.0) * 0.3
        )
        
        analysis['sql_complexity'] = sql_complexity
        
        # Generate SQL optimization suggestions
        suggestions = []
        if analysis['joins_count'] > 3:
            suggestions.append("Consider reducing number of joins or optimizing join conditions")
        if analysis['subqueries'] > 2:
            suggestions.append("Consider using CTEs or simplifying subqueries")
        if analysis['where_conditions'] > 2:
            suggestions.append("Consider optimizing WHERE conditions with proper indexes")
        
        analysis['optimization_suggestions'] = suggestions
        
        return analysis
    
    def get_performance_insights(self, cube_name: str) -> Dict[str, Any]:
        """Get comprehensive performance insights for a cube"""
        patterns = self.get_query_patterns(cube_name)
        trends = self.get_complexity_trends(cube_name)
        suggestions = self.get_optimization_suggestions(cube_name)
        
        insights = {
            'cube_name': cube_name,
            'query_patterns': patterns.get(cube_name, {}),
            'complexity_trends': trends.get(cube_name, {}),
            'optimization_suggestions': suggestions,
            'performance_score': self._calculate_performance_score(cube_name)
        }
        
        return insights
    
    def _calculate_performance_score(self, cube_name: str) -> float:
        """Calculate overall performance score for a cube"""
        pattern_list = self.query_patterns.get(cube_name, [])
        if not pattern_list:
            return 1.0  # Perfect score for no queries
        
        # Factors affecting performance score
        avg_execution_time = sum(p['execution_time'] for p in pattern_list) / len(pattern_list)
        avg_complexity = sum(p['pattern'].complexity_score for p in pattern_list) / len(pattern_list)
        slow_query_ratio = sum(1 for p in pattern_list if p['execution_time'] > self.config.query.slow_query_threshold) / len(pattern_list)
        
        # Calculate score (0-100)
        time_score = max(0, 100 - (avg_execution_time * 10))  # Lower time is better
        complexity_score = max(0, 100 - (avg_complexity * 50))  # Lower complexity is better
        slow_query_penalty = slow_query_ratio * 30  # Penalty for slow queries
        
        final_score = (time_score * 0.4 + complexity_score * 0.4 - slow_query_penalty)
        return max(0, min(100, final_score))
