"""
SQL Transpiler - Translate SQL between dialects.

This module transforms AST nodes to adapt syntax and functions
from one SQL dialect to another.
"""

from typing import Dict, Callable, Optional, Any, List
from dataclasses import dataclass, field
from copy import deepcopy

from .dialects import SQLDialect
from .ast_nodes import (
    ASTNode, SelectStatement, FunctionCall, Identifier, BinaryOp,
    Literal, CastExpression, ArrayExpression, IntervalExpression,
    TryExpression, LambdaExpression, UnaryOp, InExpression,
    CaseExpression, WindowFunction, TableRef, ColumnRef, IfExpression
)
from .parser import SQLParser
from .sql_generator import SQLGenerator


@dataclass
class TranspilationResult:
    """Result of a SQL transpilation."""
    success: bool
    sql: str
    source_dialect: SQLDialect
    target_dialect: SQLDialect
    warnings: List[str] = field(default_factory=list)
    unsupported_features: List[str] = field(default_factory=list)


class DialectTransformer:
    """Base class for dialect-specific transformations."""
    
    def __init__(self, source: SQLDialect, target: SQLDialect):
        self.source = source
        self.target = target
        self.warnings: List[str] = []
        self.unsupported: List[str] = []
    
    def transform(self, node: ASTNode) -> ASTNode:
        """Transform an AST node for the target dialect."""
        node = deepcopy(node)
        return self._transform_node(node)
    
    def _transform_node(self, node: Any) -> Any:
        """Recursively transform a node and its children."""
        if node is None:
            return None
        
        if isinstance(node, list):
            return [self._transform_node(item) for item in node]
        
        if isinstance(node, dict):
            return {k: self._transform_node(v) for k, v in node.items()}
        
        if not isinstance(node, ASTNode):
            return node
        
        # Transform children first (bottom-up)
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, (ASTNode, list)):
                    setattr(node, attr_name, self._transform_node(attr_value))
            except (AttributeError, TypeError):
                pass
        
        # Apply specific transformations
        return self._apply_transformation(node)
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        """Apply dialect-specific transformation to a node."""
        # Override in subclasses
        return node


class PrestoToPostgreSQLTransformer(DialectTransformer):
    """Transform Presto/Athena SQL to PostgreSQL."""
    
    # Function name mappings
    FUNCTION_MAP = {
        'SUBSTR': 'SUBSTRING',
        'STRPOS': 'POSITION',
        'LISTAGG': 'STRING_AGG',
        'APPROX_DISTINCT': 'COUNT',  # Approximation
        'APPROX_PERCENTILE': 'PERCENTILE_CONT',
        'ARBITRARY': 'MIN',  # No direct equivalent
        'ARRAY_JOIN': 'ARRAY_TO_STRING',
        'CARDINALITY': 'ARRAY_LENGTH',
        'ELEMENT_AT': None,  # Needs special handling -> arr[idx]
        'TRANSFORM': None,  # No equivalent
        'FILTER': None,  # Array filter, no direct equivalent
        'REDUCE': None,  # No equivalent
        'DATE_TRUNC': 'DATE_TRUNC',  # Same but argument order differs
        'DATE_ADD': None,  # -> date + interval
        'DATE_DIFF': None,  # -> date - date
        'TO_UNIXTIME': 'EXTRACT',  # EPOCH FROM
        'FROM_UNIXTIME': 'TO_TIMESTAMP',
        'JSON_EXTRACT': 'JSON_EXTRACT_PATH_TEXT',
        'JSON_EXTRACT_SCALAR': 'JSON_EXTRACT_PATH_TEXT',
        'REGEXP_LIKE': None,  # -> ~ operator
        'REGEXP_EXTRACT': 'SUBSTRING',  # With regex
        'SPLIT': 'STRING_TO_ARRAY',
        'SEQUENCE': 'GENERATE_SERIES',
        'TRY': None,  # No equivalent, needs CASE or COALESCE
        'IF': 'CASE',  # Needs transformation
        'NVL': 'COALESCE',
        'NVL2': None,  # CASE WHEN x IS NOT NULL THEN y ELSE z
        'TYPEOF': 'PG_TYPEOF',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, IfExpression):
            return self._transform_if_expression(node)
        elif isinstance(node, TryExpression):
            return self._transform_try(node)
        elif isinstance(node, LambdaExpression):
            return self._transform_lambda(node)
        elif isinstance(node, IntervalExpression):
            return self._transform_interval(node)
        return node
    
    def _transform_if_expression(self, node: IfExpression) -> CaseExpression:
        """Transform IF expression to CASE WHEN ... THEN ... ELSE ... END."""
        else_value = node.else_expr if node.else_expr else Literal(None, 'NULL')
        return CaseExpression(
            operand=None,
            when_clauses=[(node.condition, node.then_expr)],
            else_clause=else_value
        )
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        # Special transformations that require structural changes
        if func_name == 'IF':
            # IF(cond, then, else) -> CASE WHEN cond THEN then ELSE else END
            return self._if_to_case(node)
        elif func_name == 'REGEXP_LIKE':
            # REGEXP_LIKE(str, pattern) -> str ~ pattern
            return self._regexp_like_to_operator(node)
        elif func_name == 'ELEMENT_AT':
            # ELEMENT_AT(arr, idx) -> arr[idx]
            return self._element_at_to_subscript(node)
        elif func_name in ('TRANSFORM', 'FILTER', 'REDUCE'):
            self.unsupported.append(f"Lambda function {func_name} has no PostgreSQL equivalent")
            return node
        elif func_name == 'TRY':
            self.unsupported.append("TRY() has no PostgreSQL equivalent - consider using COALESCE or exception handling")
            return node
        
        # Direct function mapping (simple name replacement)
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name:
                node.name = new_name
        
        # DATE_ADD(unit, value, date) -> date + interval 'value unit'
        if func_name == 'DATE_ADD' and len(node.args) >= 3:
            return self._date_add_to_interval(node)
        
        # DATE_DIFF(unit, start, end) -> EXTRACT(EPOCH FROM end - start) / factor
        if func_name == 'DATE_DIFF' and len(node.args) >= 3:
            return self._date_diff_to_extract(node)
        
        # CARDINALITY(arr) -> ARRAY_LENGTH(arr, 1)
        if func_name == 'CARDINALITY':
            node.args.append(Literal(1, 'INTEGER'))
        
        return node
    
    def _if_to_case(self, node: FunctionCall) -> CaseExpression:
        """Transform IF(cond, then, else) to CASE WHEN cond THEN then ELSE else END."""
        if len(node.args) < 2:
            return node
        condition = node.args[0]
        then_value = node.args[1]
        else_value = node.args[2] if len(node.args) > 2 else Literal(None, 'NULL')
        
        return CaseExpression(
            operand=None,
            when_clauses=[(condition, then_value)],
            else_clause=else_value
        )
    
    def _regexp_like_to_operator(self, node: FunctionCall) -> BinaryOp:
        """Transform REGEXP_LIKE(str, pattern) to str ~ pattern."""
        if len(node.args) < 2:
            return node
        return BinaryOp(
            left=node.args[0],
            operator='~',
            right=node.args[1]
        )
    
    def _element_at_to_subscript(self, node: FunctionCall) -> ASTNode:
        """Transform ELEMENT_AT(arr, idx) to arr[idx]."""
        # For now, just add a warning - subscript notation needs AST support
        self.warnings.append("ELEMENT_AT converted - verify array indexing (Presto 1-based vs PostgreSQL)")
        return node
    
    def _date_add_to_interval(self, node: FunctionCall) -> BinaryOp:
        """Transform DATE_ADD(unit, value, date) to date + interval."""
        unit = node.args[0]
        value = node.args[1]
        date_expr = node.args[2]
        
        # Create interval expression
        interval = IntervalExpression(value=value, unit=unit.name if isinstance(unit, Identifier) else str(unit))
        return BinaryOp(left=date_expr, operator='+', right=interval)
    
    def _date_diff_to_extract(self, node: FunctionCall) -> ASTNode:
        """Transform DATE_DIFF to PostgreSQL equivalent."""
        self.warnings.append("DATE_DIFF converted - verify precision")
        return node
    
    def _transform_try(self, node: TryExpression) -> ASTNode:
        """TRY() has no PostgreSQL equivalent."""
        self.unsupported.append("TRY() expression has no direct PostgreSQL equivalent")
        # Return the inner expression without TRY wrapper
        return node.expression
    
    def _transform_lambda(self, node: LambdaExpression) -> ASTNode:
        """Lambda expressions are not supported in PostgreSQL."""
        self.unsupported.append(f"Lambda expression not supported in PostgreSQL")
        return node
    
    def _transform_interval(self, node: IntervalExpression) -> IntervalExpression:
        """Ensure interval syntax is PostgreSQL-compatible."""
        # PostgreSQL uses INTERVAL '1 day' format
        return node


class PrestoToMySQLTransformer(DialectTransformer):
    """Transform Presto/Athena SQL to MySQL."""
    
    FUNCTION_MAP = {
        'LISTAGG': 'GROUP_CONCAT',
        'STRING_AGG': 'GROUP_CONCAT',
        'SUBSTR': 'SUBSTRING',
        'STRPOS': 'LOCATE',  # Arguments reversed
        'ARRAY_JOIN': 'GROUP_CONCAT',  # Approximation
        'CARDINALITY': 'JSON_LENGTH',  # For JSON arrays
        'CURRENT_DATE': 'CURDATE',
        'CURRENT_TIME': 'CURTIME',
        'CURRENT_TIMESTAMP': 'NOW',
        'DATE_TRUNC': None,  # Needs special handling
        'TO_UNIXTIME': 'UNIX_TIMESTAMP',
        'FROM_UNIXTIME': 'FROM_UNIXTIME',
        'REGEXP_LIKE': 'REGEXP',
        'NVL': 'IFNULL',
        'COALESCE': 'COALESCE',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, ArrayExpression):
            return self._transform_array(node)
        elif isinstance(node, LambdaExpression):
            self.unsupported.append("Lambda expressions not supported in MySQL")
        elif isinstance(node, TryExpression):
            self.unsupported.append("TRY() not supported in MySQL")
            return node.expression
        return node
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name:
                # Handle reversed arguments for STRPOS -> LOCATE
                if func_name == 'STRPOS' and len(node.args) >= 2:
                    node.args = [node.args[1], node.args[0]]
                node.name = new_name
        
        # IF is native in MySQL, keep as is
        
        # DATE_TRUNC needs special handling
        if func_name == 'DATE_TRUNC':
            return self._date_trunc_to_mysql(node)
        
        return node
    
    def _transform_array(self, node: ArrayExpression) -> ASTNode:
        """Convert arrays to JSON in MySQL."""
        self.warnings.append("Arrays converted to JSON format for MySQL")
        # Would need JSON_ARRAY() function
        return FunctionCall(name='JSON_ARRAY', args=node.elements)
    
    def _date_trunc_to_mysql(self, node: FunctionCall) -> ASTNode:
        """DATE_TRUNC('day', date) -> DATE(date) or similar."""
        if len(node.args) < 2:
            return node
        
        unit = node.args[0]
        date_expr = node.args[1]
        
        unit_str = unit.value.upper() if isinstance(unit, Literal) else str(unit)
        
        # Map to MySQL date functions
        if unit_str in ("'DAY'", 'DAY'):
            return FunctionCall(name='DATE', args=[date_expr])
        elif unit_str in ("'MONTH'", 'MONTH'):
            return FunctionCall(
                name='DATE_FORMAT',
                args=[date_expr, Literal('%Y-%m-01', 'STRING')]
            )
        elif unit_str in ("'YEAR'", 'YEAR'):
            return FunctionCall(
                name='DATE_FORMAT',
                args=[date_expr, Literal('%Y-01-01', 'STRING')]
            )
        
        self.warnings.append(f"DATE_TRUNC with unit {unit_str} may need manual adjustment")
        return node


class PrestoToBigQueryTransformer(DialectTransformer):
    """Transform Presto/Athena SQL to BigQuery."""
    
    FUNCTION_MAP = {
        'LISTAGG': 'STRING_AGG',
        'SUBSTR': 'SUBSTR',
        'STRPOS': 'STRPOS',
        'CARDINALITY': 'ARRAY_LENGTH',
        'ELEMENT_AT': None,  # arr[OFFSET(idx)] or arr[ORDINAL(idx)]
        'ARRAY_JOIN': 'ARRAY_TO_STRING',
        'TRY': 'SAFE',  # SAFE.function_name()
        'APPROX_DISTINCT': 'APPROX_COUNT_DISTINCT',
        'APPROX_PERCENTILE': 'APPROX_QUANTILES',
        'DATE_TRUNC': 'DATE_TRUNC',
        'TO_UNIXTIME': 'UNIX_SECONDS',
        'FROM_UNIXTIME': 'TIMESTAMP_SECONDS',
        'REGEXP_LIKE': 'REGEXP_CONTAINS',
        'REGEXP_EXTRACT': 'REGEXP_EXTRACT',
        'JSON_EXTRACT': 'JSON_EXTRACT',
        'JSON_EXTRACT_SCALAR': 'JSON_EXTRACT_SCALAR',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, ArrayExpression):
            return self._transform_array(node)
        elif isinstance(node, TryExpression):
            return self._transform_try(node)
        elif isinstance(node, LambdaExpression):
            self.unsupported.append("Lambda expressions not directly supported in BigQuery")
        return node
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name:
                node.name = new_name
        
        # IF is native in BigQuery, keep as is
        
        # ELEMENT_AT(arr, n) -> arr[OFFSET(n-1)] or arr[ORDINAL(n)]
        if func_name == 'ELEMENT_AT':
            self.warnings.append("ELEMENT_AT converted - BigQuery uses arr[OFFSET(n)] (0-based) or arr[ORDINAL(n)] (1-based)")
        
        return node
    
    def _transform_array(self, node: ArrayExpression) -> ArrayExpression:
        """BigQuery uses [1,2,3] syntax which is similar."""
        # Arrays are compatible
        return node
    
    def _transform_try(self, node: TryExpression) -> ASTNode:
        """Transform TRY(expr) to SAFE prefix in BigQuery."""
        inner = node.expression
        if isinstance(inner, CastExpression):
            # TRY(CAST(x AS y)) -> SAFE_CAST(x AS y)
            # Set the is_try_cast flag to generate SAFE_CAST
            inner.is_try_cast = True
            return inner
        elif isinstance(inner, FunctionCall):
            # TRY(func(...)) -> SAFE.func(...) or warning
            self.warnings.append(f"TRY({inner.name}) -> Consider using SAFE.{inner.name}()")
            return inner
        return inner


class PostgreSQLToPrestoTransformer(DialectTransformer):
    """Transform PostgreSQL SQL to Presto/Athena."""
    
    FUNCTION_MAP = {
        'SUBSTRING': 'SUBSTR',
        'STRING_AGG': 'LISTAGG',
        'POSITION': 'STRPOS',
        'ARRAY_LENGTH': 'CARDINALITY',
        'ARRAY_TO_STRING': 'ARRAY_JOIN',
        'STRING_TO_ARRAY': 'SPLIT',
        'GENERATE_SERIES': 'SEQUENCE',
        'TO_TIMESTAMP': 'FROM_UNIXTIME',
        'PG_TYPEOF': 'TYPEOF',
        'NOW': 'CURRENT_TIMESTAMP',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, CastExpression):
            return self._transform_cast(node)
        return node
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        if func_name in self.FUNCTION_MAP:
            node.name = self.FUNCTION_MAP[func_name]
        
        # ARRAY_LENGTH(arr, 1) -> CARDINALITY(arr)
        if func_name == 'ARRAY_LENGTH' and len(node.args) > 1:
            node.args = [node.args[0]]  # Remove dimension argument
        
        return node
    
    def _transform_cast(self, node: CastExpression) -> CastExpression:
        """Handle PostgreSQL :: cast syntax - already parsed as CAST."""
        return node


class MySQLToPrestoTransformer(DialectTransformer):
    """Transform MySQL SQL to Presto/Athena."""
    
    FUNCTION_MAP = {
        'GROUP_CONCAT': 'LISTAGG',
        'IFNULL': 'COALESCE',
        'LOCATE': 'STRPOS',  # Arguments reversed
        'CURDATE': 'CURRENT_DATE',
        'CURTIME': 'CURRENT_TIME',
        'NOW': 'CURRENT_TIMESTAMP',
        'UNIX_TIMESTAMP': 'TO_UNIXTIME',
        'FROM_UNIXTIME': 'FROM_UNIXTIME',
        'JSON_ARRAY': None,  # -> ARRAY[]
        'JSON_LENGTH': 'CARDINALITY',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        return node
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name:
                # Handle reversed arguments for LOCATE -> STRPOS
                if func_name == 'LOCATE' and len(node.args) >= 2:
                    node.args = [node.args[1], node.args[0]]
                node.name = new_name
            elif func_name == 'JSON_ARRAY':
                # Convert JSON_ARRAY to ARRAY[]
                return ArrayExpression(elements=node.args)
        
        return node


class TSQLToPrestoTransformer(DialectTransformer):
    """Transform T-SQL (SQL Server) to Presto/Athena."""
    
    FUNCTION_MAP = {
        'ISNULL': 'COALESCE',
        'LEN': 'LENGTH',
        'CHARINDEX': 'STRPOS',  # Arguments reversed
        'GETDATE': 'CURRENT_TIMESTAMP',
        'GETUTCDATE': 'CURRENT_TIMESTAMP',
        'NEWID': 'UUID',
        'DATEPART': None,  # Needs EXTRACT transformation
        'DATEDIFF': 'DATE_DIFF',  # Argument order: unit, start, end
        'DATEADD': 'DATE_ADD',  # Argument order: unit, value, date
        'CONVERT': None,  # -> CAST
        'STR': 'CAST',  # STR(num) -> CAST(num AS VARCHAR)
        'STUFF': None,  # Complex transformation
        'REPLICATE': 'REPEAT',
        'SPACE': None,  # SPACE(n) -> REPEAT(' ', n)
        'LEFT': None,  # LEFT(s, n) -> SUBSTR(s, 1, n)
        'RIGHT': None,  # RIGHT(s, n) -> SUBSTR(s, LENGTH(s)-n+1, n)
        'STRING_SPLIT': 'SPLIT',
        'STRING_AGG': 'LISTAGG',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, IfExpression):
            # T-SQL IIF is parsed as IfExpression, keep it
            return node
        return node
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        # Special transformations
        if func_name == 'IIF':
            # IIF(cond, then, else) is same as IF() in Presto
            return IfExpression(
                condition=node.args[0] if len(node.args) > 0 else None,
                then_expr=node.args[1] if len(node.args) > 1 else None,
                else_expr=node.args[2] if len(node.args) > 2 else None
            )
        
        if func_name == 'CHARINDEX' and len(node.args) >= 2:
            # CHARINDEX(substr, str) -> STRPOS(str, substr)
            node.name = 'STRPOS'
            node.args = [node.args[1], node.args[0]]
            return node
        
        if func_name == 'LEFT' and len(node.args) >= 2:
            # LEFT(str, n) -> SUBSTR(str, 1, n)
            return FunctionCall(
                name='SUBSTR',
                args=[node.args[0], Literal(1, 'INTEGER'), node.args[1]]
            )
        
        if func_name == 'RIGHT' and len(node.args) >= 2:
            # RIGHT(str, n) -> SUBSTR(str, LENGTH(str) - n + 1, n)
            str_expr = node.args[0]
            n_expr = node.args[1]
            start = BinaryOp(
                left=BinaryOp(
                    left=FunctionCall(name='LENGTH', args=[str_expr]),
                    operator='-',
                    right=n_expr
                ),
                operator='+',
                right=Literal(1, 'INTEGER')
            )
            return FunctionCall(name='SUBSTR', args=[str_expr, start, n_expr])
        
        if func_name == 'SPACE' and len(node.args) >= 1:
            # SPACE(n) -> REPEAT(' ', n)
            return FunctionCall(
                name='REPEAT',
                args=[Literal(' ', 'STRING'), node.args[0]]
            )
        
        if func_name == 'CONVERT' and len(node.args) >= 2:
            # CONVERT(type, expr) -> CAST(expr AS type)
            target_type = node.args[0]
            expr = node.args[1]
            type_str = target_type.name if isinstance(target_type, Identifier) else str(target_type)
            return CastExpression(expression=expr, target_type=type_str.upper())
        
        # Direct function mapping
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name:
                node.name = new_name
        
        return node


class TSQLToPostgreSQLTransformer(DialectTransformer):
    """Transform T-SQL (SQL Server) to PostgreSQL."""
    
    FUNCTION_MAP = {
        'ISNULL': 'COALESCE',
        'LEN': 'LENGTH',
        'CHARINDEX': 'POSITION',  # Different syntax: POSITION(substr IN str)
        'GETDATE': 'NOW',
        'GETUTCDATE': 'NOW',
        'NEWID': 'GEN_RANDOM_UUID',
        'DATEDIFF': None,  # Complex transformation
        'DATEADD': None,  # -> date + interval
        'REPLICATE': 'REPEAT',
        'STRING_SPLIT': 'STRING_TO_ARRAY',
        'STRING_AGG': 'STRING_AGG',
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, IfExpression):
            return self._transform_if(node)
        return node
    
    def _transform_if(self, node: IfExpression) -> CaseExpression:
        """Transform IIF/IF to CASE WHEN for PostgreSQL."""
        else_value = node.else_expr if node.else_expr else Literal(None, 'NULL')
        return CaseExpression(
            operand=None,
            when_clauses=[(node.condition, node.then_expr)],
            else_clause=else_value
        )
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        # IIF -> CASE WHEN
        if func_name == 'IIF' and len(node.args) >= 3:
            return CaseExpression(
                operand=None,
                when_clauses=[(node.args[0], node.args[1])],
                else_clause=node.args[2]
            )
        
        if func_name == 'CHARINDEX' and len(node.args) >= 2:
            # CHARINDEX(substr, str) -> POSITION(substr IN str)
            # PostgreSQL syntax is special, use STRPOS as alternative
            node.name = 'STRPOS'
            node.args = [node.args[1], node.args[0]]
            return node
        
        if func_name == 'LEFT' and len(node.args) >= 2:
            # LEFT(str, n) -> SUBSTRING(str, 1, n)
            return FunctionCall(
                name='SUBSTRING',
                args=[node.args[0], Literal(1, 'INTEGER'), node.args[1]]
            )
        
        if func_name == 'RIGHT' and len(node.args) >= 2:
            # RIGHT(str, n) -> SUBSTRING(str FROM LENGTH(str) - n + 1)
            str_expr = node.args[0]
            n_expr = node.args[1]
            start = BinaryOp(
                left=BinaryOp(
                    left=FunctionCall(name='LENGTH', args=[str_expr]),
                    operator='-',
                    right=n_expr
                ),
                operator='+',
                right=Literal(1, 'INTEGER')
            )
            return FunctionCall(name='SUBSTRING', args=[str_expr, start, n_expr])
        
        if func_name == 'SPACE' and len(node.args) >= 1:
            return FunctionCall(
                name='REPEAT',
                args=[Literal(' ', 'STRING'), node.args[0]]
            )
        
        # Direct function mapping
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name:
                node.name = new_name
        
        return node


class PrestoToTSQLTransformer(DialectTransformer):
    """Transform Presto/Athena to T-SQL (SQL Server)."""
    
    FUNCTION_MAP = {
        'COALESCE': 'ISNULL',  # Can keep COALESCE too, but ISNULL is more T-SQL
        'LENGTH': 'LEN',
        'STRPOS': 'CHARINDEX',  # Arguments reversed
        'SUBSTR': 'SUBSTRING',
        'CURRENT_TIMESTAMP': 'GETDATE',
        'CURRENT_DATE': 'CAST',  # CAST(GETDATE() AS DATE)
        'UUID': 'NEWID',
        'LISTAGG': 'STRING_AGG',
        'REPEAT': 'REPLICATE',
        'SPLIT': 'STRING_SPLIT',
        'CARDINALITY': None,  # No direct equivalent
        'ARRAY_AGG': None,  # No direct equivalent
    }
    
    def _apply_transformation(self, node: ASTNode) -> ASTNode:
        if isinstance(node, FunctionCall):
            return self._transform_function(node)
        elif isinstance(node, IfExpression):
            return self._transform_if(node)
        elif isinstance(node, ArrayExpression):
            self.unsupported.append("Arrays not directly supported in T-SQL")
            return node
        elif isinstance(node, LambdaExpression):
            self.unsupported.append("Lambda expressions not supported in T-SQL")
            return node
        return node
    
    def _transform_if(self, node: IfExpression) -> FunctionCall:
        """Transform IF() to IIF() for T-SQL."""
        return FunctionCall(
            name='IIF',
            args=[node.condition, node.then_expr, node.else_expr or Literal(None, 'NULL')]
        )
    
    def _transform_function(self, node: FunctionCall) -> ASTNode:
        func_name = node.name.upper()
        
        if func_name == 'STRPOS' and len(node.args) >= 2:
            # STRPOS(str, substr) -> CHARINDEX(substr, str)
            node.name = 'CHARINDEX'
            node.args = [node.args[1], node.args[0]]
            return node
        
        if func_name == 'CURRENT_DATE':
            # CURRENT_DATE -> CAST(GETDATE() AS DATE)
            return CastExpression(
                expression=FunctionCall(name='GETDATE', args=[]),
                target_type='DATE'
            )
        
        # Direct function mapping
        if func_name in self.FUNCTION_MAP:
            new_name = self.FUNCTION_MAP[func_name]
            if new_name and new_name != 'CAST':  # Skip CAST special handling
                node.name = new_name
        
        return node


class SQLTranspiler:
    """Main transpiler class for converting SQL between dialects."""
    
    # Mapping of (source, target) -> Transformer class
    TRANSFORMERS = {
        (SQLDialect.PRESTO, SQLDialect.POSTGRESQL): PrestoToPostgreSQLTransformer,
        (SQLDialect.ATHENA, SQLDialect.POSTGRESQL): PrestoToPostgreSQLTransformer,
        (SQLDialect.TRINO, SQLDialect.POSTGRESQL): PrestoToPostgreSQLTransformer,
        (SQLDialect.PRESTO, SQLDialect.MYSQL): PrestoToMySQLTransformer,
        (SQLDialect.ATHENA, SQLDialect.MYSQL): PrestoToMySQLTransformer,
        (SQLDialect.PRESTO, SQLDialect.BIGQUERY): PrestoToBigQueryTransformer,
        (SQLDialect.ATHENA, SQLDialect.BIGQUERY): PrestoToBigQueryTransformer,
        (SQLDialect.POSTGRESQL, SQLDialect.PRESTO): PostgreSQLToPrestoTransformer,
        (SQLDialect.POSTGRESQL, SQLDialect.ATHENA): PostgreSQLToPrestoTransformer,
        (SQLDialect.MYSQL, SQLDialect.PRESTO): MySQLToPrestoTransformer,
        (SQLDialect.MYSQL, SQLDialect.ATHENA): MySQLToPrestoTransformer,
        # T-SQL (SQL Server) conversions
        (SQLDialect.TSQL, SQLDialect.PRESTO): TSQLToPrestoTransformer,
        (SQLDialect.TSQL, SQLDialect.ATHENA): TSQLToPrestoTransformer,
        (SQLDialect.TSQL, SQLDialect.POSTGRESQL): TSQLToPostgreSQLTransformer,
        (SQLDialect.TSQL, SQLDialect.MYSQL): TSQLToPrestoTransformer,  # Similar transformations
        (SQLDialect.PRESTO, SQLDialect.TSQL): PrestoToTSQLTransformer,
        (SQLDialect.ATHENA, SQLDialect.TSQL): PrestoToTSQLTransformer,
        (SQLDialect.POSTGRESQL, SQLDialect.TSQL): PrestoToTSQLTransformer,  # Via similar transformations
        (SQLDialect.MYSQL, SQLDialect.TSQL): PrestoToTSQLTransformer,  # Via similar transformations
    }
    
    def transpile(
        self,
        sql: str,
        source_dialect: SQLDialect,
        target_dialect: SQLDialect
    ) -> TranspilationResult:
        """
        Transpile SQL from one dialect to another.
        
        Args:
            sql: The SQL query to transpile
            source_dialect: The source SQL dialect
            target_dialect: The target SQL dialect
            
        Returns:
            TranspilationResult with the transpiled SQL and any warnings
        """
        # Create parser with source dialect
        parser = SQLParser(dialect=source_dialect)
        generator = SQLGenerator(dialect=target_dialect)
        
        # Parse with source dialect
        try:
            result = parser.parse(sql)
        except Exception as e:
            return TranspilationResult(
                success=False,
                sql=sql,
                source_dialect=source_dialect,
                target_dialect=target_dialect,
                warnings=[f"Parse error: {str(e)}"]
            )
        
        # Same dialect, just reformat
        if source_dialect == target_dialect:
            return TranspilationResult(
                success=True,
                sql=generator.generate(result.statement),
                source_dialect=source_dialect,
                target_dialect=target_dialect
            )
        
        # Get transformer
        transformer_key = (source_dialect, target_dialect)
        if transformer_key not in self.TRANSFORMERS:
            # Try to find an indirect path
            transformer = self._find_transformer(source_dialect, target_dialect)
            if not transformer:
                return TranspilationResult(
                    success=False,
                    sql=sql,
                    source_dialect=source_dialect,
                    target_dialect=target_dialect,
                    warnings=[f"No transformer available from {source_dialect.value} to {target_dialect.value}"]
                )
        else:
            transformer = self.TRANSFORMERS[transformer_key](source_dialect, target_dialect)
        
        # Transform AST
        transformed = transformer.transform(result.statement)
        
        # Generate target SQL
        try:
            target_sql = generator.generate(transformed)
        except Exception as e:
            return TranspilationResult(
                success=False,
                sql=sql,
                source_dialect=source_dialect,
                target_dialect=target_dialect,
                warnings=[f"Generation error: {str(e)}"]
            )
        
        return TranspilationResult(
            success=True,
            sql=target_sql,
            source_dialect=source_dialect,
            target_dialect=target_dialect,
            warnings=transformer.warnings,
            unsupported_features=transformer.unsupported
        )
    
    def _find_transformer(
        self,
        source: SQLDialect,
        target: SQLDialect
    ) -> Optional[DialectTransformer]:
        """Try to find an indirect transformation path."""
        # For now, just return None if no direct transformer
        # Could implement multi-hop transformations
        return None
    
    @staticmethod
    def get_supported_conversions() -> Dict[str, List[str]]:
        """Get a dictionary of supported source -> target conversions."""
        conversions: Dict[str, List[str]] = {}
        for (source, target) in SQLTranspiler.TRANSFORMERS.keys():
            source_name = source.value
            target_name = target.value
            if source_name not in conversions:
                conversions[source_name] = []
            conversions[source_name].append(target_name)
        return conversions


def transpile(
    sql: str,
    source_dialect: str | SQLDialect,
    target_dialect: str | SQLDialect
) -> TranspilationResult:
    """
    Convenience function to transpile SQL between dialects.
    
    Args:
        sql: The SQL query to transpile
        source_dialect: Source dialect name or SQLDialect enum
        target_dialect: Target dialect name or SQLDialect enum
        
    Returns:
        TranspilationResult with the transpiled SQL
    
    Example:
        >>> result = transpile(
        ...     "SELECT IF(x > 0, 'positive', 'negative') FROM t",
        ...     "presto",
        ...     "postgresql"
        ... )
        >>> print(result.sql)
        SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'negative' END FROM t
    """
    # Convert string to enum if needed
    if isinstance(source_dialect, str):
        source_dialect = SQLDialect(source_dialect.lower())
    if isinstance(target_dialect, str):
        target_dialect = SQLDialect(target_dialect.lower())
    
    transpiler = SQLTranspiler()
    return transpiler.transpile(sql, source_dialect, target_dialect)
