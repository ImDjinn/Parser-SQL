"""
DBT Model Converter - Convert SQL DML/DDL to dbt models.

This module transforms traditional SQL statements (INSERT, MERGE, UPDATE, etc.)
into dbt model files with appropriate materializations and configurations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .dialects import SQLDialect
from .ast_nodes import (
    ASTNode, SelectStatement, InsertStatement, UpdateStatement, 
    DeleteStatement, MergeStatement, CreateTableStatement, CreateViewStatement,
    TableRef, ColumnRef, BinaryOp, ExistsExpression, InExpression,
    SubqueryExpression, FunctionCall, Identifier
)
from .parser import SQLParser
from .sql_generator import SQLGenerator
from .transpiler import SQLTranspiler


class DbtMaterialization(Enum):
    """dbt materialization types."""
    VIEW = "view"
    TABLE = "table"
    INCREMENTAL = "incremental"
    EPHEMERAL = "ephemeral"
    SNAPSHOT = "snapshot"


class IncrementalStrategy(Enum):
    """Incremental strategies for dbt."""
    APPEND = "append"
    MERGE = "merge"
    DELETE_INSERT = "delete+insert"
    INSERT_OVERWRITE = "insert_overwrite"


@dataclass
class DbtConfig:
    """Configuration for a dbt model."""
    materialized: DbtMaterialization = DbtMaterialization.TABLE
    unique_key: Optional[List[str]] = None
    incremental_strategy: Optional[IncrementalStrategy] = None
    merge_update_columns: Optional[List[str]] = None
    partition_by: Optional[Dict[str, Any]] = None
    cluster_by: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    schema: Optional[str] = None
    alias: Optional[str] = None
    pre_hook: Optional[List[str]] = None
    post_hook: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_jinja(self) -> str:
        """Generate the {{ config(...) }} block."""
        parts = []
        
        parts.append(f"materialized='{self.materialized.value}'")
        
        if self.unique_key:
            if len(self.unique_key) == 1:
                parts.append(f"unique_key='{self.unique_key[0]}'")
            else:
                keys = ", ".join(f"'{k}'" for k in self.unique_key)
                parts.append(f"unique_key=[{keys}]")
        
        if self.incremental_strategy:
            parts.append(f"incremental_strategy='{self.incremental_strategy.value}'")
        
        if self.merge_update_columns:
            cols = ", ".join(f"'{c}'" for c in self.merge_update_columns)
            parts.append(f"merge_update_columns=[{cols}]")
        
        if self.partition_by:
            # Athena/BigQuery style partitioning
            parts.append(f"partition_by={self.partition_by}")
        
        if self.cluster_by:
            cols = ", ".join(f"'{c}'" for c in self.cluster_by)
            parts.append(f"cluster_by=[{cols}]")
        
        if self.schema:
            parts.append(f"schema='{self.schema}'")
        
        if self.alias:
            parts.append(f"alias='{self.alias}'")
        
        if self.tags:
            tags = ", ".join(f"'{t}'" for t in self.tags)
            parts.append(f"tags=[{tags}]")
        
        for key, value in self.extra.items():
            if isinstance(value, str):
                parts.append(f"{key}='{value}'")
            elif isinstance(value, bool):
                parts.append(f"{key}={str(value).lower()}")
            else:
                parts.append(f"{key}={value}")
        
        config_body = ",\n    ".join(parts)
        return f"{{{{ config(\n    {config_body}\n) }}}}"


@dataclass
class DbtModel:
    """Represents a dbt model file."""
    name: str
    config: DbtConfig
    sql: str
    incremental_filter: Optional[str] = None
    source_tables: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_file_content(self) -> str:
        """Generate the complete dbt model file content."""
        parts = []
        
        # Config block
        parts.append(self.config.to_jinja())
        parts.append("")
        
        # Main SQL
        if self.incremental_filter and self.config.materialized == DbtMaterialization.INCREMENTAL:
            # Add incremental filter
            parts.append(self.sql)
            parts.append("")
            parts.append("{% if is_incremental() %}")
            parts.append(self.incremental_filter)
            parts.append("{% endif %}")
        else:
            parts.append(self.sql)
        
        return "\n".join(parts)


@dataclass
class ConversionResult:
    """Result of a SQL to dbt conversion."""
    success: bool
    models: List[DbtModel] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DbtConverter:
    """Converts SQL DML/DDL statements to dbt models."""
    
    def __init__(
        self,
        source_dialect: SQLDialect = SQLDialect.TSQL,
        target_dialect: SQLDialect = SQLDialect.ATHENA,
        default_schema: Optional[str] = None
    ):
        self.source_dialect = source_dialect
        self.target_dialect = target_dialect
        self.default_schema = default_schema
        self.parser = SQLParser(dialect=source_dialect)
        self.generator = SQLGenerator(dialect=target_dialect, inline=False)
        self.transpiler = SQLTranspiler()
    
    def convert(self, sql: str, model_name: Optional[str] = None) -> ConversionResult:
        """
        Convert SQL statement(s) to dbt model(s).
        
        Args:
            sql: The SQL code to convert
            model_name: Optional name for the model (derived from table name if not provided)
            
        Returns:
            ConversionResult with the generated dbt models
        """
        result = ConversionResult(success=True)
        
        try:
            parse_result = self.parser.parse(sql)
            statement = parse_result.statement
            
            if isinstance(statement, InsertStatement):
                model = self._convert_insert(statement, model_name)
                result.models.append(model)
                
            elif isinstance(statement, MergeStatement):
                model = self._convert_merge(statement, model_name)
                result.models.append(model)
                
            elif isinstance(statement, UpdateStatement):
                model = self._convert_update(statement, model_name)
                result.models.append(model)
                
            elif isinstance(statement, DeleteStatement):
                result.warnings.append(
                    "DELETE statements typically don't translate to dbt models. "
                    "Consider using soft deletes or incremental models with delete+insert strategy."
                )
                # Try to extract useful info
                if model_name:
                    config = DbtConfig(
                        materialized=DbtMaterialization.TABLE,
                        extra={"full_refresh": True}
                    )
                    model = DbtModel(
                        name=model_name,
                        config=config,
                        sql="-- DELETE statement requires manual conversion\n-- Original: " + sql[:100],
                        warnings=["DELETE requires manual conversion"]
                    )
                    result.models.append(model)
                    
            elif isinstance(statement, CreateTableStatement):
                model = self._convert_create_table(statement, model_name)
                result.models.append(model)
                
            elif isinstance(statement, CreateViewStatement):
                model = self._convert_create_view(statement, model_name)
                result.models.append(model)
                
            elif isinstance(statement, SelectStatement):
                # Pure SELECT - just wrap in a model
                model = self._convert_select(statement, model_name or "unnamed_model")
                result.models.append(model)
                
            else:
                result.success = False
                result.errors.append(f"Unsupported statement type: {type(statement).__name__}")
                
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
        
        return result
    
    def _convert_insert(self, stmt: InsertStatement, model_name: Optional[str]) -> DbtModel:
        """Convert INSERT INTO statement to dbt model."""
        warnings = []
        
        # Determine model name from target table
        table_name = stmt.table.name if stmt.table else "unnamed"
        name = model_name or self._to_model_name(table_name)
        
        # Analyze the INSERT to determine materialization
        config = DbtConfig()
        incremental_filter = None
        
        if stmt.query:
            # INSERT INTO ... SELECT ...
            
            # Check for incremental patterns
            is_incremental, unique_keys, filter_expr = self._detect_incremental_pattern(stmt)
            
            if is_incremental:
                config.materialized = DbtMaterialization.INCREMENTAL
                config.unique_key = unique_keys
                config.incremental_strategy = IncrementalStrategy.MERGE
                incremental_filter = filter_expr
            else:
                config.materialized = DbtMaterialization.TABLE
            
            # Transpile the SELECT query
            select_sql = self._transpile_select(stmt.query)
            
        elif stmt.values:
            # INSERT INTO ... VALUES ... - usually seed data
            warnings.append(
                "INSERT VALUES converted to seed-like model. "
                "Consider using dbt seeds for static data."
            )
            config.materialized = DbtMaterialization.TABLE
            
            # Generate a SELECT from VALUES
            select_sql = self._values_to_select(stmt.columns, stmt.values)
        else:
            select_sql = "-- No SELECT or VALUES found"
            warnings.append("INSERT statement has no SELECT or VALUES clause")
        
        return DbtModel(
            name=name,
            config=config,
            sql=select_sql,
            incremental_filter=incremental_filter,
            source_tables=self._extract_source_tables(stmt.query) if stmt.query else [],
            warnings=warnings
        )
    
    def _convert_merge(self, stmt: MergeStatement, model_name: Optional[str]) -> DbtModel:
        """Convert MERGE statement to dbt incremental model."""
        warnings = []
        
        # Get target table name
        table_name = stmt.target.name if stmt.target else "unnamed"
        name = model_name or self._to_model_name(table_name)
        
        # MERGE always becomes incremental
        config = DbtConfig(
            materialized=DbtMaterialization.INCREMENTAL,
            incremental_strategy=IncrementalStrategy.MERGE
        )
        
        # Extract unique key from ON condition
        unique_keys = self._extract_join_keys(stmt.on_condition)
        if unique_keys:
            config.unique_key = unique_keys
        else:
            warnings.append("Could not extract unique_key from MERGE condition. Please add manually.")
        
        # Extract update columns from WHEN MATCHED
        update_columns = []
        for when_clause in stmt.when_clauses:
            if when_clause.matched and when_clause.assignments:
                for assignment in when_clause.assignments:
                    if hasattr(assignment, 'column'):
                        col_name = assignment.column.column if isinstance(assignment.column, ColumnRef) else str(assignment.column)
                        update_columns.append(col_name)
        
        if update_columns:
            config.merge_update_columns = update_columns
        
        # Build SELECT from source
        if stmt.source:
            source_name = stmt.source.name if hasattr(stmt.source, 'name') else "source"
            
            # Try to build SELECT with all columns from source
            # This is a simplification - real conversion may need more analysis
            if hasattr(stmt.source, 'query'):
                # Source is a subquery
                select_sql = self._transpile_select(stmt.source.query)
            else:
                # Source is a table reference
                select_sql = f"SELECT *\nFROM {{{{ ref('{self._to_model_name(source_name)}') }}}}"
                warnings.append(f"Using SELECT * from source table. Consider specifying explicit columns.")
        else:
            select_sql = "-- Source table not found in MERGE"
            warnings.append("Could not determine source for MERGE")
        
        # Generate incremental filter based on typical patterns
        incremental_filter = self._generate_incremental_filter(stmt)
        
        return DbtModel(
            name=name,
            config=config,
            sql=select_sql,
            incremental_filter=incremental_filter,
            warnings=warnings
        )
    
    def _convert_update(self, stmt: UpdateStatement, model_name: Optional[str]) -> DbtModel:
        """Convert UPDATE statement to dbt model."""
        warnings = []
        
        table_name = stmt.table.name if stmt.table else "unnamed"
        name = model_name or self._to_model_name(table_name)
        
        warnings.append(
            "UPDATE statements are complex to convert. "
            "Generated an incremental model with merge strategy. Review carefully."
        )
        
        config = DbtConfig(
            materialized=DbtMaterialization.INCREMENTAL,
            incremental_strategy=IncrementalStrategy.MERGE
        )
        
        # Try to identify the key columns from WHERE clause
        if stmt.where_clause:
            keys = self._extract_equality_columns(stmt.where_clause)
            if keys:
                config.unique_key = keys
        
        # Extract updated columns
        update_columns = []
        for assignment in stmt.assignments:
            if hasattr(assignment, 'column'):
                col_name = assignment.column.column if isinstance(assignment.column, ColumnRef) else str(assignment.column)
                update_columns.append(col_name)
        
        if update_columns:
            config.merge_update_columns = update_columns
        
        # Build a SELECT that would produce the updated records
        # This is approximate and needs manual review
        if stmt.from_clause:
            # UPDATE ... FROM ... pattern (common in T-SQL)
            select_sql = f"-- Converted from UPDATE...FROM pattern\n"
            select_sql += f"-- Review and adjust column selection\n"
            select_sql += f"SELECT\n"
            
            # List updated columns with their new values
            for assignment in stmt.assignments:
                col = assignment.column.column if isinstance(assignment.column, ColumnRef) else str(assignment.column)
                val = self.generator.generate(assignment.value) if hasattr(assignment, 'value') else '?'
                select_sql += f"    {val} AS {col},\n"
            
            select_sql += f"    -- Add other columns as needed\n"
            select_sql += f"FROM {self.generator.generate(stmt.from_clause)}"
        else:
            select_sql = f"-- UPDATE without FROM clause\n"
            select_sql += f"-- Original table: {table_name}\n"
            select_sql += f"SELECT * FROM {{{{ this }}}}"
        
        return DbtModel(
            name=name,
            config=config,
            sql=select_sql,
            warnings=warnings
        )
    
    def _convert_create_table(self, stmt: CreateTableStatement, model_name: Optional[str]) -> DbtModel:
        """Convert CREATE TABLE to dbt model."""
        table_name = stmt.table.name if stmt.table else "unnamed"
        name = model_name or self._to_model_name(table_name)
        
        config = DbtConfig(materialized=DbtMaterialization.TABLE)
        
        if stmt.temporary:
            config.extra["temporary"] = True
        
        if stmt.as_query:
            # CREATE TABLE AS SELECT
            select_sql = self._transpile_select(stmt.as_query)
        else:
            # CREATE TABLE with column definitions
            # This becomes a model that creates the structure
            select_sql = self._columns_to_select(stmt.columns)
        
        return DbtModel(
            name=name,
            config=config,
            sql=select_sql,
            source_tables=self._extract_source_tables(stmt.as_query) if stmt.as_query else []
        )
    
    def _convert_create_view(self, stmt: CreateViewStatement, model_name: Optional[str]) -> DbtModel:
        """Convert CREATE VIEW to dbt model."""
        view_name = stmt.name if stmt.name else "unnamed"
        name = model_name or self._to_model_name(view_name)
        
        config = DbtConfig(materialized=DbtMaterialization.VIEW)
        
        if stmt.query:
            select_sql = self._transpile_select(stmt.query)
        else:
            select_sql = "-- No query found in CREATE VIEW"
        
        return DbtModel(
            name=name,
            config=config,
            sql=select_sql
        )
    
    def _convert_select(self, stmt: SelectStatement, model_name: str) -> DbtModel:
        """Convert pure SELECT to dbt model."""
        config = DbtConfig(materialized=DbtMaterialization.VIEW)
        select_sql = self._transpile_select(stmt)
        
        return DbtModel(
            name=model_name,
            config=config,
            sql=select_sql,
            source_tables=self._extract_source_tables(stmt)
        )
    
    def _transpile_select(self, stmt: SelectStatement) -> str:
        """Transpile SELECT statement to target dialect and replace table refs with refs."""
        # Generate SQL from source AST
        source_generator = SQLGenerator(dialect=self.source_dialect, inline=False)
        source_sql = source_generator.generate(stmt)
        
        # Transpile to target dialect
        transpile_result = self.transpiler.transpile(source_sql, self.source_dialect, self.target_dialect)
        
        if transpile_result.success:
            transpiled_sql = transpile_result.sql
        else:
            # Fallback to source SQL if transpilation fails
            transpiled_sql = source_sql
        
        # Replace table references with dbt ref() or source()
        return self._replace_table_refs(transpiled_sql, stmt)
    
    def _replace_table_refs(self, sql: str, stmt: SelectStatement) -> str:
        """Replace table references with dbt ref() calls."""
        import re
        
        # Extract all referenced tables
        tables = self._extract_source_tables(stmt)
        
        result = sql
        for table in tables:
            # Simple replacement - could be improved with proper AST manipulation
            model_name = self._to_model_name(table)
            replacement = f"{{{{ ref('{model_name}') }}}}"
            
            # Match schema.table or just table name
            # Handle cases like dbo.TableName -> {{ ref('table_name') }}
            pattern = rf'(?:\w+\.)*\b{re.escape(table)}\b(?!\s*\()'
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _detect_incremental_pattern(self, stmt: InsertStatement) -> Tuple[bool, List[str], Optional[str]]:
        """
        Detect if an INSERT statement follows an incremental pattern.
        
        Common patterns:
        - INSERT INTO ... SELECT ... WHERE NOT EXISTS (SELECT ... WHERE key = key)
        - INSERT INTO ... SELECT ... WHERE date > (SELECT MAX(date) FROM target)
        - INSERT INTO ... SELECT ... WHERE id NOT IN (SELECT id FROM target)
        """
        is_incremental = False
        unique_keys = []
        filter_expr = None
        
        if not stmt.query or not stmt.query.where_clause:
            return False, [], None
        
        where = stmt.query.where_clause
        
        # Check for NOT EXISTS pattern
        if self._has_not_exists_pattern(where):
            is_incremental = True
            unique_keys = self._extract_not_exists_keys(where)
            if unique_keys:
                filter_expr = f"AND {unique_keys[0]} NOT IN (SELECT {unique_keys[0]} FROM {{{{ this }}}})"
        
        # Check for date/timestamp comparison pattern
        date_col = self._detect_date_filter(where)
        if date_col:
            is_incremental = True
            filter_expr = f"AND {date_col} > (SELECT MAX({date_col}) FROM {{{{ this }}}})"
        
        return is_incremental, unique_keys, filter_expr
    
    def _has_not_exists_pattern(self, expr) -> bool:
        """Check if expression contains NOT EXISTS subquery."""
        if isinstance(expr, ExistsExpression):
            return True
        if isinstance(expr, BinaryOp):
            return self._has_not_exists_pattern(expr.left) or self._has_not_exists_pattern(expr.right)
        return False
    
    def _extract_not_exists_keys(self, expr) -> List[str]:
        """Extract key columns from NOT EXISTS pattern."""
        # Simplified - would need full AST traversal for complete implementation
        return []
    
    def _detect_date_filter(self, expr) -> Optional[str]:
        """Detect date/timestamp column used for filtering."""
        # Look for common patterns like: date > '...' or updated_at > (SELECT MAX...)
        date_columns = ['date', 'created_at', 'updated_at', 'modified_at', 'load_date', 'etl_date']
        
        if isinstance(expr, BinaryOp):
            if isinstance(expr.left, ColumnRef):
                col_name = expr.left.column.lower()
                if any(dc in col_name for dc in date_columns):
                    return expr.left.column
        
        return None
    
    def _extract_join_keys(self, condition) -> List[str]:
        """Extract join key columns from a condition."""
        keys = []
        
        if isinstance(condition, BinaryOp):
            if condition.operator == '=':
                if isinstance(condition.left, ColumnRef):
                    keys.append(condition.left.column)
                if isinstance(condition.right, ColumnRef):
                    # Avoid duplicates
                    if condition.right.column not in keys:
                        keys.append(condition.right.column)
            elif condition.operator.upper() == 'AND':
                keys.extend(self._extract_join_keys(condition.left))
                keys.extend(self._extract_join_keys(condition.right))
        
        return keys
    
    def _extract_equality_columns(self, where_clause) -> List[str]:
        """Extract columns used in equality conditions in WHERE clause."""
        columns = []
        
        if isinstance(where_clause, BinaryOp):
            if where_clause.operator == '=':
                if isinstance(where_clause.left, ColumnRef):
                    columns.append(where_clause.left.column)
            elif where_clause.operator.upper() == 'AND':
                columns.extend(self._extract_equality_columns(where_clause.left))
                columns.extend(self._extract_equality_columns(where_clause.right))
        
        return columns
    
    def _extract_source_tables(self, stmt: SelectStatement) -> List[str]:
        """Extract all source table names from a SELECT statement."""
        tables = []
        
        if stmt and stmt.from_clause:
            for table in stmt.from_clause.tables:
                if isinstance(table, TableRef):
                    tables.append(table.name)
        
        return tables
    
    def _generate_incremental_filter(self, stmt: MergeStatement) -> Optional[str]:
        """Generate an incremental filter for a MERGE statement."""
        # Default incremental filter based on common patterns
        return "WHERE updated_at > (SELECT COALESCE(MAX(updated_at), '1900-01-01') FROM {{ this }})"
    
    def _values_to_select(self, columns: List[str], values: List) -> str:
        """Convert VALUES clause to SELECT with UNION ALL."""
        if not values:
            return "-- No values provided"
        
        selects = []
        for row in values:
            cols_vals = []
            for i, val in enumerate(row):
                col_name = columns[i] if columns and i < len(columns) else f"col{i}"
                val_str = self.generator.generate(val) if hasattr(val, 'to_dict') else str(val)
                cols_vals.append(f"{val_str} AS {col_name}")
            selects.append(f"SELECT {', '.join(cols_vals)}")
        
        return "\nUNION ALL\n".join(selects)
    
    def _columns_to_select(self, columns) -> str:
        """Generate a SELECT with NULL values for column definitions (schema only)."""
        if not columns:
            return "-- No columns defined"
        
        cols = []
        for col in columns:
            col_name = col.name if hasattr(col, 'name') else str(col)
            data_type = col.data_type if hasattr(col, 'data_type') else 'STRING'
            cols.append(f"CAST(NULL AS {data_type}) AS {col_name}")
        
        return f"SELECT\n    " + ",\n    ".join(cols) + "\nWHERE 1 = 0  -- Schema only, no data"
    
    def _to_model_name(self, table_name: str) -> str:
        """Convert table name to dbt model name convention."""
        # Remove schema prefix
        if '.' in table_name:
            table_name = table_name.split('.')[-1]
        
        # Convert to lowercase snake_case
        import re
        # Handle CamelCase
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', table_name).lower()
        # Remove any non-alphanumeric characters except underscore
        name = re.sub(r'[^a-z0-9_]', '_', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        return name


def convert_to_dbt(
    sql: str,
    source_dialect: str | SQLDialect = SQLDialect.TSQL,
    target_dialect: str | SQLDialect = SQLDialect.ATHENA,
    model_name: Optional[str] = None
) -> ConversionResult:
    """
    Convenience function to convert SQL to dbt model.
    
    Args:
        sql: The SQL code to convert
        source_dialect: Source SQL dialect (default: T-SQL)
        target_dialect: Target dbt dialect (default: Athena)
        model_name: Optional name for the model
        
    Returns:
        ConversionResult with the generated dbt models
    
    Example:
        >>> result = convert_to_dbt(
        ...     "INSERT INTO target SELECT * FROM source WHERE date > '2024-01-01'",
        ...     "tsql",
        ...     "athena"
        ... )
        >>> print(result.models[0].to_file_content())
    """
    if isinstance(source_dialect, str):
        source_dialect = SQLDialect(source_dialect.lower())
    if isinstance(target_dialect, str):
        target_dialect = SQLDialect(target_dialect.lower())
    
    converter = DbtConverter(source_dialect, target_dialect)
    return converter.convert(sql, model_name)
