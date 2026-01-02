# SQL Parser

A high-performance SQL parser written in Python that converts SQL code into structured JSON/AST. Supports multiple SQL dialects with cross-dialect transpilation, dbt model generation, and SQL formatting.

## Features

- **Multi-Dialect Support**: Parse SQL from 10 different dialects
  - Standard SQL, Presto, Athena, Trino, PostgreSQL, MySQL, BigQuery, Snowflake, Spark, T-SQL

- **Complete SQL Coverage**:
  - DML: SELECT, INSERT, UPDATE, DELETE, MERGE, VALUES
  - DDL: CREATE TABLE/VIEW, DROP, ALTER, TRUNCATE
  - Admin: EXPLAIN, VACUUM, GRANT, REVOKE
  - Expressions: CASE, subqueries, CTEs, window functions
  - Advanced: GROUPING SETS, CUBE, ROLLUP, EXTRACT, JSON functions
  - Presto/Athena: UNNEST, ARRAY[], MAP(), lambdas, TRY_CAST
  - T-SQL: TOP, NOLOCK hints, bracket identifiers `[column]`, CROSS/OUTER APPLY
  - Data Lakes: catalog.schema.table (3-level naming for Databricks/Unity Catalog)

- **SQL Formatter**: Format and indent SQL code with customizable styles

- **Transpilation**: Convert SQL between dialects with automatic function mapping

- **dbt Converter**: Generate dbt models from T-SQL scripts with proper materializations

- **Jinja/dbt Support**: Parse SQL containing `{{ }}` and `{% %}` templates

## Installation

```bash
# Clone the repository
git clone https://github.com/ImDjinn/Parser-SQL.git
cd Parser-SQL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Parsing

```python
from sql_parser import SQLParser

# Parse a SQL query
parser = SQLParser()
result = parser.parse("SELECT id, name FROM users WHERE status = 'active'")

# Access parse results
print(result.tables_referenced)  # ['users']
print(result.columns_referenced) # ['id', 'name', 'status']

# Get the AST
ast = result.statement
print(ast.to_dict())  # Full AST as dictionary
```

### SQL Formatting

```python
from sql_parser import format_sql, minify_sql, validate_sql

# Format ugly SQL
ugly_sql = "SELECT a,b,c FROM t WHERE x>1 AND y<2 ORDER BY a"
formatted = format_sql(ugly_sql)
print(formatted)
# SELECT
#     a,
#     b,
#     c
# FROM t
# WHERE ((x > 1) AND (y < 2))
# ORDER BY a

# Minify SQL
minified = minify_sql(formatted)
print(minified)
# SELECT a, b, c FROM t WHERE ((x > 1) AND (y < 2)) ORDER BY a

# Validate SQL syntax
result = validate_sql("SELECT * FROM users")
print(result['valid'])  # True
print(result['info'])   # {'statement_type': 'SelectStatement', 'tables': ['users'], ...}
```

### Export to JSON

```python
from sql_parser import SQLParser
from sql_parser.json_exporter import export_to_json

result = SQLParser().parse("SELECT * FROM orders WHERE total > 100")
json_output = export_to_json(result, indent=2)
print(json_output)
```

Output:
```json
{
  "statement_type": "SELECT",
  "tables_referenced": ["orders"],
  "columns_referenced": ["total"],
  "has_aggregation": false,
  "has_subquery": false,
  "statement": {
    "node_type": "SelectStatement",
    "select": [{"node_type": "Star"}],
    "from_clause": {...},
    "where_clause": {...}
  }
}
```

### Dialect-Specific Parsing

```python
from sql_parser import SQLParser, SQLDialect

# Parse T-SQL with TOP and bracket identifiers
parser = SQLParser(dialect=SQLDialect.TSQL)
result = parser.parse("SELECT TOP 10 [First Name] FROM [Users Table] WITH (NOLOCK)")

# Parse with CROSS APPLY (T-SQL)
result = parser.parse("""
    SELECT u.name, o.total
    FROM users u
    CROSS APPLY (SELECT TOP 1 * FROM orders WHERE user_id = u.id) o
""")

# Parse Presto/Athena with lambdas and arrays
parser = SQLParser(dialect=SQLDialect.PRESTO)
result = parser.parse("""
    SELECT TRANSFORM(items, x -> x * 2) as doubled,
           TRY_CAST(value AS INTEGER) as safe_value
    FROM orders
    CROSS JOIN UNNEST(line_items) AS t(item)
""")

# Parse Databricks/Unity Catalog with 3-level names
result = parser.parse("SELECT * FROM hive_metastore.bronze.raw_events")
```

## SQL Generation

Convert an AST back to SQL:

```python
from sql_parser import SQLParser
from sql_parser.sql_generator import SQLGenerator, SQLDialect

# Parse SQL
result = SQLParser().parse("SELECT id, name FROM users")

# Generate SQL (with optional dialect)
generator = SQLGenerator(dialect=SQLDialect.POSTGRESQL)
sql = generator.generate(result.statement)
print(sql)  # SELECT id, name FROM users
```

## Transpilation

Convert SQL between dialects with automatic function mapping:

```python
from sql_parser import transpile, SQLDialect

# T-SQL to Presto
result = transpile(
    "SELECT ISNULL(name, 'Unknown'), GETDATE() FROM users",
    source_dialect=SQLDialect.TSQL,
    target_dialect=SQLDialect.PRESTO
)
print(result.sql)
# SELECT COALESCE(name, 'Unknown'), CURRENT_TIMESTAMP FROM users

# Presto to BigQuery (TRY_CAST → SAFE_CAST)
result = transpile(
    "SELECT TRY_CAST(x AS INT) FROM t",
    source_dialect=SQLDialect.PRESTO,
    target_dialect=SQLDialect.BIGQUERY
)
print(result.sql)
# SELECT SAFE_CAST(x AS INT64) FROM t
```

### Function Mappings

| Source Dialect | Function | Target Dialect | Mapped To |
|----------------|----------|----------------|-----------|
| T-SQL | `ISNULL(a, b)` | Presto | `COALESCE(a, b)` |
| T-SQL | `GETDATE()` | Presto | `CURRENT_TIMESTAMP` |
| T-SQL | `LEN(s)` | Presto | `LENGTH(s)` |
| MySQL | `IFNULL(a, b)` | Presto | `COALESCE(a, b)` |
| MySQL | `NOW()` | Presto | `CURRENT_TIMESTAMP` |
| Presto | `TRY_CAST` | BigQuery | `SAFE_CAST` |
| Presto | `ARRAY_JOIN` | MySQL | `GROUP_CONCAT` |

## dbt Model Converter

Convert T-SQL DML/DDL scripts to dbt models with proper materializations:

```python
from sql_parser import convert_to_dbt, SQLDialect
from sql_parser.dbt_converter import DbtMaterialization

# Convert INSERT to dbt table model
result = convert_to_dbt(
    sql="INSERT INTO analytics.users SELECT * FROM staging.users",
    source_dialect=SQLDialect.TSQL,
    target_dialect=SQLDialect.ATHENA,
    model_name="stg_users"
)

print(result.models[0].content)
```

Output:
```sql
{{
  config(
    materialized='table'
  )
}}

SELECT *
FROM {{ ref('users') }}
```

### Materialization Mapping

| Source Statement | dbt Materialization |
|------------------|---------------------|
| `INSERT INTO ... SELECT` | `table` |
| `MERGE INTO ... WHEN MATCHED` | `incremental` |
| `CREATE VIEW` | `view` |
| `CREATE TABLE AS SELECT` | `table` |

### MERGE to Incremental Example

```python
result = convert_to_dbt("""
    MERGE INTO target t
    USING source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.name = s.name
    WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
""", SQLDialect.TSQL, SQLDialect.ATHENA, "incremental_users")

print(result.models[0].content)
```

Output:
```sql
{{
  config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='merge'
  )
}}

SELECT s.id, s.name
FROM {{ ref('source') }} s
{% if is_incremental() %}
  WHERE s.updated_at > (SELECT MAX(updated_at) FROM {{ this }})
{% endif %}
```

## Admin Statements

Parse database administration statements:

```python
from sql_parser import SQLParser

parser = SQLParser()

# EXPLAIN
result = parser.parse("EXPLAIN ANALYZE SELECT * FROM users")
print(result.statement.analyze)  # True

# VACUUM (PostgreSQL)
result = parser.parse("VACUUM FULL ANALYZE users (name, email)")

# GRANT/REVOKE
result = parser.parse("GRANT SELECT, INSERT ON TABLE orders TO admin WITH GRANT OPTION")
result = parser.parse("REVOKE ALL PRIVILEGES ON DATABASE mydb FROM old_user CASCADE")
```

## Jinja Template Support

Parse SQL containing dbt/Jinja templates:

```python
from sql_parser import SQLParser

sql = """
{% set table_name = 'users' %}
SELECT * FROM {{ ref(table_name) }}
WHERE created_at > {{ var('start_date') }}
"""

result = SQLParser().parse(sql)
```

## API Reference

### SQLParser

```python
class SQLParser:
    def __init__(self, dialect: SQLDialect = None):
        """Initialize parser with optional dialect (auto-detected if None)."""
    
    def parse(self, sql: str) -> ParseResult:
        """Parse SQL string and return ParseResult."""
```

### ParseResult

```python
@dataclass
class ParseResult:
    statement: Statement          # AST root node
    tables_referenced: List[str]  # Tables used in query
    columns_referenced: List[str] # Columns referenced
    functions_used: List[str]     # Functions called
    has_aggregation: bool         # Contains aggregate functions
    has_subquery: bool            # Contains subqueries
    has_join: bool                # Contains JOIN clauses
```

### SQL Formatter

```python
from sql_parser import format_sql, minify_sql, validate_sql, FormatStyle

# Format with style
formatted = format_sql(sql, style=FormatStyle.STANDARD)
formatted = format_sql(sql, style=FormatStyle.COMPACT)
formatted = format_sql(sql, style=FormatStyle.EXPANDED)

# Custom options
formatted = format_sql(sql, indent_size=2, uppercase_keywords=True)

# Minify
minified = minify_sql(sql)

# Validate
result = validate_sql(sql)
# Returns: {'valid': bool, 'error': str|None, 'formatted': str, 'info': dict}
```

### Transpiler

```python
def transpile(
    sql: str,
    source_dialect: SQLDialect,
    target_dialect: SQLDialect
) -> TranspileResult:
    """Transpile SQL from one dialect to another."""

@dataclass
class TranspileResult:
    success: bool
    sql: str              # Transpiled SQL
    errors: List[str]
    warnings: List[str]
```

### dbt Converter

```python
def convert_to_dbt(
    sql: str,
    source_dialect: SQLDialect,
    target_dialect: SQLDialect,
    model_name: str
) -> ConversionResult:
    """Convert SQL to dbt model."""

@dataclass
class ConversionResult:
    success: bool
    models: List[DbtModel]
    errors: List[str]
    warnings: List[str]
```

## Supported SQL Dialects

| Dialect | Enum Value | Special Features |
|---------|------------|------------------|
| Standard SQL | `SQLDialect.STANDARD` | ANSI SQL |
| Presto | `SQLDialect.PRESTO` | UNNEST, lambdas, TRY_CAST |
| Athena | `SQLDialect.ATHENA` | Same as Presto |
| Trino | `SQLDialect.TRINO` | Same as Presto |
| PostgreSQL | `SQLDialect.POSTGRESQL` | `::` casts, VACUUM, EXPLAIN ANALYZE |
| MySQL | `SQLDialect.MYSQL` | Backtick identifiers |
| BigQuery | `SQLDialect.BIGQUERY` | SAFE_CAST, INT64 |
| Snowflake | `SQLDialect.SNOWFLAKE` | FLATTEN |
| Spark SQL | `SQLDialect.SPARK` | Similar to Presto |
| T-SQL | `SQLDialect.TSQL` | TOP, NOLOCK, `[identifiers]`, CROSS/OUTER APPLY |

## Supported SQL Features

### Statements
- SELECT (with all clauses)
- INSERT, UPDATE, DELETE
- MERGE (with WHEN MATCHED/NOT MATCHED)
- VALUES (standalone)
- CREATE TABLE/VIEW, DROP, ALTER, TRUNCATE
- EXPLAIN, VACUUM
- GRANT, REVOKE

### Clauses & Expressions
- FROM with JOINs (INNER, LEFT, RIGHT, FULL, CROSS, NATURAL)
- CROSS APPLY, OUTER APPLY (T-SQL)
- WHERE, GROUP BY, HAVING, ORDER BY
- LIMIT, OFFSET, TOP (T-SQL)
- DISTINCT, DISTINCT ON (PostgreSQL)
- CTEs (WITH clause, recursive)
- UNION, INTERSECT, EXCEPT
- Window functions (OVER, PARTITION BY, frame specs)
- CASE expressions
- Subqueries (scalar, IN, EXISTS)
- GROUPING SETS, CUBE, ROLLUP
- EXTRACT(field FROM expr)
- JSON functions (JSON_OBJECT, JSON_ARRAY)
- ALL/ANY/SOME comparisons

### Identifiers
- Regular: `table_name`
- Quoted: `"table name"`
- Backticks: `` `table` `` (MySQL)
- Brackets: `[table]` (T-SQL)
- 3-level: `catalog.schema.table` (Databricks)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=sql_parser --cov-report=html
```

Current test coverage: **589 tests passing**

## Project Structure

```
sql_parser/
├── __init__.py          # Public API exports
├── tokenizer.py         # Lexical analyzer
├── ast_nodes.py         # AST node definitions
├── parser.py            # SQL parser
├── json_exporter.py     # JSON export
├── sql_generator.py     # SQL generation from AST
├── formatter.py         # SQL formatting utilities
├── transpiler.py        # Cross-dialect transpilation
├── dbt_converter.py     # dbt model generation
└── dialects.py          # Dialect definitions

tests/
├── test_parser.py       # Parser unit tests
├── test_new_features.py # New features tests (62 tests)
├── test_dbt_converter.py # dbt converter tests
└── test_regression.py   # Comprehensive regression tests
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request
