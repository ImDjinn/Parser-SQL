# SQL Parser

A high-performance SQL parser written in Python that converts SQL code into structured JSON/AST. Supports multiple SQL dialects with cross-dialect transpilation and dbt model generation.

## Features

- **Multi-Dialect Support**: Parse SQL from 10 different dialects
  - Standard SQL, Presto, Athena, Trino, PostgreSQL, MySQL, BigQuery, Snowflake, Spark, T-SQL

- **Complete SQL Coverage**:
  - DML: SELECT, INSERT, UPDATE, DELETE, MERGE
  - DDL: CREATE TABLE/VIEW, DROP, ALTER, TRUNCATE
  - Expressions: CASE, subqueries, CTEs, window functions
  - Presto/Athena: UNNEST, ARRAY[], MAP(), lambdas, TRY_CAST
  - T-SQL: TOP, NOLOCK hints, bracket identifiers `[column]`

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
print(result.statement_type)     # 'SELECT'

# Get the AST
ast = result.statement
print(ast.to_dict())  # Full AST as dictionary
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
    "columns": [{"node_type": "Star"}],
    "from_clause": {...},
    "where_clause": {...}
  }
}
```

### Dialect-Specific Parsing

```python
from sql_parser import SQLParser, SQLDialect

# Parse T-SQL with bracket identifiers
parser = SQLParser(dialect=SQLDialect.TSQL)
result = parser.parse("SELECT TOP 10 [First Name] FROM [Users Table] WITH (NOLOCK)")

# Parse Presto/Athena with lambdas and arrays
parser = SQLParser(dialect=SQLDialect.PRESTO)
result = parser.parse("""
    SELECT TRANSFORM(items, x -> x * 2) as doubled,
           TRY_CAST(value AS INTEGER) as safe_value
    FROM orders
    CROSS JOIN UNNEST(line_items) AS t(item)
""")
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
print(result.jinja_variables)  # ['table_name']
print(result.jinja_refs)       # ['users']
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
    statement_type: str           # 'SELECT', 'INSERT', etc.
    tables_referenced: Set[str]   # Tables used in query
    columns_referenced: Set[str]  # Columns referenced
    functions_used: Set[str]      # Functions called
    has_aggregation: bool         # Contains aggregate functions
    has_subquery: bool            # Contains subqueries
    has_cte: bool                 # Contains WITH clause
    cte_names: List[str]          # CTE names defined
    jinja_variables: List[str]    # Jinja variables found
    jinja_refs: List[str]         # dbt ref() calls found
    dialect: SQLDialect           # Detected/specified dialect
    warnings: List[str]           # Parse warnings
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
| PostgreSQL | `SQLDialect.POSTGRESQL` | `::` casts |
| MySQL | `SQLDialect.MYSQL` | Backtick identifiers |
| BigQuery | `SQLDialect.BIGQUERY` | SAFE_CAST, INT64 |
| Snowflake | `SQLDialect.SNOWFLAKE` | FLATTEN |
| Spark SQL | `SQLDialect.SPARK` | Similar to Presto |
| T-SQL | `SQLDialect.TSQL` | TOP, NOLOCK, `[identifiers]` |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=sql_parser --cov-report=html
```

Current test coverage: **265 tests passing**

## Project Structure

```
sql_parser/
├── __init__.py          # Public API exports
├── tokenizer.py         # Lexical analyzer
├── ast_nodes.py         # AST node definitions
├── parser.py            # SQL parser
├── json_exporter.py     # JSON export
├── sql_generator.py     # SQL generation from AST
├── transpiler.py        # Cross-dialect transpilation
├── dbt_converter.py     # dbt model generation
└── dialects.py          # Dialect definitions

tests/
├── test_parser.py       # Parser unit tests
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
