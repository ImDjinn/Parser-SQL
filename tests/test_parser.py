"""
Tests complets pour le SQL Parser.
"""

import pytest
import sys
sys.path.insert(0, '..')

from sql_parser import SQLParser, SQLGenerator, transpile, SQLDialect


class TestBasicParsing:
    """Tests de parsing basique."""
    
    def test_simple_select(self):
        parser = SQLParser()
        result = parser.parse("SELECT id, name FROM users")
        assert result.statement is not None
        assert len(result.tables_referenced) == 1
        assert "users" in result.tables_referenced
    
    def test_select_with_where(self):
        parser = SQLParser()
        result = parser.parse("SELECT * FROM users WHERE age > 18")
        assert result.statement is not None
        assert result.statement.where_clause is not None
    
    def test_select_with_join(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT u.id, o.total 
            FROM users u 
            JOIN orders o ON u.id = o.user_id
        """)
        assert result.has_join is True
    
    def test_select_with_group_by(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT department, COUNT(*) as cnt 
            FROM employees 
            GROUP BY department 
            HAVING COUNT(*) > 5
        """)
        assert result.statement.group_by is not None
        assert result.statement.having_clause is not None
        assert result.has_aggregation is True
    
    def test_select_with_order_limit(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT * FROM products 
            ORDER BY price DESC 
            LIMIT 10 OFFSET 20
        """)
        assert result.statement.order_by is not None
        assert result.statement.limit is not None
    
    def test_subquery(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT * FROM users 
            WHERE id IN (SELECT user_id FROM orders WHERE total > 100)
        """)
        assert result.has_subquery is True
    
    def test_cte(self):
        parser = SQLParser()
        result = parser.parse("""
            WITH active_users AS (
                SELECT * FROM users WHERE status = 'active'
            )
            SELECT * FROM active_users
        """)
        assert result.statement.ctes is not None
        assert len(result.statement.ctes) == 1


class TestDMLStatements:
    """Tests pour INSERT, UPDATE, DELETE, MERGE."""
    
    def test_insert_values(self):
        parser = SQLParser()
        result = parser.parse("INSERT INTO users (id, name) VALUES (1, 'John'), (2, 'Jane')")
        assert type(result.statement).__name__ == "InsertStatement"
        assert result.statement.table.name == "users"
        assert len(result.statement.values) == 2
    
    def test_insert_select(self):
        parser = SQLParser()
        result = parser.parse("INSERT INTO archive SELECT * FROM users WHERE status = 'inactive'")
        assert type(result.statement).__name__ == "InsertStatement"
        assert result.statement.query is not None
    
    def test_update(self):
        parser = SQLParser()
        result = parser.parse("UPDATE users SET status = 'inactive' WHERE last_login < '2024-01-01'")
        assert type(result.statement).__name__ == "UpdateStatement"
        assert len(result.statement.assignments) == 1
        assert result.statement.where_clause is not None
    
    def test_delete(self):
        parser = SQLParser()
        result = parser.parse("DELETE FROM orders WHERE created_at < NOW() - INTERVAL 30 DAY")
        assert type(result.statement).__name__ == "DeleteStatement"
        assert result.statement.where_clause is not None
    
    def test_merge(self):
        parser = SQLParser()
        result = parser.parse("""
            MERGE INTO target t 
            USING source s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.name = s.name
            WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
        """)
        assert type(result.statement).__name__ == "MergeStatement"
        assert len(result.statement.when_clauses) == 2


class TestDDLStatements:
    """Tests pour CREATE, DROP, ALTER, TRUNCATE."""
    
    def test_create_table(self):
        parser = SQLParser()
        result = parser.parse("""
            CREATE TABLE products (
                id BIGINT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        assert type(result.statement).__name__ == "CreateTableStatement"
        assert len(result.statement.columns) == 4
    
    def test_create_view(self):
        parser = SQLParser()
        result = parser.parse("CREATE VIEW active_users AS SELECT * FROM users WHERE status = 'active'")
        assert type(result.statement).__name__ == "CreateViewStatement"
        assert result.statement.query is not None
    
    def test_drop_table(self):
        parser = SQLParser()
        result = parser.parse("DROP TABLE IF EXISTS temp_data CASCADE")
        assert type(result.statement).__name__ == "DropStatement"
        assert result.statement.if_exists is True
        assert result.statement.cascade is True
    
    def test_alter_table(self):
        parser = SQLParser()
        result = parser.parse("ALTER TABLE users ADD COLUMN email VARCHAR(255)")
        assert type(result.statement).__name__ == "AlterTableStatement"
    
    def test_truncate(self):
        parser = SQLParser()
        result = parser.parse("TRUNCATE TABLE logs")
        assert type(result.statement).__name__ == "TruncateStatement"


class TestWindowFunctions:
    """Tests pour les fonctions de fenêtre."""
    
    def test_row_number(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT id, ROW_NUMBER() OVER (ORDER BY created_at) as rn 
            FROM users
        """)
        assert "row_number" in [f.lower() for f in result.functions_used]
    
    def test_partition_by(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT department, salary,
                   RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
            FROM employees
        """)
        assert result.statement is not None
    
    def test_lag_lead(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT date, value,
                   LAG(value, 1) OVER (ORDER BY date) as prev_value,
                   LEAD(value, 1) OVER (ORDER BY date) as next_value
            FROM metrics
        """)
        assert "lag" in [f.lower() for f in result.functions_used]
        assert "lead" in [f.lower() for f in result.functions_used]


class TestPrestoAthena:
    """Tests pour les fonctionnalités Presto/Athena."""
    
    def test_unnest(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT * FROM users CROSS JOIN UNNEST(tags) AS t(tag)")
        assert result.statement is not None
    
    def test_array_expression(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT ARRAY[1, 2, 3] as arr")
        assert result.statement is not None
    
    def test_map_expression(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT MAP(ARRAY['a', 'b'], ARRAY[1, 2]) as m")
        assert result.statement is not None
    
    def test_lambda(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT TRANSFORM(arr, x -> x * 2) FROM t")
        assert result.statement is not None
    
    def test_try_expression(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT TRY(CAST(x AS INTEGER)) FROM t")
        assert result.statement is not None
    
    def test_if_expression(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT IF(x > 0, 'positive', 'negative') FROM t")
        assert result.statement is not None
    
    def test_interval(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT date_col + INTERVAL '1' DAY FROM t")
        assert result.statement is not None
    
    def test_at_time_zone(self):
        parser = SQLParser(dialect=SQLDialect.PRESTO)
        result = parser.parse("SELECT ts AT TIME ZONE 'UTC' FROM t")
        assert result.statement is not None


class TestSQLGeneration:
    """Tests pour la génération SQL."""
    
    def test_simple_generation(self):
        parser = SQLParser()
        generator = SQLGenerator()
        
        result = parser.parse("SELECT id, name FROM users WHERE age > 18")
        sql = generator.generate(result.statement)
        
        assert "SELECT" in sql
        assert "FROM users" in sql
        assert "WHERE" in sql
    
    def test_roundtrip(self):
        """Test que parsing + génération préserve la sémantique."""
        parser = SQLParser()
        generator = SQLGenerator(inline=True)
        
        original = "SELECT id, name FROM users WHERE age > 18 ORDER BY name"
        result = parser.parse(original)
        generated = generator.generate(result.statement)
        
        # Re-parse le SQL généré
        result2 = parser.parse(generated)
        
        assert result.tables_referenced == result2.tables_referenced
    
    def test_generate_insert(self):
        parser = SQLParser()
        generator = SQLGenerator()
        
        result = parser.parse("INSERT INTO users (id, name) VALUES (1, 'John')")
        sql = generator.generate(result.statement)
        
        assert "INSERT INTO" in sql
        assert "VALUES" in sql
    
    def test_generate_update(self):
        parser = SQLParser()
        generator = SQLGenerator()
        
        result = parser.parse("UPDATE users SET status = 'active' WHERE id = 1")
        sql = generator.generate(result.statement)
        
        assert "UPDATE" in sql
        assert "SET" in sql
    
    def test_generate_create_table(self):
        parser = SQLParser()
        generator = SQLGenerator()
        
        result = parser.parse("CREATE TABLE test (id INT PRIMARY KEY, name VARCHAR(100))")
        sql = generator.generate(result.statement)
        
        assert "CREATE TABLE" in sql


class TestTranspiler:
    """Tests pour le transpileur."""
    
    def test_presto_to_postgresql_if(self):
        result = transpile(
            "SELECT IF(x > 0, 'positive', 'negative') FROM t",
            "presto", "postgresql"
        )
        assert result.success
        assert "CASE WHEN" in result.sql
        assert "THEN" in result.sql
    
    def test_presto_to_bigquery_try_cast(self):
        result = transpile(
            "SELECT TRY(CAST(x AS INTEGER)) FROM t",
            "presto", "bigquery"
        )
        assert result.success
        assert "SAFE_CAST" in result.sql
    
    def test_mysql_to_presto(self):
        result = transpile(
            "SELECT IFNULL(x, 0), CURDATE(), GROUP_CONCAT(name) FROM t GROUP BY cat",
            "mysql", "presto"
        )
        assert result.success
        assert "COALESCE" in result.sql
        assert "CURRENT_DATE" in result.sql
        assert "LISTAGG" in result.sql
    
    def test_postgresql_to_presto(self):
        result = transpile(
            "SELECT SUBSTRING(name, 1, 5), STRING_AGG(val, ',') FROM t GROUP BY name",
            "postgresql", "presto"
        )
        assert result.success
        assert "SUBSTR" in result.sql
        assert "LISTAGG" in result.sql
    
    def test_tsql_to_presto(self):
        result = transpile(
            "SELECT ISNULL(x, 0), LEN(name), GETDATE() FROM users",
            "tsql", "presto"
        )
        assert result.success
        assert "COALESCE" in result.sql
        assert "LENGTH" in result.sql
        assert "CURRENT_TIMESTAMP" in result.sql
    
    def test_presto_to_tsql(self):
        result = transpile(
            "SELECT IF(x > 0, 'y', 'n'), LENGTH(name) FROM t",
            "presto", "tsql"
        )
        assert result.success
        assert "IIF" in result.sql
        assert "LEN" in result.sql
    
    def test_tsql_to_postgresql(self):
        result = transpile(
            "SELECT IIF(status = 1, 'Active', 'Inactive'), ISNULL(email, 'N/A') FROM users",
            "tsql", "postgresql"
        )
        assert result.success
        assert "CASE WHEN" in result.sql
        assert "COALESCE" in result.sql
    
    def test_tsql_left_right(self):
        result = transpile(
            "SELECT LEFT(name, 5), RIGHT(name, 3) FROM users",
            "tsql", "presto"
        )
        assert result.success
        assert "SUBSTR" in result.sql
    
    def test_same_dialect_no_change(self):
        original = "SELECT id, name FROM users"
        result = transpile(original, "presto", "presto")
        assert result.success


class TestDialectDetection:
    """Tests pour la détection automatique de dialecte."""
    
    def test_detect_presto(self):
        from sql_parser.dialects import detect_dialect
        
        sql = "SELECT UNNEST(arr), TRY_CAST(x AS INT) FROM t"
        dialect = detect_dialect(sql)
        assert dialect in (SQLDialect.PRESTO, SQLDialect.ATHENA)
    
    def test_detect_mysql(self):
        from sql_parser.dialects import detect_dialect
        
        sql = "SELECT GROUP_CONCAT(name) FROM `users`"
        dialect = detect_dialect(sql)
        assert dialect == SQLDialect.MYSQL
    
    def test_detect_postgresql(self):
        from sql_parser.dialects import detect_dialect
        
        sql = "SELECT x::int, generate_series(1, 10) FROM t"
        dialect = detect_dialect(sql)
        assert dialect == SQLDialect.POSTGRESQL
    
    def test_detect_tsql(self):
        from sql_parser.dialects import detect_dialect
        
        sql = "SELECT GETDATE(), ISNULL(x, 0) FROM [users]"
        dialect = detect_dialect(sql)
        assert dialect == SQLDialect.TSQL
    
    def test_detect_bigquery(self):
        from sql_parser.dialects import detect_dialect
        
        sql = "SELECT SAFE_CAST(x AS INT64), STRUCT(a, b) FROM `project.dataset.table`"
        dialect = detect_dialect(sql)
        assert dialect == SQLDialect.BIGQUERY


class TestEdgeCases:
    """Tests pour les cas limites."""
    
    def test_quoted_identifiers(self):
        parser = SQLParser()
        result = parser.parse('SELECT * FROM "my-table" WHERE "my-column" = 1')
        assert result.statement is not None
    
    def test_nested_subqueries(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT * FROM (
                SELECT * FROM (
                    SELECT id FROM users
                ) sub1
            ) sub2
        """)
        assert result.statement is not None
    
    def test_multiple_joins(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT * 
            FROM a 
            JOIN b ON a.id = b.a_id
            LEFT JOIN c ON b.id = c.b_id
            CROSS JOIN d
        """)
        assert result.has_join is True
    
    def test_complex_expression(self):
        parser = SQLParser()
        result = parser.parse("""
            SELECT 
                CASE 
                    WHEN x > 100 THEN 'high'
                    WHEN x > 50 THEN 'medium'
                    ELSE 'low'
                END as category,
                COALESCE(a, b, c, 0) as value
            FROM t
        """)
        assert result.statement is not None
    
    def test_empty_string_literal(self):
        parser = SQLParser()
        result = parser.parse("SELECT * FROM t WHERE name != ''")
        assert result.statement is not None
    
    def test_numeric_literals(self):
        parser = SQLParser()
        result = parser.parse("SELECT 1, 1.5, -3.14, 1e10, 2.5e-3 FROM dual")
        assert result.statement is not None
    
    def test_date_literals(self):
        parser = SQLParser()
        result = parser.parse("SELECT * FROM t WHERE dt = DATE '2024-01-01'")
        assert result.statement is not None


class TestJinjaTemplates:
    """Tests pour les templates Jinja (dbt)."""
    
    def test_ref_function(self):
        parser = SQLParser(dialect=SQLDialect.ATHENA)
        result = parser.parse("SELECT * FROM {{ ref('users') }}")
        assert result.statement is not None
    
    def test_source_function(self):
        parser = SQLParser(dialect=SQLDialect.ATHENA)
        result = parser.parse("SELECT * FROM {{ source('raw', 'events') }}")
        assert result.statement is not None
    
    def test_var_function(self):
        parser = SQLParser(dialect=SQLDialect.ATHENA)
        result = parser.parse("SELECT * FROM t WHERE date > '{{ var(\"start_date\") }}'")
        assert result.statement is not None
    
    def test_config_block(self):
        parser = SQLParser(dialect=SQLDialect.ATHENA)
        result = parser.parse("""
            {{ config(materialized='table') }}
            SELECT * FROM users
        """)
        assert result.statement is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
