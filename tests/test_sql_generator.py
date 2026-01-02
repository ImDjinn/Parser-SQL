"""
Tests exhaustifs pour le générateur SQL et le transpileur.
"""

import pytest
import sys
sys.path.insert(0, '..')

from sql_parser import SQLParser, SQLGenerator, SQLDialect, transpile


# ============================================================
# SECTION 1: SQL GENERATOR - BASIC SELECT
# ============================================================

class TestGeneratorBasicSelect:
    """Tests de génération SELECT basique."""
    
    def test_generate_select_star(self):
        result = SQLParser().parse("SELECT * FROM users")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "SELECT" in sql
        assert "*" in sql
        assert "users" in sql
    
    def test_generate_select_columns(self):
        result = SQLParser().parse("SELECT id, name FROM users")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "id" in sql
        assert "name" in sql
    
    def test_generate_select_with_alias(self):
        result = SQLParser().parse("SELECT id AS user_id FROM users")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "user_id" in sql
    
    def test_generate_select_distinct(self):
        result = SQLParser().parse("SELECT DISTINCT category FROM products")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "DISTINCT" in sql
    
    def test_generate_qualified_column(self):
        result = SQLParser().parse("SELECT u.name FROM users u")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "u.name" in sql or "name" in sql


# ============================================================
# SECTION 2: SQL GENERATOR - WHERE CLAUSE
# ============================================================

class TestGeneratorWhere:
    """Tests de génération WHERE."""
    
    def test_generate_where_equals(self):
        result = SQLParser().parse("SELECT * FROM users WHERE id = 1")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "WHERE" in sql
        assert "id" in sql
    
    def test_generate_where_string(self):
        result = SQLParser().parse("SELECT * FROM users WHERE status = 'active'")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "'active'" in sql
    
    def test_generate_where_and(self):
        result = SQLParser().parse("SELECT * FROM users WHERE a = 1 AND b = 2")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "AND" in sql


# ============================================================
# SECTION 3: SQL GENERATOR - JOINS
# ============================================================

class TestGeneratorJoins:
    """Tests de génération JOIN."""
    
    def test_generate_inner_join(self):
        result = SQLParser().parse("SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "JOIN" in sql
        assert "ON" in sql
    
    def test_generate_left_join(self):
        result = SQLParser().parse("SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "LEFT" in sql
    
    def test_generate_cross_join(self):
        result = SQLParser().parse("SELECT * FROM a CROSS JOIN b")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "CROSS JOIN" in sql


# ============================================================
# SECTION 4: SQL GENERATOR - GROUP BY, ORDER BY, LIMIT
# ============================================================

class TestGeneratorClauses:
    """Tests des clauses GROUP BY, ORDER BY, LIMIT."""
    
    def test_generate_group_by(self):
        result = SQLParser().parse("SELECT category, COUNT(*) FROM products GROUP BY category")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "GROUP BY" in sql
    
    def test_generate_having(self):
        result = SQLParser().parse("SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 5")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "HAVING" in sql
    
    def test_generate_order_by_asc(self):
        result = SQLParser().parse("SELECT * FROM t ORDER BY created_at ASC")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "ORDER BY" in sql
    
    def test_generate_order_by_desc(self):
        result = SQLParser().parse("SELECT * FROM t ORDER BY created_at DESC")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "DESC" in sql
    
    def test_generate_limit(self):
        result = SQLParser().parse("SELECT * FROM t LIMIT 10")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "LIMIT 10" in sql
    
    def test_generate_limit_offset(self):
        result = SQLParser().parse("SELECT * FROM t LIMIT 10 OFFSET 20")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "LIMIT 10" in sql
        assert "OFFSET 20" in sql


# ============================================================
# SECTION 5: SQL GENERATOR - EXPRESSIONS
# ============================================================

class TestGeneratorExpressions:
    """Tests de génération d'expressions."""
    
    def test_generate_cast(self):
        result = SQLParser().parse("SELECT CAST(value AS INTEGER) FROM t")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "CAST" in sql
        assert "AS" in sql
    
    def test_generate_try_cast_presto(self):
        result = SQLParser().parse("SELECT TRY_CAST(value AS INTEGER) FROM t")
        gen = SQLGenerator(dialect=SQLDialect.PRESTO)
        sql = gen.generate(result.statement)
        assert "TRY_CAST" in sql
    
    def test_generate_safe_cast_bigquery(self):
        result = SQLParser().parse("SELECT TRY_CAST(value AS INTEGER) FROM t")
        gen = SQLGenerator(dialect=SQLDialect.BIGQUERY)
        sql = gen.generate(result.statement)
        assert "SAFE_CAST" in sql
    
    def test_generate_case_expression(self):
        result = SQLParser().parse("SELECT CASE WHEN x = 1 THEN 'one' ELSE 'other' END FROM t")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "CASE" in sql
        assert "WHEN" in sql
        assert "THEN" in sql
        assert "ELSE" in sql
        assert "END" in sql
    
    def test_generate_function_call(self):
        result = SQLParser().parse("SELECT UPPER(name) FROM users")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "UPPER(" in sql
    
    def test_generate_function_distinct(self):
        result = SQLParser().parse("SELECT COUNT(DISTINCT id) FROM users")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "DISTINCT" in sql


# ============================================================
# SECTION 6: SQL GENERATOR - DML
# ============================================================

class TestGeneratorDML:
    """Tests de génération DML."""
    
    def test_generate_insert_values(self):
        result = SQLParser().parse("INSERT INTO users (name) VALUES ('John')")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "INSERT INTO" in sql
        assert "VALUES" in sql
    
    def test_generate_insert_select(self):
        result = SQLParser().parse("INSERT INTO archive SELECT * FROM current_data")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "INSERT INTO" in sql
        assert "SELECT" in sql
    
    def test_generate_update(self):
        result = SQLParser().parse("UPDATE users SET status = 'inactive' WHERE id = 1")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "UPDATE" in sql
        assert "SET" in sql
        assert "WHERE" in sql
    
    def test_generate_delete(self):
        result = SQLParser().parse("DELETE FROM users WHERE id = 1")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "DELETE FROM" in sql
        assert "WHERE" in sql


# ============================================================
# SECTION 7: SQL GENERATOR - DIALECT VARIATIONS
# ============================================================

class TestGeneratorDialects:
    """Tests de génération par dialecte."""
    
    def test_uppercase_keywords_default(self):
        result = SQLParser().parse("SELECT * FROM t")
        gen = SQLGenerator(uppercase_keywords=True)
        sql = gen.generate(result.statement)
        assert "SELECT" in sql
        assert "FROM" in sql
    
    def test_lowercase_keywords(self):
        result = SQLParser().parse("SELECT * FROM t")
        gen = SQLGenerator(uppercase_keywords=False)
        sql = gen.generate(result.statement)
        assert "select" in sql
        assert "from" in sql


# ============================================================
# SECTION 8: TRANSPILER - TSQL TO PRESTO
# ============================================================

class TestTranspilerTSQLToPresto:
    """Tests de transpilation T-SQL vers Presto."""
    
    def test_isnull_to_coalesce(self):
        result = transpile(
            "SELECT ISNULL(name, 'Unknown') FROM users",
            SQLDialect.TSQL,
            SQLDialect.PRESTO
        )
        assert result.success
        assert "COALESCE" in result.sql
    
    def test_getdate_to_current_timestamp(self):
        result = transpile(
            "SELECT GETDATE() FROM dual",
            SQLDialect.TSQL,
            SQLDialect.PRESTO
        )
        assert result.success
        assert "CURRENT_TIMESTAMP" in result.sql
    
    def test_len_to_length(self):
        result = transpile(
            "SELECT LEN(name) FROM users",
            SQLDialect.TSQL,
            SQLDialect.PRESTO
        )
        assert result.success
        assert "LENGTH" in result.sql
    
    def test_dateadd(self):
        result = transpile(
            "SELECT DATEADD(day, 7, created_at) FROM orders",
            SQLDialect.TSQL,
            SQLDialect.PRESTO
        )
        assert result.success
    
    def test_datediff(self):
        result = transpile(
            "SELECT DATEDIFF(day, start_date, end_date) FROM events",
            SQLDialect.TSQL,
            SQLDialect.PRESTO
        )
        assert result.success


# ============================================================
# SECTION 9: TRANSPILER - MYSQL TO PRESTO
# ============================================================

class TestTranspilerMySQLToPresto:
    """Tests de transpilation MySQL vers Presto."""
    
    def test_ifnull_to_coalesce(self):
        result = transpile(
            "SELECT IFNULL(name, 'Unknown') FROM users",
            SQLDialect.MYSQL,
            SQLDialect.PRESTO
        )
        assert result.success
        assert "COALESCE" in result.sql
    
    def test_now_to_current_timestamp(self):
        result = transpile(
            "SELECT NOW() FROM dual",
            SQLDialect.MYSQL,
            SQLDialect.PRESTO
        )
        assert result.success
        assert "CURRENT_TIMESTAMP" in result.sql


# ============================================================
# SECTION 10: TRANSPILER - PRESTO TO BIGQUERY
# ============================================================

class TestTranspilerPrestoToBigQuery:
    """Tests de transpilation Presto vers BigQuery."""
    
    def test_cardinality_to_array_length(self):
        result = transpile(
            "SELECT CARDINALITY(items) FROM t",
            SQLDialect.PRESTO,
            SQLDialect.BIGQUERY
        )
        assert result.success
        assert "ARRAY_LENGTH" in result.sql


# ============================================================
# SECTION 11: ROUNDTRIP TESTS
# ============================================================

class TestRoundtrip:
    """Tests de parsing puis génération."""
    
    def test_roundtrip_simple_select(self):
        original = "SELECT id, name FROM users"
        result = SQLParser().parse(original)
        regenerated = SQLGenerator().generate(result.statement)
        
        # Re-parse the generated SQL
        result2 = SQLParser().parse(regenerated)
        assert result2.tables_referenced == result.tables_referenced
    
    def test_roundtrip_with_where(self):
        original = "SELECT * FROM users WHERE status = 'active'"
        result = SQLParser().parse(original)
        regenerated = SQLGenerator().generate(result.statement)
        
        result2 = SQLParser().parse(regenerated)
        assert result2.statement is not None
    
    def test_roundtrip_with_join(self):
        original = "SELECT * FROM a INNER JOIN b ON a.id = b.a_id"
        result = SQLParser().parse(original)
        regenerated = SQLGenerator().generate(result.statement)
        
        result2 = SQLParser().parse(regenerated)
        assert "a" in result2.tables_referenced
        assert "b" in result2.tables_referenced
    
    def test_roundtrip_update(self):
        original = "UPDATE users SET status = 'inactive'"
        result = SQLParser().parse(original)
        regenerated = SQLGenerator().generate(result.statement)
        
        result2 = SQLParser().parse(regenerated)
        from sql_parser.ast_nodes import UpdateStatement
        assert isinstance(result2.statement, UpdateStatement)
    
    def test_roundtrip_complex_query(self):
        original = """
            SELECT u.id, u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.status = 'active'
            GROUP BY u.id, u.name
            HAVING COUNT(o.id) > 5
            ORDER BY order_count DESC
            LIMIT 10
        """
        result = SQLParser().parse(original)
        regenerated = SQLGenerator().generate(result.statement)
        
        result2 = SQLParser().parse(regenerated)
        assert result2.has_aggregation


# ============================================================
# SECTION 12: ERROR HANDLING
# ============================================================

class TestTranspilerErrors:
    """Tests de gestion des erreurs de transpilation."""
    
    def test_invalid_sql(self):
        result = transpile("SELECT FROM", SQLDialect.TSQL, SQLDialect.PRESTO)
        assert not result.success
    
    def test_empty_sql(self):
        result = transpile("", SQLDialect.TSQL, SQLDialect.PRESTO)
        assert not result.success


# ============================================================
# SECTION 13: WINDOW FUNCTIONS
# ============================================================

class TestGeneratorWindowFunctions:
    """Tests de génération des window functions."""
    
    def test_generate_row_number(self):
        result = SQLParser().parse("SELECT ROW_NUMBER() OVER (PARTITION BY cat ORDER BY created_at DESC) FROM t")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "ROW_NUMBER()" in sql
        assert "OVER" in sql
        assert "PARTITION BY" in sql
        assert "ORDER BY" in sql
    
    def test_generate_sum_over(self):
        result = SQLParser().parse("SELECT SUM(amount) OVER (ORDER BY created_at) FROM orders")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "SUM(" in sql
        assert "OVER" in sql


# ============================================================
# SECTION 14: PRESTO/ATHENA SPECIFIC
# ============================================================

class TestGeneratorPresto:
    """Tests de génération spécifique Presto/Athena."""
    
    def test_generate_array(self):
        result = SQLParser().parse("SELECT ARRAY[1, 2, 3]")
        gen = SQLGenerator(dialect=SQLDialect.PRESTO)
        sql = gen.generate(result.statement)
        assert "ARRAY[" in sql
    
    def test_generate_lambda(self):
        result = SQLParser().parse("SELECT TRANSFORM(arr, x -> x * 2) FROM t")
        gen = SQLGenerator(dialect=SQLDialect.PRESTO)
        sql = gen.generate(result.statement)
        assert "TRANSFORM" in sql
        assert "->" in sql


# ============================================================
# SECTION 15: TRANSPILER FULL QUERY
# ============================================================

class TestTranspilerFullQueries:
    """Tests de transpilation de requêtes complètes."""
    
    def test_complex_tsql_to_presto(self):
        sql = """
            SELECT 
                ISNULL(u.name, 'Anonymous') as name,
                LEN(u.email) as email_length,
                GETDATE() as current_time
            FROM users u
        """
        result = transpile(sql, SQLDialect.TSQL, SQLDialect.PRESTO)
        assert result.success
        assert "COALESCE" in result.sql
        assert "LENGTH" in result.sql
        assert "CURRENT_TIMESTAMP" in result.sql
    
    def test_presto_to_bigquery(self):
        sql = """
            SELECT 
                id,
                CARDINALITY(items) as item_count
            FROM orders
        """
        result = transpile(sql, SQLDialect.PRESTO, SQLDialect.BIGQUERY)
        assert result.success
        assert "ARRAY_LENGTH" in result.sql


# ============================================================
# SECTION 16: CTE GENERATION
# ============================================================

class TestGeneratorCTE:
    """Tests de génération de CTEs."""
    
    def test_generate_simple_cte(self):
        result = SQLParser().parse("""
            WITH cte1 AS (SELECT 1 as a)
            SELECT * FROM cte1
        """)
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "WITH" in sql
        assert "cte1" in sql
    
    def test_generate_multiple_ctes(self):
        result = SQLParser().parse("""
            WITH cte1 AS (SELECT 1 as a), cte2 AS (SELECT 2 as b)
            SELECT * FROM cte1, cte2
        """)
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "cte1" in sql
        assert "cte2" in sql


# ============================================================
# SECTION 17: SUBQUERY GENERATION
# ============================================================

class TestGeneratorSubquery:
    """Tests de génération de sous-requêtes."""
    
    def test_generate_subquery_in_from(self):
        result = SQLParser().parse("SELECT * FROM (SELECT id FROM users) sub")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "SELECT" in sql
        assert "sub" in sql
    
    def test_generate_subquery_in_where(self):
        result = SQLParser().parse("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "IN" in sql


# ============================================================
# SECTION 18: DDL GENERATION
# ============================================================

class TestGeneratorDDL:
    """Tests de génération DDL."""
    
    def test_generate_create_table(self):
        result = SQLParser().parse("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(100))")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "CREATE TABLE" in sql
        assert "id" in sql
        assert "name" in sql
    
    def test_generate_create_view(self):
        result = SQLParser().parse("CREATE VIEW v AS SELECT * FROM t")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "CREATE VIEW" in sql
    
    def test_generate_drop_table(self):
        result = SQLParser().parse("DROP TABLE users")
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "DROP TABLE" in sql


# ============================================================
# SECTION 19: MERGE GENERATION
# ============================================================

class TestGeneratorMerge:
    """Tests de génération MERGE."""
    
    def test_generate_merge(self):
        result = SQLParser().parse("""
            MERGE INTO target_table t
            USING source_table s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.name = s.name
            WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
        """)
        gen = SQLGenerator()
        sql = gen.generate(result.statement)
        assert "MERGE INTO" in sql
        assert "USING" in sql
        assert "WHEN MATCHED" in sql


# ============================================================
# SECTION 20: DIALECT FEATURE GENERATION
# ============================================================

class TestDialectFeatures:
    """Tests de génération des fonctionnalités par dialecte."""
    
    def test_tsql_top(self):
        result = SQLParser().parse("SELECT TOP 10 * FROM users")
        gen = SQLGenerator(dialect=SQLDialect.TSQL)
        sql = gen.generate(result.statement)
        assert "TOP" in sql
    
    def test_presto_try(self):
        result = SQLParser().parse("SELECT TRY(1/0) FROM t")
        gen = SQLGenerator(dialect=SQLDialect.PRESTO)
        sql = gen.generate(result.statement)
        assert "TRY" in sql
    
    def test_bigquery_struct(self):
        result = SQLParser().parse("SELECT STRUCT(1, 2)")
        gen = SQLGenerator(dialect=SQLDialect.BIGQUERY)
        sql = gen.generate(result.statement)
        assert "STRUCT" in sql
