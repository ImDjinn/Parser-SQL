"""
Tests des noeuds AST et du parsing.
Ces tests utilisent l'approche de parsing pour vérifier que les structures AST sont correctement générées.
"""

import pytest
import sys
sys.path.insert(0, '..')

from sql_parser import SQLParser
from sql_parser.ast_nodes import (
    SelectStatement, InsertStatement, UpdateStatement, DeleteStatement, MergeStatement,
    CreateTableStatement, CreateViewStatement, DropStatement,
    Literal, Identifier, BinaryOp, FunctionCall, CaseExpression, Star,
    TableRef, JoinClause
)


# ============================================================
# SECTION 1: LITERALS
# ============================================================

class TestLiteralsParsing:
    """Tests des littéraux via parsing."""
    
    def test_integer_literal(self):
        result = SQLParser().parse("SELECT 42")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, Literal)
        assert expr.value == 42
    
    def test_negative_integer(self):
        result = SQLParser().parse("SELECT -123")
        assert result.statement is not None
    
    def test_float_literal(self):
        result = SQLParser().parse("SELECT 3.14")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, Literal)
        assert expr.value == 3.14
    
    def test_string_literal(self):
        result = SQLParser().parse("SELECT 'hello world'")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, Literal)
        assert expr.value == "hello world"
    
    def test_null_literal(self):
        result = SQLParser().parse("SELECT NULL")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, Literal)
        assert expr.value is None
    
    def test_true_literal(self):
        result = SQLParser().parse("SELECT TRUE")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, Literal)
        assert expr.value is True
    
    def test_false_literal(self):
        result = SQLParser().parse("SELECT FALSE")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, Literal)
        assert expr.value is False


# ============================================================
# SECTION 2: IDENTIFIERS AND COLUMN REFERENCES
# ============================================================

class TestColumnRefParsing:
    """Tests des références de colonnes via parsing."""
    
    def test_simple_column(self):
        result = SQLParser().parse("SELECT name FROM users")
        assert result.statement is not None
        assert "name" in result.columns_referenced
    
    def test_qualified_column(self):
        result = SQLParser().parse("SELECT u.name FROM users u")
        assert result.statement is not None
    
    def test_fully_qualified_column(self):
        result = SQLParser().parse("SELECT schema_name.table_name.column_name FROM schema_name.table_name")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert expr.column == "column_name"
        assert expr.table == "table_name"
        assert expr.schema == "schema_name"


# ============================================================
# SECTION 3: BINARY OPERATIONS
# ============================================================

class TestBinaryOpParsing:
    """Tests des opérations binaires via parsing."""
    
    def test_addition(self):
        result = SQLParser().parse("SELECT a + b FROM t")
        assert result.statement is not None
        expr = result.statement.select_items[0].expression
        assert isinstance(expr, BinaryOp)
        assert expr.operator == "+"
    
    def test_subtraction(self):
        result = SQLParser().parse("SELECT a - b FROM t")
        assert result.statement is not None
    
    def test_multiplication(self):
        result = SQLParser().parse("SELECT a * b FROM t")
        assert result.statement is not None
    
    def test_division(self):
        result = SQLParser().parse("SELECT a / b FROM t")
        assert result.statement is not None
    
    def test_modulo(self):
        result = SQLParser().parse("SELECT a % b FROM t")
        assert result.statement is not None
    
    def test_comparison_equals(self):
        result = SQLParser().parse("SELECT * FROM t WHERE a = 1")
        assert result.statement is not None
    
    def test_comparison_not_equals(self):
        result = SQLParser().parse("SELECT * FROM t WHERE a <> 1")
        assert result.statement is not None
    
    def test_comparison_less_than(self):
        result = SQLParser().parse("SELECT * FROM t WHERE a < 1")
        assert result.statement is not None
    
    def test_comparison_greater_than(self):
        result = SQLParser().parse("SELECT * FROM t WHERE a > 1")
        assert result.statement is not None
    
    def test_logical_and(self):
        result = SQLParser().parse("SELECT * FROM t WHERE a = 1 AND b = 2")
        assert result.statement is not None
    
    def test_logical_or(self):
        result = SQLParser().parse("SELECT * FROM t WHERE a = 1 OR b = 2")
        assert result.statement is not None


# ============================================================
# SECTION 4: FUNCTION CALLS
# ============================================================

class TestFunctionCallParsing:
    """Tests des appels de fonction via parsing."""
    
    def test_simple_function(self):
        result = SQLParser().parse("SELECT UPPER(name) FROM users")
        assert result.statement is not None
        assert "UPPER" in result.functions_used
    
    def test_function_multiple_args(self):
        result = SQLParser().parse("SELECT COALESCE(a, b, c) FROM t")
        assert result.statement is not None
    
    def test_function_no_args(self):
        result = SQLParser().parse("SELECT NOW()")
        assert result.statement is not None
    
    def test_nested_functions(self):
        result = SQLParser().parse("SELECT UPPER(TRIM(name)) FROM users")
        assert result.statement is not None
    
    def test_aggregate_count(self):
        result = SQLParser().parse("SELECT COUNT(*) FROM users")
        assert result.statement is not None
        assert result.has_aggregation
    
    def test_aggregate_sum(self):
        result = SQLParser().parse("SELECT SUM(amount) FROM orders")
        assert result.statement is not None
        assert result.has_aggregation
    
    def test_aggregate_avg(self):
        result = SQLParser().parse("SELECT AVG(price) FROM products")
        assert result.statement is not None
        assert result.has_aggregation
    
    def test_aggregate_min_max(self):
        result = SQLParser().parse("SELECT MIN(price), MAX(price) FROM products")
        assert result.statement is not None
        assert result.has_aggregation
    
    def test_count_distinct(self):
        result = SQLParser().parse("SELECT COUNT(DISTINCT category) FROM products")
        assert result.statement is not None
        assert result.has_aggregation


# ============================================================
# SECTION 5: CASE EXPRESSION
# ============================================================

class TestCaseExpressionParsing:
    """Tests des expressions CASE via parsing."""
    
    def test_simple_case(self):
        result = SQLParser().parse("""
            SELECT CASE status
                WHEN 1 THEN 'active'
                WHEN 2 THEN 'inactive'
                ELSE 'unknown'
            END FROM users
        """)
        assert result.statement is not None
    
    def test_searched_case(self):
        result = SQLParser().parse("""
            SELECT CASE
                WHEN age < 18 THEN 'minor'
                WHEN age >= 18 THEN 'adult'
            END FROM users
        """)
        assert result.statement is not None
    
    def test_case_no_else(self):
        result = SQLParser().parse("""
            SELECT CASE WHEN x > 0 THEN 'positive' END FROM t
        """)
        assert result.statement is not None
    
    def test_nested_case(self):
        result = SQLParser().parse("""
            SELECT CASE
                WHEN a = 1 THEN CASE WHEN b = 1 THEN 'both' ELSE 'only a' END
                ELSE 'none'
            END FROM t
        """)
        assert result.statement is not None


# ============================================================
# SECTION 6: WINDOW FUNCTIONS
# ============================================================

class TestWindowFunctionParsing:
    """Tests des window functions via parsing."""
    
    def test_row_number(self):
        result = SQLParser().parse("""
            SELECT ROW_NUMBER() OVER (ORDER BY id) FROM users
        """)
        assert result.statement is not None
    
    def test_row_number_partition(self):
        result = SQLParser().parse("""
            SELECT ROW_NUMBER() OVER (PARTITION BY category ORDER BY created_at DESC) FROM products
        """)
        assert result.statement is not None
    
    def test_rank(self):
        result = SQLParser().parse("""
            SELECT RANK() OVER (ORDER BY score DESC) FROM players
        """)
        assert result.statement is not None
    
    def test_dense_rank(self):
        result = SQLParser().parse("""
            SELECT DENSE_RANK() OVER (ORDER BY score DESC) FROM players
        """)
        assert result.statement is not None
    
    def test_lead(self):
        result = SQLParser().parse("""
            SELECT LEAD(amount, 1) OVER (ORDER BY created_at) FROM orders
        """)
        assert result.statement is not None
    
    def test_lag(self):
        result = SQLParser().parse("""
            SELECT LAG(amount, 1) OVER (ORDER BY created_at) FROM orders
        """)
        assert result.statement is not None
    
    def test_sum_over(self):
        result = SQLParser().parse("""
            SELECT SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at) FROM orders
        """)
        assert result.statement is not None
    
    def test_window_frame_rows(self):
        result = SQLParser().parse("""
            SELECT SUM(amount) OVER (
                ORDER BY created_at 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) FROM orders
        """)
        assert result.statement is not None
    
    def test_window_frame_range(self):
        result = SQLParser().parse("""
            SELECT AVG(price) OVER (
                ORDER BY created_at
                RANGE BETWEEN 7 PRECEDING AND CURRENT ROW
            ) FROM products
        """)
        assert result.statement is not None


# ============================================================
# SECTION 7: SUBQUERIES
# ============================================================

class TestSubqueryParsing:
    """Tests des sous-requêtes via parsing."""
    
    def test_scalar_subquery(self):
        result = SQLParser().parse("""
            SELECT (SELECT COUNT(*) FROM orders) as total FROM dual
        """)
        assert result.has_subquery
    
    def test_subquery_in_from(self):
        result = SQLParser().parse("""
            SELECT * FROM (SELECT id, name FROM users) sub
        """)
        assert result.has_subquery
    
    def test_subquery_in_where(self):
        result = SQLParser().parse("""
            SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)
        """)
        assert result.has_subquery
    
    def test_exists_subquery(self):
        result = SQLParser().parse("""
            SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)
        """)
        assert result.has_subquery
    
    def test_correlated_subquery(self):
        result = SQLParser().parse("""
            SELECT *, (SELECT MAX(amount) FROM orders o WHERE o.user_id = u.id) 
            FROM users u
        """)
        assert result.has_subquery


# ============================================================
# SECTION 8: CTEs
# ============================================================

class TestCTEParsing:
    """Tests des CTEs via parsing."""
    
    def test_simple_cte(self):
        result = SQLParser().parse("""
            WITH active_users AS (
                SELECT * FROM users WHERE status = 'active'
            )
            SELECT * FROM active_users
        """)
        assert result.statement is not None
        assert result.statement.ctes is not None
        assert len(result.statement.ctes) == 1
    
    def test_multiple_ctes(self):
        result = SQLParser().parse("""
            WITH 
                cte1 AS (SELECT 1 as a),
                cte2 AS (SELECT 2 as b)
            SELECT * FROM cte1, cte2
        """)
        assert result.statement is not None
        assert len(result.statement.ctes) == 2
    
    def test_recursive_cte(self):
        result = SQLParser().parse("""
            WITH RECURSIVE nums AS (
                SELECT 1 as n
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 10
            )
            SELECT * FROM nums
        """)
        assert result.statement is not None


# ============================================================
# SECTION 9: JOINS
# ============================================================

class TestJoinParsing:
    """Tests des JOINs via parsing."""
    
    def test_inner_join(self):
        result = SQLParser().parse("""
            SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id
        """)
        assert "users" in result.tables_referenced
        assert "orders" in result.tables_referenced
    
    def test_left_join(self):
        result = SQLParser().parse("""
            SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id
        """)
        assert result.statement is not None
    
    def test_right_join(self):
        result = SQLParser().parse("""
            SELECT * FROM users u RIGHT JOIN orders o ON u.id = o.user_id
        """)
        assert result.statement is not None
    
    def test_full_outer_join(self):
        result = SQLParser().parse("""
            SELECT * FROM a FULL OUTER JOIN b ON a.id = b.a_id
        """)
        assert result.statement is not None
    
    def test_cross_join(self):
        result = SQLParser().parse("""
            SELECT * FROM a CROSS JOIN b
        """)
        assert result.statement is not None
    
    def test_join_using(self):
        result = SQLParser().parse("""
            SELECT * FROM a JOIN b USING (id)
        """)
        assert result.statement is not None
    
    def test_multiple_joins(self):
        result = SQLParser().parse("""
            SELECT * FROM a
            JOIN b ON a.id = b.a_id
            JOIN c ON b.id = c.b_id
            LEFT JOIN d ON c.id = d.c_id
        """)
        assert len(result.tables_referenced) == 4


# ============================================================
# SECTION 10: INSERT STATEMENTS
# ============================================================

class TestInsertParsing:
    """Tests des INSERT via parsing."""
    
    def test_insert_values(self):
        result = SQLParser().parse("""
            INSERT INTO users (id, name) VALUES (1, 'John')
        """)
        assert isinstance(result.statement, InsertStatement)
    
    def test_insert_multiple_rows(self):
        result = SQLParser().parse("""
            INSERT INTO users (id, name) VALUES (1, 'John'), (2, 'Jane')
        """)
        assert isinstance(result.statement, InsertStatement)
    
    def test_insert_select(self):
        result = SQLParser().parse("""
            INSERT INTO archive SELECT * FROM current_data
        """)
        assert isinstance(result.statement, InsertStatement)
    
    def test_insert_with_columns(self):
        result = SQLParser().parse("""
            INSERT INTO users (name, email) SELECT name, email FROM staging
        """)
        assert isinstance(result.statement, InsertStatement)


# ============================================================
# SECTION 11: UPDATE STATEMENTS
# ============================================================

class TestUpdateParsing:
    """Tests des UPDATE via parsing."""
    
    def test_simple_update(self):
        result = SQLParser().parse("""
            UPDATE users SET status = 'inactive' WHERE id = 1
        """)
        assert isinstance(result.statement, UpdateStatement)
    
    def test_update_multiple_columns(self):
        result = SQLParser().parse("""
            UPDATE users SET name = 'John', email = 'john@example.com' WHERE id = 1
        """)
        assert isinstance(result.statement, UpdateStatement)
    
    def test_update_with_expression(self):
        result = SQLParser().parse("""
            UPDATE products SET price = price * 1.1 WHERE category = 'electronics'
        """)
        assert isinstance(result.statement, UpdateStatement)


# ============================================================
# SECTION 12: DELETE STATEMENTS
# ============================================================

class TestDeleteParsing:
    """Tests des DELETE via parsing."""
    
    def test_delete_with_where(self):
        result = SQLParser().parse("""
            DELETE FROM users WHERE status = 'deleted'
        """)
        assert isinstance(result.statement, DeleteStatement)
    
    def test_delete_all(self):
        result = SQLParser().parse("""
            DELETE FROM staging_table
        """)
        assert isinstance(result.statement, DeleteStatement)


# ============================================================
# SECTION 13: MERGE STATEMENTS
# ============================================================

class TestMergeParsing:
    """Tests des MERGE via parsing."""
    
    def test_merge_basic(self):
        result = SQLParser().parse("""
            MERGE INTO target_table t
            USING source_table s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.name = s.name
            WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
        """)
        assert isinstance(result.statement, MergeStatement)


# ============================================================
# SECTION 14: DDL STATEMENTS
# ============================================================

class TestDDLParsing:
    """Tests des DDL via parsing."""
    
    def test_create_table(self):
        result = SQLParser().parse("""
            CREATE TABLE users (
                id INT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) UNIQUE
            )
        """)
        assert isinstance(result.statement, CreateTableStatement)
    
    def test_create_table_if_not_exists(self):
        result = SQLParser().parse("""
            CREATE TABLE IF NOT EXISTS logs (id INT)
        """)
        assert isinstance(result.statement, CreateTableStatement)
    
    def test_create_view(self):
        result = SQLParser().parse("""
            CREATE VIEW active_users AS
            SELECT * FROM users WHERE status = 'active'
        """)
        assert isinstance(result.statement, CreateViewStatement)
    
    def test_create_or_replace_view(self):
        result = SQLParser().parse("""
            CREATE OR REPLACE VIEW user_summary AS
            SELECT id, name FROM users
        """)
        assert isinstance(result.statement, CreateViewStatement)
    
    def test_drop_table(self):
        result = SQLParser().parse("""
            DROP TABLE users
        """)
        assert isinstance(result.statement, DropStatement)
    
    def test_drop_table_if_exists(self):
        result = SQLParser().parse("""
            DROP TABLE IF EXISTS logs
        """)
        assert isinstance(result.statement, DropStatement)
    
    def test_drop_view(self):
        result = SQLParser().parse("""
            DROP VIEW user_summary
        """)
        assert isinstance(result.statement, DropStatement)


# ============================================================
# SECTION 15: PRESTO/ATHENA SPECIFIC
# ============================================================

class TestPrestoFeatures:
    """Tests des fonctionnalités spécifiques Presto/Athena."""
    
    def test_array_constructor(self):
        result = SQLParser().parse("SELECT ARRAY[1, 2, 3]")
        assert result.statement is not None
    
    def test_array_subscript(self):
        result = SQLParser().parse("SELECT arr[1] FROM t")
        assert result.statement is not None
    
    def test_map_constructor(self):
        result = SQLParser().parse("SELECT MAP(ARRAY['a', 'b'], ARRAY[1, 2])")
        assert result.statement is not None
    
    def test_unnest(self):
        result = SQLParser().parse("SELECT * FROM UNNEST(ARRAY[1, 2, 3]) AS t(x)")
        assert result.statement is not None
    
    def test_lambda_single(self):
        result = SQLParser().parse("SELECT TRANSFORM(arr, x -> x * 2) FROM t")
        assert result.statement is not None
    
    def test_lambda_multi_param(self):
        result = SQLParser().parse("SELECT REDUCE(arr, 0, (s, x) -> s + x, s -> s) FROM t")
        assert result.statement is not None
    
    def test_try_expression(self):
        result = SQLParser().parse("SELECT TRY(1/0) FROM t")
        assert result.statement is not None
    
    def test_try_cast(self):
        result = SQLParser().parse("SELECT TRY_CAST(col AS INTEGER) FROM t")
        assert result.statement is not None
    
    def test_if_expression(self):
        result = SQLParser().parse("SELECT IF(condition, 1, 0) FROM t")
        assert result.statement is not None
    
    def test_interval(self):
        result = SQLParser().parse("SELECT INTERVAL '1' DAY")
        assert result.statement is not None


# ============================================================
# SECTION 16: T-SQL SPECIFIC
# ============================================================

class TestTSQLFeatures:
    """Tests des fonctionnalités spécifiques T-SQL."""
    
    def test_top(self):
        result = SQLParser().parse("SELECT TOP 10 * FROM users")
        assert result.statement is not None
    
    def test_top_percent(self):
        result = SQLParser().parse("SELECT TOP 10 PERCENT * FROM users")
        assert result.statement is not None
    
    def test_bracket_identifiers(self):
        result = SQLParser().parse("SELECT [user name] FROM [my table]")
        assert result.statement is not None
    
    def test_nolock_hint(self):
        result = SQLParser().parse("SELECT * FROM users WITH (NOLOCK)")
        assert result.statement is not None


# ============================================================
# SECTION 17: JINJA TEMPLATES
# ============================================================

class TestJinjaTemplates:
    """Tests du support Jinja."""
    
    def test_jinja_variable(self):
        result = SQLParser().parse("SELECT * FROM users WHERE id = {{ user_id }}")
        assert result.statement is not None
    
    def test_jinja_ref(self):
        result = SQLParser().parse("SELECT * FROM {{ ref('users') }}")
        assert result.statement is not None
    
    def test_jinja_var(self):
        result = SQLParser().parse("SELECT * FROM users WHERE status = {{ var('status') }}")
        assert result.statement is not None
    
    def test_jinja_config(self):
        result = SQLParser().parse("""
            {{ config(materialized='table') }}
            SELECT * FROM users
        """)
        assert result.statement is not None
    
    def test_jinja_if_block(self):
        result = SQLParser().parse("""
            SELECT * FROM users
            {% if condition %}
            WHERE status = 'active'
            {% endif %}
        """)
        assert result.statement is not None


# ============================================================
# SECTION 18: SET OPERATIONS
# ============================================================

class TestSetOperations:
    """Tests des opérations ensemblistes."""
    
    def test_union(self):
        result = SQLParser().parse("SELECT 1 UNION SELECT 2")
        assert result.statement is not None
    
    def test_union_all(self):
        result = SQLParser().parse("SELECT 1 UNION ALL SELECT 2")
        assert result.statement is not None
    
    def test_intersect(self):
        result = SQLParser().parse("SELECT 1 INTERSECT SELECT 1")
        assert result.statement is not None
    
    def test_except(self):
        result = SQLParser().parse("SELECT 1 EXCEPT SELECT 2")
        assert result.statement is not None


# ============================================================
# SECTION 19: TO_DICT OUTPUT
# ============================================================

class TestToDictOutput:
    """Tests de l'export to_dict."""
    
    def test_select_to_dict(self):
        result = SQLParser().parse("SELECT id, name FROM users")
        d = result.to_dict()
        assert "statement" in d
        assert d["statement"]["node_type"] == "SelectStatement"
    
    def test_insert_to_dict(self):
        result = SQLParser().parse("INSERT INTO users (id) VALUES (1)")
        d = result.to_dict()
        assert "statement" in d
    
    def test_update_to_dict(self):
        result = SQLParser().parse("UPDATE users SET name = 'test'")
        d = result.to_dict()
        assert "statement" in d
    
    def test_delete_to_dict(self):
        result = SQLParser().parse("DELETE FROM users WHERE id = 1")
        d = result.to_dict()
        assert "statement" in d


# ============================================================
# SECTION 20: EDGE CASES
# ============================================================

class TestEdgeCases:
    """Tests des cas limites."""
    
    def test_very_long_query(self):
        columns = ", ".join([f"col{i}" for i in range(100)])
        sql = f"SELECT {columns} FROM big_table"
        result = SQLParser().parse(sql)
        assert result.statement is not None
    
    def test_deeply_nested_subqueries(self):
        result = SQLParser().parse("""
            SELECT * FROM (
                SELECT * FROM (
                    SELECT * FROM (
                        SELECT id FROM users
                    ) t1
                ) t2
            ) t3
        """)
        assert result.has_subquery
    
    def test_complex_expression(self):
        result = SQLParser().parse("""
            SELECT 
                CASE 
                    WHEN a > 0 AND b < 10 THEN (a + b) * 2
                    WHEN c IS NOT NULL THEN c / 2
                    ELSE COALESCE(d, 0)
                END
            FROM t
        """)
        assert result.statement is not None
    
    def test_unicode_in_strings(self):
        result = SQLParser().parse("SELECT '日本語' FROM t")
        assert result.statement is not None
    
    def test_empty_table_alias(self):
        result = SQLParser().parse("SELECT * FROM users")
        assert result.statement is not None
    
    def test_multiple_statements_semicolon(self):
        result = SQLParser().parse("SELECT 1; SELECT 2")
        assert result.statement is not None
    
    def test_comments_single_line(self):
        result = SQLParser().parse("""
            -- This is a comment
            SELECT * FROM users
        """)
        assert result.statement is not None
    
    def test_comments_multiline(self):
        result = SQLParser().parse("""
            /* This is a 
               multiline comment */
            SELECT * FROM users
        """)
        assert result.statement is not None
