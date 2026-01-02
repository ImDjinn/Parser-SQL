"""
Tests pour les nouvelles fonctionnalités ajoutées:
- catalog.schema.table (3 niveaux)
- VALUES comme statement autonome
- CROSS APPLY / OUTER APPLY
- EXPLAIN statement
- VACUUM statement
- GRANT/REVOKE statements
- SQL Formatter
"""

import pytest
from sql_parser.parser import parse
from sql_parser.sql_generator import generate_sql
from sql_parser.formatter import format_sql, minify_sql, validate_sql, FormatStyle, SQLFormatter
from sql_parser.ast_nodes import (
    ValuesStatement, ExplainStatement, VacuumStatement, 
    GrantStatement, RevokeStatement, TableRef, JoinType
)


# ============== Tests catalog.schema.table (3 niveaux) ==============

class TestCatalogSchemaTable:
    """Tests pour le support des noms de tables à 3 niveaux."""
    
    def test_three_level_table_name(self):
        """Test parsing catalog.schema.table."""
        sql = "SELECT * FROM hive_metastore.default.sales"
        result = parse(sql)
        
        table = result.statement.from_clause.tables[0]
        assert table.catalog == "hive_metastore"
        assert table.schema == "default"
        assert table.name == "sales"
    
    def test_three_level_table_roundtrip(self):
        """Test round-trip pour catalog.schema.table."""
        sql = "SELECT * FROM catalog.schema.table_name"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "catalog.schema.table_name" in regenerated
    
    def test_three_level_with_alias(self):
        """Test catalog.schema.table avec alias."""
        sql = "SELECT s.id FROM hive_metastore.default.sales s"
        result = parse(sql)
        
        table = result.statement.from_clause.tables[0]
        assert table.catalog == "hive_metastore"
        assert table.alias == "s"
    
    def test_mixed_level_joins(self):
        """Test JOIN avec différents niveaux de qualification."""
        sql = """
        SELECT a.id, b.name 
        FROM catalog.schema.table1 a 
        JOIN schema.table2 b ON a.id = b.id
        """
        result = parse(sql)
        assert result.statement is not None
        
    def test_databricks_style(self):
        """Test style Databricks avec hive_metastore."""
        sql = "SELECT * FROM hive_metastore.bronze.raw_events"
        result = parse(sql)
        
        table = result.statement.from_clause.tables[0]
        assert table.catalog == "hive_metastore"
        assert table.schema == "bronze"
        assert table.name == "raw_events"


# ============== Tests VALUES statement ==============

class TestValuesStatement:
    """Tests pour VALUES comme statement autonome."""
    
    def test_simple_values(self):
        """Test VALUES simple."""
        sql = "VALUES (1, 2, 3)"
        result = parse(sql)
        
        assert isinstance(result.statement, ValuesStatement)
        assert len(result.statement.rows) == 1
        assert len(result.statement.rows[0]) == 3
    
    def test_multiple_rows(self):
        """Test VALUES avec plusieurs lignes."""
        sql = "VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        result = parse(sql)
        
        assert isinstance(result.statement, ValuesStatement)
        assert len(result.statement.rows) == 3
    
    def test_values_roundtrip(self):
        """Test round-trip VALUES."""
        sql = "VALUES (1, 'hello'), (2, 'world')"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "VALUES" in regenerated
        assert "(1, 'hello')" in regenerated
        assert "(2, 'world')" in regenerated
    
    def test_values_with_expressions(self):
        """Test VALUES avec expressions."""
        sql = "VALUES (1 + 2, 'test'), (3 * 4, NULL)"
        result = parse(sql)
        
        assert isinstance(result.statement, ValuesStatement)
        assert len(result.statement.rows) == 2


# ============== Tests CROSS APPLY / OUTER APPLY ==============

class TestApplyJoins:
    """Tests pour CROSS APPLY et OUTER APPLY."""
    
    def test_cross_apply(self):
        """Test CROSS APPLY basique."""
        sql = "SELECT u.name, o.total FROM users u CROSS APPLY get_orders(u.id) o"
        result = parse(sql)
        
        join = result.statement.from_clause.joins[0]
        assert join.join_type == JoinType.CROSS_APPLY
    
    def test_outer_apply(self):
        """Test OUTER APPLY basique."""
        sql = "SELECT u.name, o.total FROM users u OUTER APPLY get_orders(u.id) o"
        result = parse(sql)
        
        join = result.statement.from_clause.joins[0]
        assert join.join_type == JoinType.OUTER_APPLY
    
    def test_cross_apply_roundtrip(self):
        """Test round-trip CROSS APPLY."""
        sql = "SELECT * FROM users u CROSS APPLY fn(u.id) f"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "CROSS APPLY" in regenerated
    
    def test_outer_apply_roundtrip(self):
        """Test round-trip OUTER APPLY."""
        sql = "SELECT * FROM users u OUTER APPLY fn(u.id) f"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "OUTER APPLY" in regenerated
    
    def test_apply_with_subquery(self):
        """Test APPLY avec sous-requête."""
        sql = """
        SELECT u.name, top_order.amount
        FROM users u
        CROSS APPLY (
            SELECT TOP 1 amount FROM orders WHERE user_id = u.id ORDER BY amount DESC
        ) top_order
        """
        result = parse(sql)
        assert result.statement is not None


# ============== Tests TOP (T-SQL) ==============

class TestTopStatement:
    """Tests pour SELECT TOP (T-SQL)."""
    
    def test_simple_top(self):
        """Test TOP simple."""
        sql = "SELECT TOP 10 * FROM users"
        result = parse(sql)
        
        assert result.statement.top is not None
    
    def test_top_with_columns(self):
        """Test TOP avec colonnes spécifiques."""
        sql = "SELECT TOP 5 id, name, email FROM users"
        result = parse(sql)
        
        assert result.statement.top is not None
        assert len(result.statement.select_items) == 3
    
    def test_top_percent(self):
        """Test TOP n PERCENT."""
        sql = "SELECT TOP 10 PERCENT * FROM employees"
        result = parse(sql)
        
        assert result.statement.top is not None
        assert result.statement.top_percent is True
    
    def test_top_with_ties(self):
        """Test TOP n WITH TIES."""
        sql = "SELECT TOP 5 WITH TIES salary FROM employees ORDER BY salary DESC"
        result = parse(sql)
        
        assert result.statement.top is not None
        assert result.statement.top_with_ties is True
    
    def test_top_parentheses(self):
        """Test TOP (n) avec parenthèses."""
        sql = "SELECT TOP (100) * FROM products"
        result = parse(sql)
        
        assert result.statement.top is not None
    
    def test_top_roundtrip(self):
        """Test round-trip TOP."""
        sql = "SELECT TOP 10 id, name FROM users"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "TOP" in regenerated
        assert "10" in regenerated
    
    def test_top_percent_roundtrip(self):
        """Test round-trip TOP PERCENT."""
        sql = "SELECT TOP 5 PERCENT * FROM employees"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "TOP" in regenerated
        assert "PERCENT" in regenerated
    
    def test_top_with_where(self):
        """Test TOP avec WHERE."""
        sql = "SELECT TOP 10 * FROM orders WHERE status = 'active'"
        result = parse(sql)
        
        assert result.statement.top is not None
        assert result.statement.where_clause is not None


# ============== Tests EXPLAIN statement ==============

class TestExplainStatement:
    """Tests pour le statement EXPLAIN."""
    
    def test_simple_explain(self):
        """Test EXPLAIN simple."""
        sql = "EXPLAIN SELECT * FROM users"
        result = parse(sql)
        
        assert isinstance(result.statement, ExplainStatement)
        assert result.statement.analyze is False
    
    def test_explain_analyze(self):
        """Test EXPLAIN ANALYZE."""
        sql = "EXPLAIN ANALYZE SELECT * FROM users"
        result = parse(sql)
        
        assert isinstance(result.statement, ExplainStatement)
        assert result.statement.analyze is True
    
    def test_explain_format(self):
        """Test EXPLAIN avec FORMAT."""
        sql = "EXPLAIN FORMAT JSON SELECT * FROM users"
        result = parse(sql)
        
        assert isinstance(result.statement, ExplainStatement)
        assert result.statement.format == "JSON"
    
    def test_explain_roundtrip(self):
        """Test round-trip EXPLAIN."""
        sql = "EXPLAIN ANALYZE SELECT * FROM orders WHERE total > 100"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "EXPLAIN" in regenerated
        assert "ANALYZE" in regenerated
    
    def test_explain_complex_query(self):
        """Test EXPLAIN avec requête complexe."""
        sql = """
        EXPLAIN ANALYZE SELECT u.name, COUNT(o.id)
        FROM users u
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.name
        """
        result = parse(sql)
        
        assert isinstance(result.statement, ExplainStatement)
        assert result.statement.analyze is True


# ============== Tests VACUUM statement ==============

class TestVacuumStatement:
    """Tests pour le statement VACUUM."""
    
    def test_simple_vacuum(self):
        """Test VACUUM simple."""
        sql = "VACUUM"
        result = parse(sql)
        
        assert isinstance(result.statement, VacuumStatement)
        assert result.statement.table is None
    
    def test_vacuum_table(self):
        """Test VACUUM sur une table."""
        sql = "VACUUM users"
        result = parse(sql)
        
        assert isinstance(result.statement, VacuumStatement)
        assert result.statement.table.name == "users"
    
    def test_vacuum_full(self):
        """Test VACUUM FULL."""
        sql = "VACUUM FULL users"
        result = parse(sql)
        
        assert result.statement.full is True
    
    def test_vacuum_analyze(self):
        """Test VACUUM ANALYZE."""
        sql = "VACUUM ANALYZE users"
        result = parse(sql)
        
        assert result.statement.analyze is True
    
    def test_vacuum_all_options(self):
        """Test VACUUM avec toutes les options."""
        sql = "VACUUM FULL VERBOSE ANALYZE users"
        result = parse(sql)
        
        assert result.statement.full is True
        assert result.statement.verbose is True
        assert result.statement.analyze is True
    
    def test_vacuum_with_columns(self):
        """Test VACUUM ANALYZE avec colonnes."""
        sql = "VACUUM ANALYZE users (name, email)"
        result = parse(sql)
        
        assert result.statement.analyze is True
        assert "name" in result.statement.columns
        assert "email" in result.statement.columns
    
    def test_vacuum_roundtrip(self):
        """Test round-trip VACUUM."""
        sql = "VACUUM FULL ANALYZE users (id, name)"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "VACUUM" in regenerated
        assert "FULL" in regenerated
        assert "ANALYZE" in regenerated


# ============== Tests GRANT/REVOKE statements ==============

class TestGrantStatement:
    """Tests pour le statement GRANT."""
    
    def test_simple_grant(self):
        """Test GRANT simple."""
        sql = "GRANT SELECT ON users TO admin"
        result = parse(sql)
        
        assert isinstance(result.statement, GrantStatement)
        assert "SELECT" in result.statement.privileges
        assert "admin" in result.statement.grantees
    
    def test_grant_multiple_privileges(self):
        """Test GRANT avec plusieurs privilèges."""
        sql = "GRANT SELECT, INSERT, UPDATE ON orders TO manager"
        result = parse(sql)
        
        assert "SELECT" in result.statement.privileges
        assert "INSERT" in result.statement.privileges
        assert "UPDATE" in result.statement.privileges
    
    def test_grant_all_privileges(self):
        """Test GRANT ALL PRIVILEGES."""
        sql = "GRANT ALL PRIVILEGES ON users TO superuser"
        result = parse(sql)
        
        assert "ALL PRIVILEGES" in result.statement.privileges
    
    def test_grant_with_object_type(self):
        """Test GRANT avec type d'objet."""
        sql = "GRANT SELECT ON TABLE orders TO reader"
        result = parse(sql)
        
        assert result.statement.object_type == "TABLE"
    
    def test_grant_on_schema(self):
        """Test GRANT sur un schéma."""
        sql = "GRANT USAGE ON SCHEMA public TO app_user"
        result = parse(sql)
        
        assert result.statement.object_type == "SCHEMA"
        assert result.statement.object_name == "public"
    
    def test_grant_with_option(self):
        """Test GRANT WITH GRANT OPTION."""
        sql = "GRANT SELECT ON users TO admin WITH GRANT OPTION"
        result = parse(sql)
        
        assert result.statement.with_grant_option is True
    
    def test_grant_multiple_grantees(self):
        """Test GRANT à plusieurs utilisateurs."""
        sql = "GRANT SELECT ON users TO user1, user2, user3"
        result = parse(sql)
        
        assert len(result.statement.grantees) == 3
    
    def test_grant_roundtrip(self):
        """Test round-trip GRANT."""
        sql = "GRANT SELECT, INSERT ON TABLE orders TO manager WITH GRANT OPTION"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "GRANT" in regenerated
        assert "SELECT" in regenerated
        assert "INSERT" in regenerated
        assert "WITH GRANT OPTION" in regenerated


class TestRevokeStatement:
    """Tests pour le statement REVOKE."""
    
    def test_simple_revoke(self):
        """Test REVOKE simple."""
        sql = "REVOKE SELECT ON users FROM guest"
        result = parse(sql)
        
        assert isinstance(result.statement, RevokeStatement)
        assert "SELECT" in result.statement.privileges
        assert "guest" in result.statement.grantees
    
    def test_revoke_all_privileges(self):
        """Test REVOKE ALL PRIVILEGES."""
        sql = "REVOKE ALL PRIVILEGES ON DATABASE mydb FROM old_user"
        result = parse(sql)
        
        assert "ALL PRIVILEGES" in result.statement.privileges
        assert result.statement.object_type == "DATABASE"
    
    def test_revoke_cascade(self):
        """Test REVOKE CASCADE."""
        sql = "REVOKE SELECT ON users FROM guest CASCADE"
        result = parse(sql)
        
        assert result.statement.cascade is True
    
    def test_revoke_roundtrip(self):
        """Test round-trip REVOKE."""
        sql = "REVOKE ALL PRIVILEGES ON TABLE users FROM old_admin CASCADE"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "REVOKE" in regenerated
        assert "ALL PRIVILEGES" in regenerated
        assert "CASCADE" in regenerated


# ============== Tests SQL Formatter ==============

class TestSQLFormatter:
    """Tests pour le formateur SQL."""
    
    def test_format_simple_query(self):
        """Test formatage requête simple."""
        ugly = "SELECT a,b,c FROM t WHERE x>1"
        formatted = format_sql(ugly)
        
        assert "SELECT" in formatted
        assert "\n" in formatted  # Multi-ligne
    
    def test_format_preserves_semantics(self):
        """Test que le formatage préserve la sémantique."""
        original = "SELECT id, name FROM users WHERE active = 1"
        formatted = format_sql(original)
        
        # Re-parser pour vérifier la sémantique
        result1 = parse(original)
        result2 = parse(formatted)
        
        assert len(result1.statement.select_items) == len(result2.statement.select_items)
    
    def test_minify_sql(self):
        """Test minification SQL."""
        multiline = """
        SELECT
            id,
            name
        FROM users
        WHERE active = 1
        """
        minified = minify_sql(multiline)
        
        assert "\n" not in minified
        assert "SELECT" in minified
        assert "FROM" in minified
    
    def test_validate_sql_valid(self):
        """Test validation SQL valide."""
        sql = "SELECT * FROM users"
        result = validate_sql(sql)
        
        assert result['valid'] is True
        assert result['error'] is None
        assert result['info']['statement_type'] == 'SelectStatement'
        assert 'users' in result['info']['tables']
    
    def test_validate_sql_invalid(self):
        """Test validation SQL invalide."""
        sql = "SELCT * FORM users"  # Typos volontaires
        result = validate_sql(sql)
        
        assert result['valid'] is False
        assert result['error'] is not None
    
    def test_format_style_compact(self):
        """Test style compact."""
        sql = "SELECT a, b FROM t"
        formatted = format_sql(sql, style=FormatStyle.COMPACT)
        
        # Le style compact devrait être plus court
        assert formatted is not None
    
    def test_formatter_with_joins(self):
        """Test formatage avec JOINs."""
        ugly = "SELECT u.name,o.total FROM users u INNER JOIN orders o ON u.id=o.user_id WHERE o.total>100"
        formatted = format_sql(ugly)
        
        assert "INNER JOIN" in formatted or "JOIN" in formatted
        assert "ON" in formatted
    
    def test_formatter_with_aggregation(self):
        """Test formatage avec agrégation."""
        ugly = "SELECT department,COUNT(*),AVG(salary) FROM employees GROUP BY department HAVING COUNT(*)>5"
        formatted = format_sql(ugly)
        
        assert "GROUP BY" in formatted
        assert "HAVING" in formatted
    
    def test_formatter_complex_query(self):
        """Test formatage requête complexe."""
        ugly = """SELECT u.id,u.name,COUNT(o.id) as order_count,SUM(o.total) as total_spent FROM users u LEFT JOIN orders o ON u.id=o.user_id WHERE u.status='active' GROUP BY u.id,u.name HAVING COUNT(o.id)>0 ORDER BY total_spent DESC LIMIT 100"""
        formatted = format_sql(ugly)
        
        # Vérifier les clauses principales
        assert "SELECT" in formatted
        assert "FROM" in formatted
        assert "LEFT JOIN" in formatted
        assert "WHERE" in formatted
        assert "GROUP BY" in formatted
        assert "HAVING" in formatted
        assert "ORDER BY" in formatted
        assert "LIMIT" in formatted
    
    def test_validate_returns_metadata(self):
        """Test que validate_sql retourne les métadonnées."""
        sql = """
        SELECT u.name, COUNT(o.id)
        FROM users u
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.name
        """
        result = validate_sql(sql)
        
        assert result['valid'] is True
        assert result['info']['has_aggregation'] is True
        assert result['info']['has_join'] is True
        assert 'users' in result['info']['tables']
        assert 'orders' in result['info']['tables']


# ============== Tests de régression ==============

class TestRegressions:
    """Tests de régression pour s'assurer que les anciennes fonctionnalités marchent."""
    
    def test_basic_select(self):
        """Test SELECT basique toujours fonctionnel."""
        sql = "SELECT id, name FROM users WHERE active = 1"
        result = parse(sql)
        regenerated = generate_sql(result)
        
        assert "SELECT" in regenerated
        assert "FROM users" in regenerated
    
    def test_subquery(self):
        """Test sous-requête toujours fonctionnelle."""
        sql = "SELECT * FROM (SELECT id FROM users) AS sub"
        result = parse(sql)
        
        assert result.has_subquery is True
    
    def test_cte(self):
        """Test CTE toujours fonctionnel."""
        sql = """
        WITH active_users AS (
            SELECT * FROM users WHERE active = 1
        )
        SELECT * FROM active_users
        """
        result = parse(sql)
        
        assert result.statement.ctes is not None
    
    def test_window_functions(self):
        """Test fonctions de fenêtrage toujours fonctionnelles."""
        sql = "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn FROM employees"
        result = parse(sql)
        
        assert result.statement is not None
    
    def test_case_expression(self):
        """Test CASE toujours fonctionnel."""
        sql = "SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'negative' END FROM t"
        result = parse(sql)
        
        assert result.statement is not None
    
    def test_union(self):
        """Test UNION toujours fonctionnel."""
        sql = "SELECT id FROM users UNION SELECT id FROM admins"
        result = parse(sql)
        
        assert result.statement is not None
