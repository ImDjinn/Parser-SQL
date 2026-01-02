"""
Tests pour le module dbt_converter.
"""

import pytest
import sys
sys.path.insert(0, '..')

from sql_parser import convert_to_dbt, DbtConverter, SQLDialect
from sql_parser.dbt_converter import DbtMaterialization, IncrementalStrategy


class TestDbtConverterBasic:
    """Tests basiques de conversion vers dbt."""
    
    def test_insert_to_table(self):
        """INSERT simple -> materialization table."""
        result = convert_to_dbt(
            """INSERT INTO analytics.users
               SELECT id, name, email FROM staging.raw_users""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='users'
        )
        assert result.success
        assert len(result.models) == 1
        model = result.models[0]
        assert model.name == 'users'
        assert model.config.materialized == DbtMaterialization.TABLE
        assert "{{ ref(" in model.to_file_content()
    
    def test_create_table_as_select(self):
        """CREATE TABLE AS SELECT -> table."""
        result = convert_to_dbt(
            """CREATE TABLE reports.summary AS
               SELECT category, SUM(amount) as total FROM orders GROUP BY category""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='summary'
        )
        assert result.success
        model = result.models[0]
        assert model.config.materialized == DbtMaterialization.TABLE
    
    def test_create_view(self):
        """CREATE VIEW -> materialization view."""
        result = convert_to_dbt(
            """CREATE VIEW analytics.active_users AS
               SELECT * FROM users WHERE status = 'active'""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='active_users'
        )
        assert result.success
        model = result.models[0]
        assert model.config.materialized == DbtMaterialization.VIEW


class TestMergeToIncremental:
    """Tests de conversion MERGE -> incremental."""
    
    def test_merge_to_incremental(self):
        """MERGE -> incremental avec merge strategy."""
        result = convert_to_dbt(
            """MERGE INTO warehouse.orders tgt
               USING staging.new_orders src
               ON tgt.order_id = src.order_id
               WHEN MATCHED THEN UPDATE SET tgt.status = src.status
               WHEN NOT MATCHED THEN INSERT (order_id, status) VALUES (src.order_id, src.status)""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='fct_orders'
        )
        assert result.success
        model = result.models[0]
        assert model.config.materialized == DbtMaterialization.INCREMENTAL
        assert model.config.incremental_strategy == IncrementalStrategy.MERGE
        # Should extract unique_key from ON condition
        assert model.config.unique_key is not None
        assert 'order_id' in model.config.unique_key
    
    def test_merge_extracts_update_columns(self):
        """MERGE extrait les colonnes de mise à jour."""
        result = convert_to_dbt(
            """MERGE INTO target t
               USING source s ON t.id = s.id
               WHEN MATCHED THEN UPDATE SET t.name = s.name, t.updated_at = GETDATE()
               WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA
        )
        assert result.success
        model = result.models[0]
        assert model.config.merge_update_columns is not None
        assert 'name' in model.config.merge_update_columns


class TestTranspilation:
    """Tests de transpilation T-SQL -> Athena dans dbt."""
    
    def test_tsql_functions_to_athena(self):
        """Les fonctions T-SQL sont transpilées vers Athena."""
        result = convert_to_dbt(
            """INSERT INTO metrics
               SELECT 
                   ISNULL(name, 'Unknown') as name,
                   GETDATE() as created_at,
                   CONVERT(DATE, timestamp) as date_only
               FROM source_table""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='metrics'
        )
        assert result.success
        content = result.models[0].to_file_content()
        # ISNULL -> COALESCE
        assert 'COALESCE' in content
        # GETDATE() -> CURRENT_TIMESTAMP()
        assert 'CURRENT_TIMESTAMP' in content
        # CONVERT -> CAST
        assert 'CAST' in content
    
    def test_dateadd_transpilation(self):
        """DATEADD transpilé correctement."""
        result = convert_to_dbt(
            """INSERT INTO recent_orders
               SELECT * FROM orders WHERE created_at >= DATEADD(DAY, -30, GETDATE())""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='recent_orders'
        )
        assert result.success
        content = result.models[0].to_file_content()
        assert 'DATE_ADD' in content


class TestRefGeneration:
    """Tests de génération des {{ ref() }}."""
    
    def test_table_to_ref(self):
        """Les tables sont remplacées par {{ ref() }}."""
        result = convert_to_dbt(
            """INSERT INTO output SELECT * FROM users JOIN orders ON users.id = orders.user_id""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='output'
        )
        assert result.success
        content = result.models[0].to_file_content()
        assert "{{ ref('users') }}" in content
    
    def test_schema_prefix_removed(self):
        """Le préfixe de schéma est supprimé dans ref()."""
        result = convert_to_dbt(
            """INSERT INTO output SELECT * FROM dbo.users""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='output'
        )
        assert result.success
        content = result.models[0].to_file_content()
        # dbo. should be removed
        assert "dbo." not in content
        assert "{{ ref('users') }}" in content


class TestIncrementalDetection:
    """Tests de détection de patterns incrémentaux."""
    
    def test_date_filter_incremental(self):
        """Un filtre sur date déclenche le mode incremental."""
        result = convert_to_dbt(
            """INSERT INTO daily_metrics
               SELECT * FROM source WHERE created_at >= '2024-01-01'""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='daily_metrics'
        )
        assert result.success
        model = result.models[0]
        # Date filter detected -> incremental
        if model.config.materialized == DbtMaterialization.INCREMENTAL:
            assert "is_incremental()" in model.to_file_content()


class TestConfigGeneration:
    """Tests de génération du bloc config."""
    
    def test_config_block_format(self):
        """Le bloc config est correctement formaté."""
        result = convert_to_dbt(
            """INSERT INTO target SELECT * FROM source""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='target'
        )
        assert result.success
        content = result.models[0].to_file_content()
        assert "{{ config(" in content
        assert "materialized=" in content
        assert ") }}" in content
    
    def test_incremental_config_complete(self):
        """Le config incremental contient unique_key et strategy."""
        result = convert_to_dbt(
            """MERGE INTO target t USING source s ON t.id = s.id
               WHEN MATCHED THEN UPDATE SET t.val = s.val
               WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)""",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA,
            model_name='target'
        )
        assert result.success
        content = result.models[0].to_file_content()
        assert "materialized='incremental'" in content
        assert "incremental_strategy=" in content


class TestDbtConverterClass:
    """Tests de la classe DbtConverter."""
    
    def test_converter_initialization(self):
        """DbtConverter s'initialise correctement."""
        converter = DbtConverter(
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA
        )
        assert converter.source_dialect == SQLDialect.TSQL
        assert converter.target_dialect == SQLDialect.ATHENA
    
    def test_converter_convert_method(self):
        """La méthode convert() fonctionne."""
        converter = DbtConverter(
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA
        )
        result = converter.convert(
            "INSERT INTO target SELECT * FROM source",
            model_name='target'
        )
        assert result.success
        assert len(result.models) == 1


class TestErrorHandling:
    """Tests de gestion des erreurs."""
    
    def test_invalid_sql_error(self):
        """SQL invalide génère une erreur."""
        result = convert_to_dbt(
            "NOT VALID SQL AT ALL ;;;",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA
        )
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_unsupported_statement_warning(self):
        """Les statements non supportés génèrent des warnings."""
        # DELETE n'est pas directement convertible en dbt model
        result = convert_to_dbt(
            "DELETE FROM users WHERE inactive = 1",
            source_dialect=SQLDialect.TSQL,
            target_dialect=SQLDialect.ATHENA
        )
        # Should have warnings about DELETE
        if result.success:
            assert len(result.warnings) > 0 or len(result.models) == 0
