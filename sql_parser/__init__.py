"""
SQL Parser - Un parser SQL performant qui convertit le code SQL en JSON structuré.

Ce module fournit:
- Tokenizer: Analyse lexicale du SQL
- AST Nodes: Représentation structurée des éléments SQL
- Parser: Analyse syntaxique et construction de l'AST
- Export JSON: Conversion de l'AST en JSON
- SQL Generator: Reconstruction du SQL à partir de l'AST (réversibilité)
- SQL Formatter: Formatage et indentation du code SQL
- Transpiler: Traduction de SQL entre dialectes (Presto ↔ PostgreSQL ↔ MySQL ↔ BigQuery)
- DBT Converter: Conversion de DML/DDL en modèles dbt (T-SQL → dbt Athena)

Usage:
    from sql_parser import SQLParser, SQLGenerator, transpile, convert_to_dbt
    
    # Parser SQL -> AST
    parser = SQLParser()
    result = parser.parse("SELECT * FROM users WHERE age > 18")
    json_output = result.to_json()
    
    # Générer SQL <- AST (réversible)
    generator = SQLGenerator()
    sql = generator.generate(result)
    
    # Formater du SQL
    from sql_parser import format_sql, minify_sql
    formatted = format_sql("SELECT a,b,c FROM t WHERE x>1")
    minified = minify_sql(formatted)
    
    # Transpiler entre dialectes
    result = transpile(
        "SELECT IF(x > 0, 'positive', 'negative') FROM t",
        source_dialect="presto",
        target_dialect="postgresql"
    )
    print(result.sql)  # CASE WHEN x > 0 THEN 'positive' ELSE 'negative' END
    
    # Convertir T-SQL vers dbt Athena
    result = convert_to_dbt(
        "INSERT INTO target SELECT * FROM source WHERE date > '2024-01-01'",
        source_dialect="tsql",
        target_dialect="athena"
    )
    print(result.models[0].to_file_content())
"""

from .tokenizer import SQLTokenizer, Token, TokenType
from .ast_nodes import *
from .parser import SQLParser
from .json_exporter import ASTToJSONExporter
from .sql_generator import SQLGenerator, generate_sql
from .dialects import SQLDialect
from .transpiler import SQLTranspiler, transpile, TranspilationResult
from .dbt_converter import DbtConverter, DbtModel, DbtConfig, convert_to_dbt, ConversionResult
from .formatter import SQLFormatter, format_sql, minify_sql, validate_sql, FormatStyle, FormatOptions

__version__ = "1.3.0"
__all__ = [
    "SQLParser",
    "SQLTokenizer", 
    "Token",
    "TokenType",
    "ASTToJSONExporter",
    "SQLGenerator",
    "generate_sql",
    "SQLDialect",
    "SQLTranspiler",
    "transpile",
    "TranspilationResult",
    "DbtConverter",
    "DbtModel",
    "DbtConfig",
    "convert_to_dbt",
    "ConversionResult",
]
