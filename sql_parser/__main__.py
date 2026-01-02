#!/usr/bin/env python3
"""
SQL Parser - Script principal.

Permet de parser du SQL depuis la ligne de commande ou un fichier.
Supporte plusieurs dialectes: standard, presto, athena, trino, postgresql, mysql, bigquery, snowflake, spark.

Usage:
    python -m sql_parser "SELECT * FROM users"
    python -m sql_parser -f query.sql
    python -m sql_parser -f query.sql -o result.json
    python -m sql_parser --dialect athena -f dbt_model.sql
"""

import argparse
import sys
import json
from pathlib import Path

from .parser import SQLParser
from .dialects import SQLDialect
from .json_exporter import ASTToJSONExporter


def main():
    parser = argparse.ArgumentParser(
        description="Parse SQL et génère une représentation JSON structurée.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Parser une requête directement
  python -m sql_parser "SELECT id, name FROM users WHERE age > 18"
  
  # Parser depuis un fichier
  python -m sql_parser -f query.sql
  
  # Sauvegarder le résultat dans un fichier
  python -m sql_parser -f query.sql -o result.json
  
  # Mode compact (sans métadonnées)
  python -m sql_parser --compact "SELECT * FROM users"
  
  # Afficher seulement les tokens
  python -m sql_parser --tokens "SELECT * FROM users"
  
  # Spécifier un dialecte (Presto/Athena pour dbt)
  python -m sql_parser --dialect athena -f model.sql
  
  # Parser une requête dbt avec Jinja
  python -m sql_parser --dialect athena "SELECT * FROM {{ ref('users') }}"
  
  # Transpiler de Presto vers PostgreSQL
  python -m sql_parser --dialect presto --transpile postgresql "SELECT IF(x > 0, 'pos', 'neg') FROM t"
  
  # Transpiler de MySQL vers Presto
  python -m sql_parser --dialect mysql --transpile presto "SELECT IFNULL(x, 0), CURDATE() FROM t"

Dialectes supportés: standard, presto, athena, trino, postgresql, mysql, bigquery, snowflake, spark
"""
    )
    
    parser.add_argument(
        "sql",
        nargs="?",
        help="Requête SQL à parser"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Fichier SQL à parser"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Fichier de sortie JSON"
    )
    
    parser.add_argument(
        "-d", "--dialect",
        type=str,
        choices=['standard', 'presto', 'athena', 'trino', 'postgresql', 'mysql', 'bigquery', 'snowflake', 'spark'],
        default=None,
        help="Dialecte SQL à utiliser (auto-détection si non spécifié)"
    )
    
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Mode compact (sans métadonnées ni informations de parsing)"
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclure les métadonnées"
    )
    
    parser.add_argument(
        "--no-parse-info",
        action="store_true",
        help="Exclure les informations de parsing"
    )
    
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation du JSON (défaut: 2)"
    )
    
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Afficher seulement les tokens (analyse lexicale)"
    )
    
    parser.add_argument(
        "--statement-only",
        action="store_true",
        help="Afficher seulement le statement (sans wrapper)"
    )
    
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Générer du SQL à partir de l'AST (réversibilité)"
    )
    
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Générer le SQL sur une seule ligne (avec --generate)"
    )
    
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Mots-clés en minuscules (avec --generate)"
    )
    
    parser.add_argument(
        "--transpile", "-t",
        type=str,
        metavar="TARGET",
        choices=['standard', 'presto', 'athena', 'trino', 'postgresql', 'mysql', 'bigquery', 'snowflake', 'spark'],
        help="Transpiler vers un autre dialecte (ex: --dialect presto --transpile postgresql)"
    )
    
    args = parser.parse_args()
    
    # Récupération du SQL
    sql = None
    
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Erreur: Le fichier '{args.file}' n'existe pas.", file=sys.stderr)
            sys.exit(1)
        sql = filepath.read_text(encoding='utf-8')
    elif args.sql:
        sql = args.sql
    else:
        # Lire depuis stdin si disponible
        if not sys.stdin.isatty():
            sql = sys.stdin.read()
        else:
            print("Erreur: Aucune requête SQL fournie.", file=sys.stderr)
            print("Usage: python -m sql_parser \"SELECT * FROM users\"", file=sys.stderr)
            print("       python -m sql_parser -f query.sql", file=sys.stderr)
            sys.exit(1)
    
    if not sql or not sql.strip():
        print("Erreur: La requête SQL est vide.", file=sys.stderr)
        sys.exit(1)
    
    # Mode tokens uniquement
    if args.tokens:
        from .tokenizer import SQLTokenizer
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        
        result = {
            "tokens": [
                {
                    "type": token.type.name,
                    "value": token.value,
                    "line": token.line,
                    "column": token.column
                }
                for token in tokens
                if token.type.name != "EOF"
            ]
        }
        
        output_json = json.dumps(result, indent=args.indent, ensure_ascii=False)
        
        if args.output:
            Path(args.output).write_text(output_json, encoding='utf-8')
            print(f"Tokens sauvegardés dans '{args.output}'")
        else:
            print(output_json)
        
        return
    
    # Déterminer le dialecte
    dialect = None
    if args.dialect:
        dialect_map = {
            'standard': SQLDialect.STANDARD,
            'presto': SQLDialect.PRESTO,
            'athena': SQLDialect.ATHENA,
            'trino': SQLDialect.TRINO,
            'postgresql': SQLDialect.POSTGRESQL,
            'mysql': SQLDialect.MYSQL,
            'bigquery': SQLDialect.BIGQUERY,
            'snowflake': SQLDialect.SNOWFLAKE,
            'spark': SQLDialect.SPARK,
        }
        dialect = dialect_map.get(args.dialect)
    
    # Parsing
    try:
        sql_parser = SQLParser(dialect=dialect)
        result = sql_parser.parse(sql)
    except Exception as e:
        print(f"Erreur de parsing: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Mode génération SQL (réversibilité)
    if args.generate:
        from .sql_generator import SQLGenerator
        
        generator = SQLGenerator(
            dialect=dialect or SQLDialect.STANDARD,
            indent=args.indent,
            uppercase_keywords=not args.lowercase,
            inline=args.inline
        )
        
        generated_sql = generator.generate(result)
        
        if args.output:
            Path(args.output).write_text(generated_sql, encoding='utf-8')
            print(f"SQL généré sauvegardé dans '{args.output}'")
        else:
            print(generated_sql)
        return
    
    # Mode transpilation entre dialectes
    if args.transpile:
        from .transpiler import transpile as do_transpile
        
        source = dialect or SQLDialect.STANDARD
        target_map = {
            'standard': SQLDialect.STANDARD,
            'presto': SQLDialect.PRESTO,
            'athena': SQLDialect.ATHENA,
            'trino': SQLDialect.TRINO,
            'postgresql': SQLDialect.POSTGRESQL,
            'mysql': SQLDialect.MYSQL,
            'bigquery': SQLDialect.BIGQUERY,
            'snowflake': SQLDialect.SNOWFLAKE,
            'spark': SQLDialect.SPARK,
        }
        target = target_map[args.transpile]
        
        transpile_result = do_transpile(sql, source, target)
        
        if not transpile_result.success:
            print(f"Erreur de transpilation:", file=sys.stderr)
            for warning in transpile_result.warnings:
                print(f"  - {warning}", file=sys.stderr)
            sys.exit(1)
        
        # Afficher les warnings
        if transpile_result.warnings:
            print(f"-- Warnings:", file=sys.stderr)
            for warning in transpile_result.warnings:
                print(f"--   {warning}", file=sys.stderr)
        
        if transpile_result.unsupported_features:
            print(f"-- Unsupported features:", file=sys.stderr)
            for feature in transpile_result.unsupported_features:
                print(f"--   {feature}", file=sys.stderr)
        
        if args.output:
            Path(args.output).write_text(transpile_result.sql, encoding='utf-8')
            print(f"SQL transpilé sauvegardé dans '{args.output}'")
        else:
            print(transpile_result.sql)
        return
    
    # Export JSON
    if args.statement_only:
        output_dict = result.statement.to_dict()
    else:
        exporter = ASTToJSONExporter(
            indent=args.indent,
            include_metadata=not args.no_metadata,
            include_parse_info=not args.no_parse_info,
            compact=args.compact
        )
        output_dict = exporter.export_to_dict(result)
    
    output_json = json.dumps(output_dict, indent=args.indent, ensure_ascii=False)
    
    # Sortie
    if args.output:
        Path(args.output).write_text(output_json, encoding='utf-8')
        print(f"Résultat sauvegardé dans '{args.output}'")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
