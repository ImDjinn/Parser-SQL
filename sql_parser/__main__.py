#!/usr/bin/env python3
"""
SQL Parser - Script principal.

Permet de parser du SQL depuis la ligne de commande ou un fichier.

Usage:
    python -m sql_parser "SELECT * FROM users"
    python -m sql_parser -f query.sql
    python -m sql_parser -f query.sql -o result.json
"""

import argparse
import sys
import json
from pathlib import Path

from .parser import SQLParser
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
    
    # Parsing
    try:
        sql_parser = SQLParser()
        result = sql_parser.parse(sql)
    except Exception as e:
        print(f"Erreur de parsing: {e}", file=sys.stderr)
        sys.exit(1)
    
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
