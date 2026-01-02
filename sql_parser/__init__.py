"""
SQL Parser - Un parser SQL performant qui convertit le code SQL en JSON structuré.

Ce module fournit:
- Tokenizer: Analyse lexicale du SQL
- AST Nodes: Représentation structurée des éléments SQL
- Parser: Analyse syntaxique et construction de l'AST
- Export JSON: Conversion de l'AST en JSON

Usage:
    from sql_parser import SQLParser
    
    parser = SQLParser()
    result = parser.parse("SELECT * FROM users WHERE age > 18")
    json_output = result.to_json()
"""

from .tokenizer import SQLTokenizer, Token, TokenType
from .ast_nodes import *
from .parser import SQLParser
from .json_exporter import ASTToJSONExporter

__version__ = "1.0.0"
__all__ = [
    "SQLParser",
    "SQLTokenizer", 
    "Token",
    "TokenType",
    "ASTToJSONExporter",
]
