"""
SQL Formatter - Formateur et indenteur SQL.

Ce module fournit des fonctionnalités de formatage SQL avec des options personnalisables.

Usage:
    from sql_parser.formatter import format_sql, SQLFormatter, FormatStyle
    
    # Formatage simple
    formatted = format_sql("SELECT a,b,c FROM t WHERE x>1")
    
    # Avec style personnalisé
    formatted = format_sql(sql, style=FormatStyle.COMPACT)
    
    # Configuration avancée
    formatter = SQLFormatter(
        indent_size=4,
        uppercase_keywords=True,
        max_line_length=120,
        align_columns=True
    )
    formatted = formatter.format(sql)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Union
import re

from .parser import SQLParser, ParseResult
from .sql_generator import SQLGenerator
from .dialects import SQLDialect


class FormatStyle(Enum):
    """Styles de formatage prédéfinis."""
    STANDARD = "standard"      # Style par défaut avec indentation
    COMPACT = "compact"        # Minimal, une seule ligne quand possible
    EXPANDED = "expanded"      # Très étendu, chaque clause sur sa ligne
    ALIGNED = "aligned"        # Colonnes alignées


@dataclass
class FormatOptions:
    """Options de formatage SQL."""
    
    # Indentation
    indent_size: int = 4
    indent_char: str = " "
    
    # Casse des mots-clés
    uppercase_keywords: bool = True
    
    # Colonnes
    columns_per_line: int = 1         # Nombre de colonnes par ligne (0 = toutes)
    align_columns: bool = False        # Aligner les colonnes verticalement
    
    # Opérateurs
    spaces_around_operators: bool = True
    
    # Parenthèses
    space_after_open_paren: bool = False
    space_before_close_paren: bool = False
    
    # Virgules
    comma_at_end: bool = True          # True: "col," | False: ", col" (leading comma)
    space_after_comma: bool = True
    
    # Lignes
    max_line_length: int = 120
    newline_before_and_or: bool = True  # AND/OR sur nouvelle ligne
    
    # Clauses
    newline_after_select: bool = True
    newline_before_from: bool = True
    newline_before_where: bool = True
    newline_before_join: bool = True
    newline_before_group_by: bool = True
    newline_before_having: bool = True
    newline_before_order_by: bool = True
    
    # Style compact
    inline_simple_queries: bool = False  # Requêtes simples sur une ligne


class SQLFormatter:
    """
    Formateur SQL avec options personnalisables.
    
    Utilise le parser pour construire l'AST puis régénère le SQL
    avec le formatage souhaité.
    """
    
    def __init__(self, 
                 options: FormatOptions = None,
                 style: FormatStyle = FormatStyle.STANDARD,
                 dialect: SQLDialect = SQLDialect.STANDARD):
        """
        Initialise le formateur.
        
        Args:
            options: Options de formatage personnalisées
            style: Style prédéfini (ignoré si options fourni)
            dialect: Dialecte SQL
        """
        self.dialect = dialect
        
        if options:
            self.options = options
        else:
            self.options = self._get_style_options(style)
        
        self.parser = SQLParser(dialect=dialect)
    
    def _get_style_options(self, style: FormatStyle) -> FormatOptions:
        """Retourne les options pour un style prédéfini."""
        if style == FormatStyle.COMPACT:
            return FormatOptions(
                indent_size=2,
                columns_per_line=0,  # Toutes sur une ligne
                newline_after_select=False,
                newline_before_from=False,
                inline_simple_queries=True
            )
        elif style == FormatStyle.EXPANDED:
            return FormatOptions(
                indent_size=4,
                columns_per_line=1,
                newline_before_and_or=True,
                align_columns=True
            )
        elif style == FormatStyle.ALIGNED:
            return FormatOptions(
                indent_size=4,
                columns_per_line=1,
                align_columns=True,
                comma_at_end=False  # Leading comma
            )
        else:  # STANDARD
            return FormatOptions()
    
    def format(self, sql: str) -> str:
        """
        Formate une requête SQL.
        
        Args:
            sql: Code SQL à formater
            
        Returns:
            Code SQL formaté
        """
        # Parser le SQL
        result = self.parser.parse(sql)
        
        # Créer un générateur avec les options appropriées
        generator = SQLGenerator(
            dialect=self.dialect,
            uppercase_keywords=self.options.uppercase_keywords,
            indent=self.options.indent_size,
            inline=self.options.inline_simple_queries
        )
        
        # Générer le SQL formaté
        formatted = generator.generate(result)
        
        # Appliquer des post-traitements si nécessaire
        formatted = self._post_process(formatted)
        
        return formatted
    
    def _post_process(self, sql: str) -> str:
        """Applique des post-traitements au SQL formaté."""
        
        # Leading comma si demandé
        if not self.options.comma_at_end:
            sql = self._convert_to_leading_comma(sql)
        
        # Compacter si style compact et requête simple
        if self.options.inline_simple_queries:
            if sql.count('\n') <= 5 and len(sql) < self.options.max_line_length:
                sql = self._compact_simple_query(sql)
        
        return sql
    
    def _convert_to_leading_comma(self, sql: str) -> str:
        """Convertit les virgules de fin en virgules de début."""
        lines = sql.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if stripped.endswith(','):
                # Retirer la virgule de fin
                result.append(stripped[:-1])
            elif i > 0 and result and not result[-1].rstrip().endswith(','):
                # Ajouter la virgule au début si la ligne précédente n'en a pas
                indent = len(line) - len(line.lstrip())
                if line.strip() and not any(line.strip().upper().startswith(kw) for kw in 
                    ['FROM', 'WHERE', 'AND', 'OR', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
                     'GROUP', 'HAVING', 'ORDER', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT']):
                    result.append(' ' * indent + ', ' + line.strip())
                else:
                    result.append(line)
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _compact_simple_query(self, sql: str) -> str:
        """Compacte une requête simple sur une ligne."""
        # Remplacer les retours à la ligne par des espaces
        compacted = ' '.join(line.strip() for line in sql.split('\n') if line.strip())
        # Normaliser les espaces multiples
        compacted = re.sub(r'\s+', ' ', compacted)
        return compacted
    
    def format_file(self, filepath: str, output_path: str = None) -> str:
        """
        Formate un fichier SQL.
        
        Args:
            filepath: Chemin du fichier à formater
            output_path: Chemin de sortie (si None, retourne le contenu)
            
        Returns:
            Contenu formaté
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            sql = f.read()
        
        formatted = self.format(sql)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
        
        return formatted


def format_sql(sql: str, 
               style: FormatStyle = FormatStyle.STANDARD,
               dialect: SQLDialect = SQLDialect.STANDARD,
               **kwargs) -> str:
    """
    Fonction utilitaire pour formater du SQL.
    
    Args:
        sql: Code SQL à formater
        style: Style de formatage prédéfini
        dialect: Dialecte SQL
        **kwargs: Options supplémentaires (passées à FormatOptions)
        
    Returns:
        Code SQL formaté
        
    Examples:
        >>> format_sql("SELECT a,b FROM t WHERE x>1")
        'SELECT\\n    a,\\n    b\\nFROM t\\nWHERE (x > 1)'
        
        >>> format_sql("SELECT a FROM t", style=FormatStyle.COMPACT)
        'SELECT a FROM t'
    """
    if kwargs:
        options = FormatOptions(**kwargs)
        formatter = SQLFormatter(options=options, dialect=dialect)
    else:
        formatter = SQLFormatter(style=style, dialect=dialect)
    
    return formatter.format(sql)


def minify_sql(sql: str, dialect: SQLDialect = SQLDialect.STANDARD) -> str:
    """
    Minifie une requête SQL (supprime espaces et retours inutiles).
    
    Args:
        sql: Code SQL à minifier
        dialect: Dialecte SQL
        
    Returns:
        Code SQL minifié
        
    Examples:
        >>> minify_sql("SELECT\\n    a,\\n    b\\nFROM t")
        'SELECT a, b FROM t'
    """
    formatter = SQLFormatter(style=FormatStyle.COMPACT, dialect=dialect)
    formatted = formatter.format(sql)
    
    # Compacter sur une ligne
    minified = ' '.join(line.strip() for line in formatted.split('\n') if line.strip())
    minified = re.sub(r'\s+', ' ', minified)
    
    return minified


def validate_sql(sql: str, dialect: SQLDialect = SQLDialect.STANDARD) -> dict:
    """
    Valide la syntaxe SQL et retourne les informations.
    
    Args:
        sql: Code SQL à valider
        dialect: Dialecte SQL
        
    Returns:
        Dict avec 'valid', 'error', 'formatted', 'info'
    """
    parser = SQLParser(dialect=dialect)
    
    try:
        result = parser.parse(sql)
        formatter = SQLFormatter(dialect=dialect)
        formatted = formatter.format(sql)
        
        return {
            'valid': True,
            'error': None,
            'formatted': formatted,
            'info': {
                'statement_type': type(result.statement).__name__,
                'tables': result.tables_referenced,
                'columns': result.columns_referenced,
                'has_subquery': result.has_subquery,
                'has_aggregation': result.has_aggregation,
                'has_join': result.has_join
            }
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'formatted': None,
            'info': None
        }
