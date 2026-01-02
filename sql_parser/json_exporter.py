"""
Exporteur JSON pour l'AST SQL.

Convertit l'AST en format JSON avec différentes options de formatage.
"""

import json
from typing import Any, Dict, Optional, Union
from .ast_nodes import ASTNode, ParseResult


class ASTToJSONExporter:
    """Exporte un AST SQL vers JSON."""
    
    def __init__(
        self,
        indent: int = 2,
        include_metadata: bool = True,
        include_parse_info: bool = True,
        compact: bool = False
    ):
        """
        Initialise l'exporteur.
        
        Args:
            indent: Indentation pour le JSON (None pour compact)
            include_metadata: Inclure les métadonnées (tables, colonnes référencées, etc.)
            include_parse_info: Inclure les informations de parsing (SQL original, warnings)
            compact: Mode compact (sans métadonnées ni info de parsing)
        """
        self.indent = None if compact else indent
        self.include_metadata = include_metadata and not compact
        self.include_parse_info = include_parse_info and not compact
    
    def export(self, parse_result: ParseResult) -> str:
        """
        Exporte un ParseResult en JSON.
        
        Args:
            parse_result: Résultat du parsing
            
        Returns:
            Chaîne JSON
        """
        data = self._build_output(parse_result)
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def export_to_dict(self, parse_result: ParseResult) -> Dict[str, Any]:
        """
        Exporte un ParseResult en dictionnaire Python.
        
        Args:
            parse_result: Résultat du parsing
            
        Returns:
            Dictionnaire
        """
        return self._build_output(parse_result)
    
    def export_to_file(self, parse_result: ParseResult, filepath: str) -> None:
        """
        Exporte un ParseResult vers un fichier JSON.
        
        Args:
            parse_result: Résultat du parsing
            filepath: Chemin du fichier de sortie
        """
        data = self._build_output(parse_result)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False)
    
    def _build_output(self, parse_result: ParseResult) -> Dict[str, Any]:
        """Construit le dictionnaire de sortie."""
        output = {
            "statement": parse_result.statement.to_dict()
        }
        
        if self.include_metadata:
            output["metadata"] = {
                "tables_referenced": parse_result.tables_referenced,
                "columns_referenced": parse_result.columns_referenced,
                "functions_used": parse_result.functions_used,
                "has_aggregation": parse_result.has_aggregation,
                "has_subquery": parse_result.has_subquery,
                "has_join": parse_result.has_join
            }
        
        if self.include_parse_info:
            output["parse_info"] = parse_result.parse_info.to_dict()
        
        return output


def to_json(
    parse_result: ParseResult,
    indent: int = 2,
    include_metadata: bool = True,
    include_parse_info: bool = True,
    compact: bool = False
) -> str:
    """
    Fonction utilitaire pour convertir un ParseResult en JSON.
    
    Args:
        parse_result: Résultat du parsing
        indent: Indentation pour le JSON
        include_metadata: Inclure les métadonnées
        include_parse_info: Inclure les informations de parsing
        compact: Mode compact
        
    Returns:
        Chaîne JSON
    """
    exporter = ASTToJSONExporter(
        indent=indent,
        include_metadata=include_metadata,
        include_parse_info=include_parse_info,
        compact=compact
    )
    return exporter.export(parse_result)


def to_dict(
    parse_result: ParseResult,
    include_metadata: bool = True,
    include_parse_info: bool = True,
    compact: bool = False
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour convertir un ParseResult en dictionnaire.
    
    Args:
        parse_result: Résultat du parsing
        include_metadata: Inclure les métadonnées
        include_parse_info: Inclure les informations de parsing
        compact: Mode compact
        
    Returns:
        Dictionnaire
    """
    exporter = ASTToJSONExporter(
        include_metadata=include_metadata,
        include_parse_info=include_parse_info,
        compact=compact
    )
    return exporter.export_to_dict(parse_result)
