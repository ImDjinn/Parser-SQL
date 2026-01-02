"""
Nœuds de l'Arbre Syntaxique Abstrait (AST) pour SQL.

Ce module définit toutes les classes représentant les différents
éléments d'une requête SQL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from enum import Enum, auto


class ASTNode(ABC):
    """Classe de base pour tous les nœuds de l'AST."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le nœud en dictionnaire."""
        pass
    
    def get_type(self) -> str:
        """Retourne le type du nœud."""
        return self.__class__.__name__


# ============== Types d'expressions ==============

class ExpressionType(Enum):
    """Types d'expressions SQL."""
    LITERAL = auto()
    IDENTIFIER = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    FUNCTION_CALL = auto()
    CASE = auto()
    SUBQUERY = auto()
    IN_LIST = auto()
    BETWEEN = auto()
    LIKE = auto()
    IS_NULL = auto()
    COLUMN_REF = auto()
    STAR = auto()
    PARAMETER = auto()


class JoinType(Enum):
    """Types de jointures SQL."""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"
    NATURAL = "NATURAL"


class OrderDirection(Enum):
    """Direction de tri."""
    ASC = "ASC"
    DESC = "DESC"


class SetOperationType(Enum):
    """Types d'opérations ensemblistes."""
    UNION = "UNION"
    UNION_ALL = "UNION ALL"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"


class ComparisonOperator(Enum):
    """Opérateurs de comparaison."""
    EQ = "="
    NE = "<>"
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    LIKE = "LIKE"
    IN = "IN"
    BETWEEN = "BETWEEN"
    IS = "IS"
    IS_NOT = "IS NOT"


class LogicalOperator(Enum):
    """Opérateurs logiques."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class ArithmeticOperator(Enum):
    """Opérateurs arithmétiques."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    CONCAT = "||"


# ============== Expressions ==============

@dataclass
class Expression(ASTNode):
    """Classe de base pour les expressions."""
    pass


@dataclass
class Literal(Expression):
    """Valeur littérale (nombre, chaîne, booléen, NULL)."""
    value: Any
    literal_type: str  # 'integer', 'float', 'string', 'boolean', 'null'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Literal",
            "value": self.value,
            "literal_type": self.literal_type
        }


@dataclass
class Identifier(Expression):
    """Identifiant simple (nom de colonne, table, etc.)."""
    name: str
    quoted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Identifier",
            "name": self.name,
            "quoted": self.quoted
        }


@dataclass 
class ColumnRef(Expression):
    """Référence à une colonne avec potentiellement table/schéma."""
    column: str
    table: Optional[str] = None
    schema: Optional[str] = None
    quoted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "ColumnRef",
            "column": self.column,
        }
        if self.table:
            result["table"] = self.table
        if self.schema:
            result["schema"] = self.schema
        if self.quoted:
            result["quoted"] = True
        return result
    
    def full_name(self) -> str:
        """Retourne le nom complet de la colonne."""
        parts = []
        if self.schema:
            parts.append(self.schema)
        if self.table:
            parts.append(self.table)
        parts.append(self.column)
        return ".".join(parts)


@dataclass
class Star(Expression):
    """Représente * ou table.* dans SELECT."""
    table: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"node_type": "Star"}
        if self.table:
            result["table"] = self.table
        return result


@dataclass
class Parameter(Expression):
    """Paramètre de requête (? ou :name)."""
    name: Optional[str] = None  # None pour ?, nom pour :param
    position: Optional[int] = None  # Position pour ?
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"node_type": "Parameter"}
        if self.name:
            result["name"] = self.name
        if self.position is not None:
            result["position"] = self.position
        return result


@dataclass
class BinaryOp(Expression):
    """Opération binaire (ex: a + b, x AND y, col = 5)."""
    left: Expression
    operator: str
    right: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "BinaryOp",
            "operator": self.operator,
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        }


@dataclass
class UnaryOp(Expression):
    """Opération unaire (ex: NOT x, -5)."""
    operator: str
    operand: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "UnaryOp",
            "operator": self.operator,
            "operand": self.operand.to_dict()
        }


@dataclass
class FunctionCall(Expression):
    """Appel de fonction (ex: COUNT(*), UPPER(name))."""
    name: str
    args: List[Expression] = field(default_factory=list)
    distinct: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "FunctionCall",
            "name": self.name.upper(),
            "arguments": [arg.to_dict() for arg in self.args]
        }
        if self.distinct:
            result["distinct"] = True
        return result


@dataclass
class CaseExpression(Expression):
    """Expression CASE WHEN ... THEN ... ELSE ... END."""
    operand: Optional[Expression] = None  # Pour CASE expr WHEN ...
    when_clauses: List[tuple] = field(default_factory=list)  # [(condition, result), ...]
    else_clause: Optional[Expression] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "CaseExpression",
            "when_clauses": [
                {"when": w.to_dict(), "then": t.to_dict()} 
                for w, t in self.when_clauses
            ]
        }
        if self.operand:
            result["operand"] = self.operand.to_dict()
        if self.else_clause:
            result["else"] = self.else_clause.to_dict()
        return result


@dataclass
class InExpression(Expression):
    """Expression IN (ex: col IN (1, 2, 3) ou col IN (SELECT ...))."""
    expression: Expression
    values: List[Expression] = field(default_factory=list)  # Pour liste de valeurs
    subquery: Optional['SelectStatement'] = None  # Pour sous-requête
    negated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "InExpression",
            "expression": self.expression.to_dict(),
            "negated": self.negated
        }
        if self.subquery:
            result["subquery"] = self.subquery.to_dict()
        else:
            result["values"] = [v.to_dict() for v in self.values]
        return result


@dataclass
class BetweenExpression(Expression):
    """Expression BETWEEN (ex: col BETWEEN 1 AND 10)."""
    expression: Expression
    low: Expression
    high: Expression
    negated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "BetweenExpression",
            "expression": self.expression.to_dict(),
            "low": self.low.to_dict(),
            "high": self.high.to_dict(),
            "negated": self.negated
        }


@dataclass
class LikeExpression(Expression):
    """Expression LIKE (ex: col LIKE '%test%')."""
    expression: Expression
    pattern: Expression
    escape: Optional[Expression] = None
    negated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "LikeExpression",
            "expression": self.expression.to_dict(),
            "pattern": self.pattern.to_dict(),
            "negated": self.negated
        }
        if self.escape:
            result["escape"] = self.escape.to_dict()
        return result


@dataclass
class IsNullExpression(Expression):
    """Expression IS NULL ou IS NOT NULL."""
    expression: Expression
    negated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "IsNullExpression",
            "expression": self.expression.to_dict(),
            "negated": self.negated
        }


@dataclass
class ExistsExpression(Expression):
    """Expression EXISTS (SELECT ...)."""
    subquery: 'SelectStatement'
    negated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "ExistsExpression",
            "subquery": self.subquery.to_dict(),
            "negated": self.negated
        }


@dataclass
class SubqueryExpression(Expression):
    """Sous-requête utilisée comme expression."""
    query: 'SelectStatement'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "SubqueryExpression",
            "query": self.query.to_dict()
        }


@dataclass
class CastExpression(Expression):
    """Expression CAST (ex: CAST(col AS INTEGER))."""
    expression: Expression
    target_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "CastExpression",
            "expression": self.expression.to_dict(),
            "target_type": self.target_type
        }


# ============== Éléments de SELECT ==============

@dataclass
class SelectItem(ASTNode):
    """Un élément dans la clause SELECT."""
    expression: Expression
    alias: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "SelectItem",
            "expression": self.expression.to_dict()
        }
        if self.alias:
            result["alias"] = self.alias
        return result


# ============== Éléments de FROM ==============

@dataclass
class TableRef(ASTNode):
    """Référence à une table."""
    name: str
    alias: Optional[str] = None
    schema: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "TableRef",
            "name": self.name
        }
        if self.alias:
            result["alias"] = self.alias
        if self.schema:
            result["schema"] = self.schema
        return result


@dataclass
class SubqueryRef(ASTNode):
    """Sous-requête dans FROM."""
    query: 'SelectStatement'
    alias: str  # Obligatoire pour les sous-requêtes dans FROM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "SubqueryRef",
            "query": self.query.to_dict(),
            "alias": self.alias
        }


@dataclass
class JoinClause(ASTNode):
    """Clause JOIN."""
    join_type: JoinType
    table: Union[TableRef, SubqueryRef, 'JoinClause']
    condition: Optional[Expression] = None  # ON condition
    using_columns: Optional[List[str]] = None  # USING (col1, col2)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "JoinClause",
            "join_type": self.join_type.value,
            "table": self.table.to_dict()
        }
        if self.condition:
            result["on"] = self.condition.to_dict()
        if self.using_columns:
            result["using"] = self.using_columns
        return result


@dataclass
class FromClause(ASTNode):
    """Clause FROM complète."""
    tables: List[Union[TableRef, SubqueryRef, JoinClause]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "FromClause",
            "tables": [t.to_dict() for t in self.tables]
        }


# ============== Clause ORDER BY ==============

@dataclass
class OrderByItem(ASTNode):
    """Un élément dans la clause ORDER BY."""
    expression: Expression
    direction: OrderDirection = OrderDirection.ASC
    nulls_first: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "OrderByItem",
            "expression": self.expression.to_dict(),
            "direction": self.direction.value
        }
        if self.nulls_first is not None:
            result["nulls_first"] = self.nulls_first
        return result


# ============== CTE (Common Table Expressions) ==============

@dataclass
class CTEDefinition(ASTNode):
    """Définition d'une CTE (WITH clause)."""
    name: str
    query: 'SelectStatement'
    columns: Optional[List[str]] = None
    recursive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "CTEDefinition",
            "name": self.name,
            "query": self.query.to_dict()
        }
        if self.columns:
            result["columns"] = self.columns
        if self.recursive:
            result["recursive"] = True
        return result


# ============== Statement principal: SELECT ==============

@dataclass
class SelectStatement(ASTNode):
    """Statement SELECT complet."""
    select_items: List[SelectItem] = field(default_factory=list)
    from_clause: Optional[FromClause] = None
    where_clause: Optional[Expression] = None
    group_by: Optional[List[Expression]] = None
    having_clause: Optional[Expression] = None
    order_by: Optional[List[OrderByItem]] = None
    limit: Optional[Expression] = None
    offset: Optional[Expression] = None
    distinct: bool = False
    ctes: Optional[List[CTEDefinition]] = None
    
    # Pour UNION, INTERSECT, EXCEPT
    set_operation: Optional[SetOperationType] = None
    set_query: Optional['SelectStatement'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "SelectStatement",
            "select": [item.to_dict() for item in self.select_items]
        }
        
        if self.distinct:
            result["distinct"] = True
            
        if self.ctes:
            result["with"] = [cte.to_dict() for cte in self.ctes]
        
        if self.from_clause:
            result["from"] = self.from_clause.to_dict()
            
        if self.where_clause:
            result["where"] = self.where_clause.to_dict()
            
        if self.group_by:
            result["group_by"] = [expr.to_dict() for expr in self.group_by]
            
        if self.having_clause:
            result["having"] = self.having_clause.to_dict()
            
        if self.order_by:
            result["order_by"] = [item.to_dict() for item in self.order_by]
            
        if self.limit:
            result["limit"] = self.limit.to_dict()
            
        if self.offset:
            result["offset"] = self.offset.to_dict()
            
        if self.set_operation and self.set_query:
            result["set_operation"] = {
                "type": self.set_operation.value,
                "query": self.set_query.to_dict()
            }
        
        return result


# ============== Informations de parsing ==============

@dataclass
class ParseInfo(ASTNode):
    """Informations sur le parsing."""
    original_sql: str
    warnings: List[str] = field(default_factory=list)
    implicit_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "ParseInfo",
            "original_sql": self.original_sql
        }
        if self.warnings:
            result["warnings"] = self.warnings
        if self.implicit_info:
            result["implicit_info"] = self.implicit_info
        return result


@dataclass
class ParseResult(ASTNode):
    """Résultat complet du parsing."""
    statement: SelectStatement
    parse_info: ParseInfo
    
    # Métadonnées extraites
    tables_referenced: List[str] = field(default_factory=list)
    columns_referenced: List[str] = field(default_factory=list)
    functions_used: List[str] = field(default_factory=list)
    has_aggregation: bool = False
    has_subquery: bool = False
    has_join: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "ParseResult",
            "statement": self.statement.to_dict(),
            "metadata": {
                "tables_referenced": self.tables_referenced,
                "columns_referenced": self.columns_referenced,
                "functions_used": self.functions_used,
                "has_aggregation": self.has_aggregation,
                "has_subquery": self.has_subquery,
                "has_join": self.has_join
            },
            "parse_info": self.parse_info.to_dict()
        }
