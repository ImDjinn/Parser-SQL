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
    LATERAL = "LATERAL"  # Presto/Athena


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
    """Représente * ou table.* dans SELECT, avec support EXCEPT/REPLACE (BigQuery)."""
    table: Optional[str] = None
    except_columns: Optional[List[str]] = None  # EXCEPT(col1, col2)
    replace_columns: Optional[List[tuple]] = None  # REPLACE(expr AS col)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"node_type": "Star"}
        if self.table:
            result["table"] = self.table
        if self.except_columns:
            result["except"] = self.except_columns
        if self.replace_columns:
            result["replace"] = [{"expression": e.to_dict(), "alias": a} for e, a in self.replace_columns]
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
class QuantifiedComparison(Expression):
    """Comparaison quantifiée: x > ALL(subquery), x = ANY(subquery), x = SOME(subquery)."""
    expression: Expression
    operator: str  # =, <>, <, >, <=, >=
    quantifier: str  # ALL, ANY, SOME
    subquery: 'SelectStatement'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "QuantifiedComparison",
            "expression": self.expression.to_dict(),
            "operator": self.operator,
            "quantifier": self.quantifier,
            "subquery": self.subquery.to_dict()
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
class NamedArgument(Expression):
    """Argument nommé pour STRUCT/ROW (ex: 1 AS field_name)."""
    expression: Expression
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "NamedArgument",
            "expression": self.expression.to_dict(),
            "name": self.name
        }


@dataclass
class KeyValuePair(Expression):
    """Paire clé:valeur pour JSON_OBJECT, etc."""
    key: Expression
    value: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "KeyValuePair",
            "key": self.key.to_dict(),
            "value": self.value.to_dict()
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
    is_try_cast: bool = False  # TRY_CAST pour Presto/Athena
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "TryCastExpression" if self.is_try_cast else "CastExpression",
            "expression": self.expression.to_dict(),
            "target_type": self.target_type
        }
        return result


@dataclass
class ExtractExpression(Expression):
    """Expression EXTRACT (ex: EXTRACT(YEAR FROM date_col))."""
    field: str  # YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, etc.
    expression: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "ExtractExpression",
            "field": self.field,
            "expression": self.expression.to_dict()
        }


# ============== Expressions Presto/Athena spécifiques ==============

@dataclass
class ArrayExpression(Expression):
    """Expression ARRAY[...] (Presto/Athena)."""
    elements: List[Expression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "ArrayExpression",
            "elements": [e.to_dict() for e in self.elements]
        }


@dataclass
class MapExpression(Expression):
    """Expression MAP(ARRAY[], ARRAY[]) ou MAP(k, v, ...) (Presto/Athena)."""
    keys: List[Expression] = field(default_factory=list)
    values: List[Expression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "MapExpression",
            "keys": [k.to_dict() for k in self.keys],
            "values": [v.to_dict() for v in self.values]
        }


@dataclass
class RowExpression(Expression):
    """Expression ROW(...) (Presto/Athena)."""
    fields: List[Expression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "RowExpression",
            "fields": [f.to_dict() for f in self.fields]
        }


@dataclass
class ArraySubscript(Expression):
    """Accès à un élément de tableau: array[index] (Presto/Athena)."""
    array: Expression
    index: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "ArraySubscript",
            "array": self.array.to_dict(),
            "index": self.index.to_dict()
        }


@dataclass
class LambdaExpression(Expression):
    """Expression lambda: x -> x + 1 (Presto/Athena)."""
    parameters: List[str]
    body: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "LambdaExpression",
            "parameters": self.parameters,
            "body": self.body.to_dict()
        }


@dataclass
class IntervalExpression(Expression):
    """Expression INTERVAL '1' DAY (Presto/Athena)."""
    value: Expression
    unit: str  # DAY, HOUR, MINUTE, SECOND, MONTH, YEAR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "IntervalExpression",
            "value": self.value.to_dict(),
            "unit": self.unit
        }


@dataclass
class AtTimeZone(Expression):
    """Expression AT TIME ZONE (Presto/Athena)."""
    expression: Expression
    timezone: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "AtTimeZone",
            "expression": self.expression.to_dict(),
            "timezone": self.timezone.to_dict()
        }


@dataclass
class TryExpression(Expression):
    """Expression TRY(...) (Presto/Athena)."""
    expression: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "TryExpression",
            "expression": self.expression.to_dict()
        }


@dataclass
class IfExpression(Expression):
    """Expression IF(cond, then, else) (Presto/Athena)."""
    condition: Expression
    then_expr: Expression
    else_expr: Optional[Expression] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "IfExpression",
            "condition": self.condition.to_dict(),
            "then": self.then_expr.to_dict()
        }
        if self.else_expr:
            result["else"] = self.else_expr.to_dict()
        return result


@dataclass
class JinjaExpression(Expression):
    """Expression Jinja (dbt): {{ ref('table') }}, {{ var('x') }}."""
    content: str
    jinja_type: str  # 'expression', 'statement', 'comment'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "JinjaExpression",
            "content": self.content,
            "jinja_type": self.jinja_type
        }


@dataclass
class WindowFunction(Expression):
    """Fonction de fenêtre avec OVER (Presto/Athena)."""
    function: FunctionCall
    partition_by: Optional[List[Expression]] = None
    order_by: Optional[List['OrderByItem']] = None
    frame_type: Optional[str] = None  # ROWS, RANGE, GROUPS
    frame_start: Optional[str] = None  # UNBOUNDED PRECEDING, CURRENT ROW, etc.
    frame_end: Optional[str] = None
    window_name: Optional[str] = None  # Reference to named window (OVER w)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "WindowFunction",
            "function": self.function.to_dict()
        }
        if self.partition_by:
            result["partition_by"] = [p.to_dict() for p in self.partition_by]
        if self.order_by:
            result["order_by"] = [o.to_dict() for o in self.order_by]
        if self.frame_type:
            result["frame"] = {
                "type": self.frame_type,
                "start": self.frame_start,
                "end": self.frame_end
            }
        return result


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
    is_jinja: bool = False  # True si c'est un template Jinja (dbt: {{ ref('...') }})
    quoted: bool = False    # True si le nom est entre guillemets
    schema_quoted: bool = False  # True si le schema est entre guillemets
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "TableRef",
            "name": self.name
        }
        if self.alias:
            result["alias"] = self.alias
        if self.schema:
            result["schema"] = self.schema
        if self.is_jinja:
            result["is_jinja"] = True
        if self.quoted:
            result["quoted"] = True
        if self.schema_quoted:
            result["schema_quoted"] = True
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
class UnnestRef(ASTNode):
    """UNNEST dans FROM (Presto/Athena)."""
    expression: Expression
    alias: Optional[str] = None
    with_ordinality: bool = False
    ordinality_alias: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "UnnestRef",
            "expression": self.expression.to_dict()
        }
        if self.alias:
            result["alias"] = self.alias
        if self.with_ordinality:
            result["with_ordinality"] = True
            if self.ordinality_alias:
                result["ordinality_alias"] = self.ordinality_alias
        return result


@dataclass
class TableSample(ASTNode):
    """TABLESAMPLE (Presto/Athena)."""
    table: Union['TableRef', 'SubqueryRef']
    sample_type: str  # BERNOULLI ou SYSTEM
    percentage: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "TableSample",
            "table": self.table.to_dict(),
            "sample_type": self.sample_type,
            "percentage": self.percentage.to_dict()
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
    
    @property
    def joins(self) -> List[JoinClause]:
        """Retourne la liste de tous les JOINs dans la clause FROM."""
        return [t for t in self.tables if isinstance(t, JoinClause)]
    
    @property
    def primary_table(self) -> Optional[Union[TableRef, SubqueryRef]]:
        """Retourne la table principale (première table non-JOIN)."""
        for t in self.tables:
            if isinstance(t, (TableRef, SubqueryRef)):
                return t
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "FromClause",
            "tables": [t.to_dict() for t in self.tables]
        }


# ============== Clause GROUP BY avancée ==============

@dataclass
class GroupingElement(ASTNode):
    """Element de base pour GROUP BY."""
    expressions: List[Expression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "GroupingElement",
            "expressions": [e.to_dict() for e in self.expressions]
        }


@dataclass
class GroupingSets(ASTNode):
    """GROUPING SETS ((a), (b), (a, b))."""
    sets: List[List[Expression]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "GroupingSets",
            "sets": [[e.to_dict() for e in s] for s in self.sets]
        }


@dataclass
class Cube(ASTNode):
    """CUBE(a, b) - toutes les combinaisons possibles."""
    expressions: List[Expression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Cube",
            "expressions": [e.to_dict() for e in self.expressions]
        }


@dataclass
class Rollup(ASTNode):
    """ROLLUP(a, b) - hiérarchie des agrégations."""
    expressions: List[Expression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Rollup",
            "expressions": [e.to_dict() for e in self.expressions]
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
class WindowDefinition(ASTNode):
    """Définition d'une fenêtre nommée (WINDOW w AS (...))."""
    name: str
    partition_by: Optional[List[Expression]] = None
    order_by: Optional[List['OrderByItem']] = None
    frame_type: Optional[str] = None  # ROWS, RANGE, GROUPS
    frame_start: Optional[str] = None
    frame_end: Optional[str] = None
    frame_exclusion: Optional[str] = None  # EXCLUDE CURRENT ROW, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"node_type": "WindowDefinition", "name": self.name}
        if self.partition_by:
            result["partition_by"] = [e.to_dict() for e in self.partition_by]
        if self.order_by:
            result["order_by"] = [o.to_dict() for o in self.order_by]
        if self.frame_type:
            result["frame_type"] = self.frame_type
        if self.frame_start:
            result["frame_start"] = self.frame_start
        if self.frame_end:
            result["frame_end"] = self.frame_end
        if self.frame_exclusion:
            result["frame_exclusion"] = self.frame_exclusion
        return result


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
    distinct_on: Optional[List[Expression]] = None  # PostgreSQL DISTINCT ON (col1, col2)
    ctes: Optional[List[CTEDefinition]] = None
    window_clause: Optional[List[WindowDefinition]] = None  # WINDOW w AS (...)
    metadata: Optional[Dict[str, Any]] = None  # Jinja config, etc.
    
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
        
        if self.distinct_on:
            result["distinct_on"] = [expr.to_dict() for expr in self.distinct_on]
        
        if self.metadata:
            result["metadata"] = self.metadata
            
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


# ============== INSERT Statement ==============

@dataclass
class InsertStatement(ASTNode):
    """Statement INSERT INTO."""
    table: TableRef
    columns: Optional[List[str]] = None
    values: Optional[List[List[Expression]]] = None  # Pour VALUES (...)
    query: Optional['SelectStatement'] = None  # Pour INSERT ... SELECT
    on_conflict: Optional['OnConflictClause'] = None  # PostgreSQL UPSERT
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "InsertStatement",
            "table": self.table.to_dict()
        }
        if self.columns:
            result["columns"] = self.columns
        if self.values:
            result["values"] = [[v.to_dict() for v in row] for row in self.values]
        if self.query:
            result["query"] = self.query.to_dict()
        if self.on_conflict:
            result["on_conflict"] = self.on_conflict.to_dict()
        return result


@dataclass
class OnConflictClause(ASTNode):
    """Clause ON CONFLICT pour PostgreSQL UPSERT."""
    conflict_target: Optional[List[str]] = None  # Colonnes en conflit
    action: str = "NOTHING"  # NOTHING ou UPDATE
    update_assignments: Optional[List['Assignment']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "OnConflictClause",
            "action": self.action
        }
        if self.conflict_target:
            result["conflict_target"] = self.conflict_target
        if self.update_assignments:
            result["update"] = [a.to_dict() for a in self.update_assignments]
        return result


# ============== UPDATE Statement ==============

@dataclass
class Assignment(ASTNode):
    """Assignation col = valeur pour UPDATE/MERGE."""
    column: str
    value: Expression
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Assignment",
            "column": self.column,
            "value": self.value.to_dict()
        }


@dataclass
class UpdateStatement(ASTNode):
    """Statement UPDATE."""
    table: TableRef
    assignments: List[Assignment]
    from_clause: Optional[FromClause] = None  # Pour UPDATE ... FROM (PostgreSQL)
    where_clause: Optional[Expression] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "UpdateStatement",
            "table": self.table.to_dict(),
            "set": [a.to_dict() for a in self.assignments]
        }
        if self.from_clause:
            result["from"] = self.from_clause.to_dict()
        if self.where_clause:
            result["where"] = self.where_clause.to_dict()
        return result


# ============== DELETE Statement ==============

@dataclass
class DeleteStatement(ASTNode):
    """Statement DELETE."""
    table: TableRef
    using: Optional[List[TableRef]] = None  # DELETE ... USING (PostgreSQL)
    where_clause: Optional[Expression] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "DeleteStatement",
            "table": self.table.to_dict()
        }
        if self.using:
            result["using"] = [t.to_dict() for t in self.using]
        if self.where_clause:
            result["where"] = self.where_clause.to_dict()
        return result


# ============== MERGE Statement ==============

@dataclass
class MergeWhenClause(ASTNode):
    """Clause WHEN pour MERGE."""
    matched: bool  # True = WHEN MATCHED, False = WHEN NOT MATCHED
    condition: Optional[Expression] = None  # AND condition optionnelle
    action: str = "UPDATE"  # UPDATE, INSERT, DELETE
    assignments: Optional[List[Assignment]] = None  # Pour UPDATE
    insert_columns: Optional[List[str]] = None  # Pour INSERT
    insert_values: Optional[List[Expression]] = None  # Pour INSERT
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "MergeWhenClause",
            "matched": self.matched,
            "action": self.action
        }
        if self.condition:
            result["condition"] = self.condition.to_dict()
        if self.assignments:
            result["set"] = [a.to_dict() for a in self.assignments]
        if self.insert_columns:
            result["columns"] = self.insert_columns
        if self.insert_values:
            result["values"] = [v.to_dict() for v in self.insert_values]
        return result


@dataclass
class MergeStatement(ASTNode):
    """Statement MERGE INTO."""
    target: TableRef
    source: Union[TableRef, 'SubqueryRef']
    on_condition: Expression
    when_clauses: List[MergeWhenClause]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "MergeStatement",
            "target": self.target.to_dict(),
            "source": self.source.to_dict(),
            "on": self.on_condition.to_dict(),
            "when_clauses": [w.to_dict() for w in self.when_clauses]
        }


# ============== DDL Statements ==============

@dataclass
class ColumnDefinition(ASTNode):
    """Définition d'une colonne pour CREATE TABLE."""
    name: str
    data_type: str
    nullable: bool = True
    default: Optional[Expression] = None
    primary_key: bool = False
    unique: bool = False
    references: Optional[str] = None  # FOREIGN KEY table(column)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "ColumnDefinition",
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable
        }
        if self.default:
            result["default"] = self.default.to_dict()
        if self.primary_key:
            result["primary_key"] = True
        if self.unique:
            result["unique"] = True
        if self.references:
            result["references"] = self.references
        return result


@dataclass
class TableConstraint(ASTNode):
    """Contrainte de table pour CREATE TABLE."""
    constraint_type: str  # PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK
    name: Optional[str] = None
    columns: Optional[List[str]] = None
    references_table: Optional[str] = None
    references_columns: Optional[List[str]] = None
    check_expression: Optional[Expression] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "TableConstraint",
            "type": self.constraint_type
        }
        if self.name:
            result["name"] = self.name
        if self.columns:
            result["columns"] = self.columns
        if self.references_table:
            result["references"] = {
                "table": self.references_table,
                "columns": self.references_columns
            }
        if self.check_expression:
            result["check"] = self.check_expression.to_dict()
        return result


@dataclass
class CreateTableStatement(ASTNode):
    """Statement CREATE TABLE."""
    table: TableRef
    columns: List[ColumnDefinition]
    constraints: Optional[List[TableConstraint]] = None
    if_not_exists: bool = False
    temporary: bool = False
    as_query: Optional['SelectStatement'] = None  # CREATE TABLE AS SELECT
    external: bool = False  # Athena EXTERNAL TABLE
    location: Optional[str] = None  # Athena LOCATION
    stored_as: Optional[str] = None  # Athena STORED AS
    row_format: Optional[str] = None  # Athena ROW FORMAT
    table_properties: Optional[Dict[str, str]] = None  # Athena TBLPROPERTIES
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "CreateTableStatement",
            "table": self.table.to_dict()
        }
        if self.if_not_exists:
            result["if_not_exists"] = True
        if self.temporary:
            result["temporary"] = True
        if self.external:
            result["external"] = True
        if self.columns:
            result["columns"] = [c.to_dict() for c in self.columns]
        if self.constraints:
            result["constraints"] = [c.to_dict() for c in self.constraints]
        if self.as_query:
            result["as"] = self.as_query.to_dict()
        if self.location:
            result["location"] = self.location
        if self.stored_as:
            result["stored_as"] = self.stored_as
        if self.row_format:
            result["row_format"] = self.row_format
        if self.table_properties:
            result["table_properties"] = self.table_properties
        return result


@dataclass
class DropStatement(ASTNode):
    """Statement DROP TABLE/VIEW/INDEX."""
    object_type: str  # TABLE, VIEW, INDEX, SCHEMA, DATABASE
    name: TableRef
    if_exists: bool = False
    cascade: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "DropStatement",
            "object_type": self.object_type,
            "name": self.name.to_dict()
        }
        if self.if_exists:
            result["if_exists"] = True
        if self.cascade:
            result["cascade"] = True
        return result


@dataclass
class AlterTableAction(ASTNode):
    """Action pour ALTER TABLE."""
    action_type: str  # ADD COLUMN, DROP COLUMN, RENAME COLUMN, MODIFY COLUMN, ADD CONSTRAINT, etc.
    column: Optional[ColumnDefinition] = None
    old_name: Optional[str] = None  # Pour RENAME
    new_name: Optional[str] = None  # Pour RENAME
    constraint: Optional[TableConstraint] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "AlterTableAction",
            "action_type": self.action_type
        }
        if self.column:
            result["column"] = self.column.to_dict()
        if self.old_name:
            result["old_name"] = self.old_name
        if self.new_name:
            result["new_name"] = self.new_name
        if self.constraint:
            result["constraint"] = self.constraint.to_dict()
        return result


@dataclass
class AlterTableStatement(ASTNode):
    """Statement ALTER TABLE."""
    table: TableRef
    actions: List[AlterTableAction]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "AlterTableStatement",
            "table": self.table.to_dict(),
            "actions": [a.to_dict() for a in self.actions]
        }


# ============== CREATE VIEW ==============

@dataclass
class CreateViewStatement(ASTNode):
    """Statement CREATE VIEW."""
    name: TableRef
    query: SelectStatement
    columns: Optional[List[str]] = None
    or_replace: bool = False
    if_not_exists: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_type": "CreateViewStatement",
            "name": self.name.to_dict(),
            "query": self.query.to_dict()
        }
        if self.columns:
            result["columns"] = self.columns
        if self.or_replace:
            result["or_replace"] = True
        if self.if_not_exists:
            result["if_not_exists"] = True
        return result


# ============== TRUNCATE ==============

@dataclass
class TruncateStatement(ASTNode):
    """Statement TRUNCATE TABLE."""
    table: TableRef
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "TruncateStatement",
            "table": self.table.to_dict()
        }


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
