"""
SQL Generator - Génère du SQL à partir d'un AST.

Ce module permet de reconstruire une requête SQL à partir de sa
représentation en arbre syntaxique abstrait (AST).
Supporte plusieurs dialectes: Standard, Presto/Athena, etc.
"""

from typing import List, Optional, Union
from .ast_nodes import (
    ASTNode, Expression, Literal, Identifier, ColumnRef, Star, Parameter,
    BinaryOp, UnaryOp, FunctionCall, CaseExpression,
    InExpression, BetweenExpression, LikeExpression, IsNullExpression,
    ExistsExpression, SubqueryExpression, CastExpression,
    SelectItem, TableRef, SubqueryRef, JoinClause, FromClause,
    JoinType, OrderDirection, SetOperationType, OrderByItem,
    CTEDefinition, SelectStatement, ParseInfo, ParseResult,
    # Presto/Athena specific
    ArrayExpression, MapExpression, RowExpression, ArraySubscript,
    LambdaExpression, IntervalExpression, AtTimeZone, TryExpression,
    IfExpression, JinjaExpression, WindowFunction, UnnestRef, TableSample
)
from .dialects import SQLDialect


class SQLGenerator:
    """
    Génère du code SQL à partir d'un AST.
    
    Supporte plusieurs dialectes et options de formatage.
    """
    
    def __init__(self, 
                 dialect: SQLDialect = SQLDialect.STANDARD,
                 indent: int = 2,
                 uppercase_keywords: bool = True,
                 inline: bool = False):
        """
        Initialise le générateur SQL.
        
        Args:
            dialect: Dialecte SQL cible
            indent: Nombre d'espaces pour l'indentation
            uppercase_keywords: Mots-clés en majuscules
            inline: Génère sur une seule ligne
        """
        self.dialect = dialect
        self.indent = indent
        self.uppercase_keywords = uppercase_keywords
        self.inline = inline
        self._indent_level = 0
    
    def generate(self, node: Union[ASTNode, ParseResult]) -> str:
        """
        Génère du SQL à partir d'un nœud AST ou ParseResult.
        
        Args:
            node: Nœud AST ou ParseResult
            
        Returns:
            Code SQL généré
        """
        if isinstance(node, ParseResult):
            return self._generate_from_result(node)
        return self._generate_node(node)
    
    def _generate_from_result(self, result: ParseResult) -> str:
        """Génère SQL depuis un ParseResult complet."""
        parts = []
        
        # Ajouter les configs dbt en premier si présentes
        if hasattr(result, 'statement') and result.statement.metadata:
            metadata = result.statement.metadata
            if 'jinja_config_raw' in metadata:
                for config in metadata['jinja_config_raw']:
                    parts.append(config)
                parts.append('')  # Ligne vide
        
        # Générer le statement
        parts.append(self._generate_node(result.statement))
        
        return '\n'.join(parts)
    
    def _generate_node(self, node: ASTNode) -> str:
        """Dispatch vers la méthode appropriée selon le type de nœud."""
        if node is None:
            return ''
        
        node_type = type(node).__name__
        method_name = f'_gen_{node_type}'
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        
        # Fallback pour les nœuds non implémentés
        return f'/* Unsupported: {node_type} */'
    
    def _kw(self, keyword: str) -> str:
        """Formate un mot-clé selon les options."""
        return keyword.upper() if self.uppercase_keywords else keyword.lower()
    
    def _indent_str(self) -> str:
        """Retourne la chaîne d'indentation courante."""
        if self.inline:
            return ''
        return ' ' * (self.indent * self._indent_level)
    
    def _newline(self) -> str:
        """Retourne un saut de ligne ou espace selon le mode."""
        return ' ' if self.inline else '\n'
    
    # ============== Expressions ==============
    
    def _gen_Literal(self, node: Literal) -> str:
        """Génère un littéral."""
        if node.value is None:
            return self._kw('NULL')
        if node.literal_type == 'boolean':
            return self._kw('TRUE') if node.value else self._kw('FALSE')
        if node.literal_type == 'string':
            # Échapper les guillemets simples
            escaped = str(node.value).replace("'", "''")
            return f"'{escaped}'"
        if node.literal_type in ('date', 'timestamp', 'time'):
            return f"{self._kw(node.literal_type)} '{node.value}'"
        return str(node.value)
    
    def _gen_Identifier(self, node: Identifier) -> str:
        """Génère un identifiant."""
        if node.quoted:
            return f'"{node.name}"'
        return node.name
    
    def _gen_ColumnRef(self, node: ColumnRef) -> str:
        """Génère une référence de colonne."""
        parts = []
        if node.schema:
            parts.append(node.schema)
        if node.table:
            parts.append(node.table)
        
        col = f'"{node.column}"' if node.quoted else node.column
        parts.append(col)
        
        return '.'.join(parts)
    
    def _gen_Star(self, node: Star) -> str:
        """Génère un *."""
        if node.table:
            return f'{node.table}.*'
        return '*'
    
    def _gen_Parameter(self, node: Parameter) -> str:
        """Génère un paramètre."""
        if node.name:
            return f':{node.name}'
        return '?'
    
    def _gen_BinaryOp(self, node: BinaryOp) -> str:
        """Génère une opération binaire."""
        left = self._generate_node(node.left)
        right = self._generate_node(node.right)
        op = self._kw(node.operator) if node.operator in ('AND', 'OR') else node.operator
        return f'({left} {op} {right})'
    
    def _gen_UnaryOp(self, node: UnaryOp) -> str:
        """Génère une opération unaire."""
        operand = self._generate_node(node.operand)
        op = self._kw(node.operator) if node.operator == 'NOT' else node.operator
        return f'{op} {operand}'
    
    def _gen_FunctionCall(self, node: FunctionCall) -> str:
        """Génère un appel de fonction."""
        name = node.name.upper() if self.uppercase_keywords else node.name.lower()
        
        args_parts = []
        if node.distinct:
            args_parts.append(self._kw('DISTINCT') + ' ')
        
        args = ', '.join(self._generate_node(arg) for arg in node.args)
        args_parts.append(args)
        
        result = f"{name}({''.join(args_parts)})"
        
        # Window function
        if hasattr(node, 'over') and node.over:
            result += ' ' + self._generate_node(node.over)
        
        return result
    
    def _gen_CastExpression(self, node: CastExpression) -> str:
        """Génère un CAST."""
        expr = self._generate_node(node.expression)
        return f"{self._kw('CAST')}({expr} {self._kw('AS')} {node.target_type})"
    
    def _gen_CaseExpression(self, node: CaseExpression) -> str:
        """Génère une expression CASE."""
        parts = [self._kw('CASE')]
        
        if node.operand:
            parts.append(' ' + self._generate_node(node.operand))
        
        for when_cond, then_expr in node.when_clauses:
            parts.append(f' {self._kw("WHEN")} {self._generate_node(when_cond)}')
            parts.append(f' {self._kw("THEN")} {self._generate_node(then_expr)}')
        
        if node.else_clause:
            parts.append(f' {self._kw("ELSE")} {self._generate_node(node.else_clause)}')
        
        parts.append(f' {self._kw("END")}')
        
        return ''.join(parts)
    
    def _gen_InExpression(self, node: InExpression) -> str:
        """Génère une expression IN."""
        expr = self._generate_node(node.expression)
        not_str = f'{self._kw("NOT")} ' if node.negated else ''
        
        if node.subquery:
            inner = self._generate_node(node.subquery)
            return f'{expr} {not_str}{self._kw("IN")} ({inner})'
        
        values = ', '.join(self._generate_node(v) for v in node.values)
        return f'{expr} {not_str}{self._kw("IN")} ({values})'
    
    def _gen_BetweenExpression(self, node: BetweenExpression) -> str:
        """Génère une expression BETWEEN."""
        expr = self._generate_node(node.expression)
        low = self._generate_node(node.low)
        high = self._generate_node(node.high)
        not_str = f'{self._kw("NOT")} ' if node.negated else ''
        return f'{expr} {not_str}{self._kw("BETWEEN")} {low} {self._kw("AND")} {high}'
    
    def _gen_LikeExpression(self, node: LikeExpression) -> str:
        """Génère une expression LIKE."""
        expr = self._generate_node(node.expression)
        pattern = self._generate_node(node.pattern)
        not_str = f'{self._kw("NOT")} ' if node.negated else ''
        return f'{expr} {not_str}{self._kw("LIKE")} {pattern}'
    
    def _gen_IsNullExpression(self, node: IsNullExpression) -> str:
        """Génère une expression IS NULL."""
        expr = self._generate_node(node.expression)
        not_str = f'{self._kw("NOT")} ' if node.negated else ''
        return f'{expr} {self._kw("IS")} {not_str}{self._kw("NULL")}'
    
    def _gen_ExistsExpression(self, node: ExistsExpression) -> str:
        """Génère une expression EXISTS."""
        subquery = self._generate_node(node.subquery)
        return f'{self._kw("EXISTS")} ({subquery})'
    
    def _gen_SubqueryExpression(self, node: SubqueryExpression) -> str:
        """Génère une sous-requête dans une expression."""
        return f'({self._generate_node(node.query)})'
    
    # ============== Presto/Athena Expressions ==============
    
    def _gen_ArrayExpression(self, node: ArrayExpression) -> str:
        """Génère un constructeur ARRAY."""
        elements = ', '.join(self._generate_node(e) for e in node.elements)
        return f'{self._kw("ARRAY")}[{elements}]'
    
    def _gen_MapExpression(self, node: MapExpression) -> str:
        """Génère un constructeur MAP."""
        pairs = ', '.join(
            f'{self._generate_node(k)}, {self._generate_node(v)}'
            for k, v in node.pairs
        )
        return f'{self._kw("MAP")}({pairs})'
    
    def _gen_RowExpression(self, node: RowExpression) -> str:
        """Génère un constructeur ROW."""
        values = ', '.join(self._generate_node(v) for v in node.values)
        return f'{self._kw("ROW")}({values})'
    
    def _gen_ArraySubscript(self, node: ArraySubscript) -> str:
        """Génère un accès à un élément d'array."""
        array = self._generate_node(node.array)
        index = self._generate_node(node.index)
        return f'{array}[{index}]'
    
    def _gen_LambdaExpression(self, node: LambdaExpression) -> str:
        """Génère une expression lambda."""
        params = ', '.join(node.parameters)
        if len(node.parameters) > 1:
            params = f'({params})'
        body = self._generate_node(node.body)
        return f'{params} -> {body}'
    
    def _gen_IntervalExpression(self, node: IntervalExpression) -> str:
        """Génère une expression INTERVAL."""
        return f"{self._kw('INTERVAL')} '{node.value}' {self._kw(node.unit)}"
    
    def _gen_AtTimeZone(self, node: AtTimeZone) -> str:
        """Génère une expression AT TIME ZONE."""
        expr = self._generate_node(node.expression)
        tz = self._generate_node(node.timezone)
        return f'{expr} {self._kw("AT TIME ZONE")} {tz}'
    
    def _gen_TryExpression(self, node: TryExpression) -> str:
        """Génère une expression TRY."""
        expr = self._generate_node(node.expression)
        return f'{self._kw("TRY")}({expr})'
    
    def _gen_IfExpression(self, node: IfExpression) -> str:
        """Génère une expression IF."""
        cond = self._generate_node(node.condition)
        then = self._generate_node(node.then_expr)
        els = self._generate_node(node.else_expr) if node.else_expr else self._kw('NULL')
        return f'{self._kw("IF")}({cond}, {then}, {els})'
    
    def _gen_JinjaExpression(self, node: JinjaExpression) -> str:
        """Génère une expression Jinja (passthrough)."""
        return node.content
    
    # ============== Clauses et statements ==============
    
    def _gen_SelectItem(self, node: SelectItem) -> str:
        """Génère un élément de SELECT."""
        expr = self._generate_node(node.expression)
        if node.alias:
            return f'{expr} {self._kw("AS")} {node.alias}'
        return expr
    
    def _gen_TableRef(self, node: TableRef) -> str:
        """Génère une référence de table."""
        # Si c'est un template Jinja, le retourner tel quel
        if hasattr(node, 'is_jinja') and node.is_jinja:
            result = node.name
        else:
            parts = []
            if node.schema:
                parts.append(node.schema)
            parts.append(node.name)
            result = '.'.join(parts)
        
        if node.alias:
            result += f' {node.alias}'
        
        return result
    
    def _gen_SubqueryRef(self, node: SubqueryRef) -> str:
        """Génère une sous-requête dans FROM."""
        self._indent_level += 1
        subquery = self._generate_node(node.query)
        self._indent_level -= 1
        
        result = f'({self._newline()}{subquery}{self._newline()})'
        if node.alias:
            result += f' {self._kw("AS")} {node.alias}'
        return result
    
    def _gen_UnnestRef(self, node: UnnestRef) -> str:
        """Génère un UNNEST dans FROM."""
        expr = self._generate_node(node.expression)
        result = f'{self._kw("UNNEST")}({expr})'
        
        if node.with_ordinality:
            result += f' {self._kw("WITH ORDINALITY")}'
        
        if node.alias:
            result += f' {self._kw("AS")} {node.alias}'
            # L'alias de ordinality est géré séparément si présent
            if node.with_ordinality and node.ordinality_alias:
                result += f'({node.alias}, {node.ordinality_alias})'
        
        return result
    
    def _gen_JoinClause(self, node: JoinClause) -> str:
        """Génère une clause JOIN."""
        join_type = node.join_type.value if node.join_type else 'INNER'
        table = self._generate_node(node.table)
        
        result = f'{self._kw(join_type)} {self._kw("JOIN")} {table}'
        
        if node.condition:
            on_expr = self._generate_node(node.condition)
            result += f' {self._kw("ON")} {on_expr}'
        elif node.using_columns:
            cols = ', '.join(node.using_columns)
            result += f' {self._kw("USING")} ({cols})'
        
        return result
    
    def _gen_FromClause(self, node: FromClause) -> str:
        """Génère une clause FROM."""
        parts = []
        
        for i, table in enumerate(node.tables):
            if i == 0:
                parts.append(self._generate_node(table))
            elif isinstance(table, JoinClause):
                parts.append(self._generate_node(table))
            else:
                parts.append(', ' + self._generate_node(table))
        
        return self._newline().join(parts) if not self.inline else ' '.join(parts)
    
    def _gen_OrderByItem(self, node: OrderByItem) -> str:
        """Génère un élément ORDER BY."""
        expr = self._generate_node(node.expression)
        direction = self._kw(node.direction.value) if node.direction else ''
        
        result = expr
        if direction:
            result += f' {direction}'
        
        if node.nulls_first is not None:
            if node.nulls_first:
                result += f' {self._kw("NULLS FIRST")}'
            else:
                result += f' {self._kw("NULLS LAST")}'
        
        return result
    
    def _gen_CTEDefinition(self, node: CTEDefinition) -> str:
        """Génère une définition de CTE."""
        result = node.name
        
        if node.columns:
            cols = ', '.join(node.columns)
            result += f'({cols})'
        
        result += f' {self._kw("AS")} ('
        
        self._indent_level += 1
        query = self._generate_node(node.query)
        self._indent_level -= 1
        
        result += self._newline() + self._indent_str() + query + self._newline()
        result += self._indent_str() + ')'
        
        return result
    
    def _gen_SelectStatement(self, node: SelectStatement) -> str:
        """Génère un statement SELECT complet."""
        parts = []
        indent = self._indent_str()
        
        # WITH clause (CTEs)
        if node.ctes:
            cte_parts = [self._kw('WITH')]
            if hasattr(node, 'recursive') and node.recursive:
                cte_parts.append(' ' + self._kw('RECURSIVE'))
            
            cte_defs = []
            for cte in node.ctes:
                cte_defs.append(self._generate_node(cte))
            
            cte_parts.append(self._newline())
            cte_parts.append((',' + self._newline()).join(cte_defs))
            parts.append(''.join(cte_parts))
            parts.append('')  # Ligne vide
        
        # SELECT
        select_parts = [self._kw('SELECT')]
        if node.distinct:
            select_parts.append(' ' + self._kw('DISTINCT'))
        
        items = [self._generate_node(item) for item in node.select_items]
        if self.inline:
            select_parts.append(' ' + ', '.join(items))
        else:
            select_parts.append(self._newline())
            select_parts.append((',' + self._newline()).join(
                self._indent_str() + '    ' + item for item in items
            ))
        
        parts.append(''.join(select_parts))
        
        # FROM
        if node.from_clause:
            from_sql = self._generate_node(node.from_clause)
            parts.append(f'{self._kw("FROM")} {from_sql}')
        
        # WHERE
        if node.where_clause:
            where_expr = self._generate_node(node.where_clause)
            parts.append(f'{self._kw("WHERE")} {where_expr}')
        
        # GROUP BY
        if node.group_by:
            exprs = ', '.join(self._generate_node(e) for e in node.group_by)
            parts.append(f'{self._kw("GROUP BY")} {exprs}')
        
        # HAVING
        if node.having_clause:
            having_expr = self._generate_node(node.having_clause)
            parts.append(f'{self._kw("HAVING")} {having_expr}')
        
        # ORDER BY
        if node.order_by:
            items = ', '.join(self._generate_node(item) for item in node.order_by)
            parts.append(f'{self._kw("ORDER BY")} {items}')
        
        # LIMIT
        if node.limit:
            limit_expr = self._generate_node(node.limit)
            parts.append(f'{self._kw("LIMIT")} {limit_expr}')
        
        # OFFSET
        if node.offset:
            offset_expr = self._generate_node(node.offset)
            parts.append(f'{self._kw("OFFSET")} {offset_expr}')
        
        # Set operations (UNION, INTERSECT, EXCEPT)
        if node.set_operation and node.set_query:
            parts.append(self._kw(node.set_operation.value))
            parts.append(self._generate_node(node.set_query))
        
        return self._newline().join(parts)


def generate_sql(node: Union[ASTNode, ParseResult], 
                 dialect: SQLDialect = SQLDialect.STANDARD,
                 **kwargs) -> str:
    """
    Fonction utilitaire pour générer du SQL.
    
    Args:
        node: Nœud AST ou ParseResult
        dialect: Dialecte SQL cible
        **kwargs: Options supplémentaires pour SQLGenerator
        
    Returns:
        Code SQL généré
    """
    generator = SQLGenerator(dialect=dialect, **kwargs)
    return generator.generate(node)
