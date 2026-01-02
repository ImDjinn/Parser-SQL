"""
SQL Generator - Génère du SQL à partir d'un AST.

Ce module permet de reconstruire une requête SQL à partir de sa
représentation en arbre syntaxique abstrait (AST).
Supporte plusieurs dialectes: Standard, Presto/Athena, etc.
"""

from typing import List, Optional, Union
from .ast_nodes import (
    ASTNode, Expression, Literal, Identifier, ColumnRef, Star, Parameter,
    BinaryOp, UnaryOp, FunctionCall, CaseExpression, NamedArgument,
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
        
        # Ajouter les configs dbt en premier si présentes (seulement pour SelectStatement)
        if hasattr(result, 'statement') and hasattr(result.statement, 'metadata') and result.statement.metadata:
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
    
    def _gen_NamedArgument(self, node: NamedArgument) -> str:
        """Génère un argument nommé (expr AS name)."""
        expr = self._generate_node(node.expression)
        return f'{expr} {self._kw("AS")} {node.name}'
    
    def _gen_FunctionCall(self, node: FunctionCall) -> str:
        """Génère un appel de fonction."""
        # Handle special transpiler markers
        if node.name == '__TSQL_VALUES__':
            # Convert to T-SQL VALUES format: (SELECT value FROM (VALUES (1),(2),(3)) AS t(value))
            values = ', '.join(f'({self._generate_node(arg)})' for arg in node.args)
            return f"(SELECT value FROM (VALUES {values}) AS t(value))"
        
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
    
    def _gen_WindowFunction(self, node) -> str:
        """Génère une fonction de fenêtre avec OVER."""
        # Générer l'appel de fonction de base
        func_str = self._gen_FunctionCall(node.function)
        
        # Construire la clause OVER
        over_parts = []
        
        # PARTITION BY
        if node.partition_by:
            partition_exprs = ', '.join(self._generate_node(e) for e in node.partition_by)
            over_parts.append(f"{self._kw('PARTITION BY')} {partition_exprs}")
        
        # ORDER BY
        if node.order_by:
            order_exprs = ', '.join(self._gen_OrderByItem(o) for o in node.order_by)
            over_parts.append(f"{self._kw('ORDER BY')} {order_exprs}")
        
        # Frame specification
        if node.frame_type:
            frame_str = node.frame_type.upper() if self.uppercase_keywords else node.frame_type.lower()
            if node.frame_end:
                frame_str += f" {self._kw('BETWEEN')} {node.frame_start} {self._kw('AND')} {node.frame_end}"
            else:
                frame_str += f" {node.frame_start}"
            over_parts.append(frame_str)
        
        over_clause = ' '.join(over_parts)
        return f"{func_str} {self._kw('OVER')} ({over_clause})"
    
    def _gen_CastExpression(self, node: CastExpression) -> str:
        """Génère un CAST ou TRY_CAST/SAFE_CAST."""
        expr = self._generate_node(node.expression)
        if node.is_try_cast:
            # Use SAFE_CAST for BigQuery, TRY_CAST for Presto/Athena
            if self.dialect == SQLDialect.BIGQUERY:
                cast_kw = self._kw('SAFE_CAST')
            else:
                cast_kw = self._kw('TRY_CAST')
        else:
            cast_kw = self._kw('CAST')
        return f"{cast_kw}({expr} {self._kw('AS')} {node.target_type})"
    
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
        value = self._generate_node(node.value)
        return f"{self._kw('INTERVAL')} {value} {self._kw(node.unit)}"
    
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
                schema = node.schema
                if getattr(node, 'schema_quoted', False):
                    schema = f'"{schema}"'
                parts.append(schema)
            # Gérer les identifiants quotés
            name = node.name
            if getattr(node, 'quoted', False):
                name = f'"{name}"'
            parts.append(name)
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
            # Check if any CTE is recursive
            has_recursive = any(getattr(cte, 'recursive', False) for cte in node.ctes)
            if has_recursive:
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
            # DISTINCT ON (PostgreSQL specific)
            if node.distinct_on:
                on_exprs = ', '.join(self._generate_node(e) for e in node.distinct_on)
                select_parts.append(f' {self._kw("ON")} ({on_exprs})')
        
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

    # ============== INSERT Statement ==============
    
    def _gen_InsertStatement(self, node) -> str:
        """Génère un statement INSERT INTO."""
        parts = [f"{self._kw('INSERT INTO')} {self._generate_node(node.table)}"]
        
        if node.columns:
            cols = ', '.join(node.columns)
            parts[0] += f" ({cols})"
        
        if node.values:
            value_rows = []
            for row in node.values:
                vals = ', '.join(self._generate_node(v) for v in row)
                value_rows.append(f"({vals})")
            parts.append(f"{self._kw('VALUES')} {', '.join(value_rows)}")
        elif node.query:
            parts.append(self._generate_node(node.query))
        
        if node.on_conflict:
            parts.append(self._gen_OnConflictClause(node.on_conflict))
        
        return self._newline().join(parts)
    
    def _gen_OnConflictClause(self, node) -> str:
        """Génère ON CONFLICT clause."""
        parts = [self._kw('ON CONFLICT')]
        if node.conflict_target:
            parts.append(f"({', '.join(node.conflict_target)})")
        parts.append(self._kw('DO'))
        parts.append(self._kw(node.action))
        if node.action == 'UPDATE' and node.update_assignments:
            parts.append(self._kw('SET'))
            assigns = ', '.join(f"{a.column} = {self._generate_node(a.value)}" for a in node.update_assignments)
            parts.append(assigns)
        return ' '.join(parts)

    # ============== UPDATE Statement ==============
    
    def _gen_UpdateStatement(self, node) -> str:
        """Génère un statement UPDATE."""
        parts = [f"{self._kw('UPDATE')} {self._generate_node(node.table)}"]
        
        assigns = ', '.join(f"{a.column} = {self._generate_node(a.value)}" for a in node.assignments)
        parts.append(f"{self._kw('SET')} {assigns}")
        
        if node.from_clause:
            parts.append(self._generate_node(node.from_clause))
        
        if node.where_clause:
            parts.append(f"{self._kw('WHERE')} {self._generate_node(node.where_clause)}")
        
        return self._newline().join(parts)
    
    def _gen_Assignment(self, node) -> str:
        """Génère une assignation."""
        return f"{node.column} = {self._generate_node(node.value)}"

    # ============== DELETE Statement ==============
    
    def _gen_DeleteStatement(self, node) -> str:
        """Génère un statement DELETE."""
        parts = [f"{self._kw('DELETE FROM')} {self._generate_node(node.table)}"]
        
        if node.using:
            using_tables = ', '.join(self._generate_node(t) for t in node.using)
            parts.append(f"{self._kw('USING')} {using_tables}")
        
        if node.where_clause:
            parts.append(f"{self._kw('WHERE')} {self._generate_node(node.where_clause)}")
        
        return self._newline().join(parts)

    # ============== MERGE Statement ==============
    
    def _gen_MergeStatement(self, node) -> str:
        """Génère un statement MERGE INTO."""
        parts = [f"{self._kw('MERGE INTO')} {self._generate_node(node.target)}"]
        parts.append(f"{self._kw('USING')} {self._generate_node(node.source)}")
        parts.append(f"{self._kw('ON')} {self._generate_node(node.on_condition)}")
        
        for when_clause in node.when_clauses:
            parts.append(self._gen_MergeWhenClause(when_clause))
        
        return self._newline().join(parts)
    
    def _gen_MergeWhenClause(self, node) -> str:
        """Génère une clause WHEN pour MERGE."""
        parts = [self._kw('WHEN')]
        if not node.matched:
            parts.append(self._kw('NOT'))
        parts.append(self._kw('MATCHED'))
        
        if node.condition:
            parts.append(f"{self._kw('AND')} {self._generate_node(node.condition)}")
        
        parts.append(self._kw('THEN'))
        
        if node.action == 'UPDATE':
            parts.append(self._kw('UPDATE SET'))
            assigns = ', '.join(f"{a.column} = {self._generate_node(a.value)}" for a in node.assignments)
            parts.append(assigns)
        elif node.action == 'DELETE':
            parts.append(self._kw('DELETE'))
        elif node.action == 'INSERT':
            parts.append(self._kw('INSERT'))
            if node.insert_columns:
                parts.append(f"({', '.join(node.insert_columns)})")
            parts.append(self._kw('VALUES'))
            vals = ', '.join(self._generate_node(v) for v in node.insert_values)
            parts.append(f"({vals})")
        
        return ' '.join(parts)

    # ============== CREATE TABLE Statement ==============
    
    def _gen_CreateTableStatement(self, node) -> str:
        """Génère un statement CREATE TABLE."""
        parts = []
        create_part = [self._kw('CREATE')]
        if node.temporary:
            create_part.append(self._kw('TEMPORARY'))
        if node.external:
            create_part.append(self._kw('EXTERNAL'))
        create_part.append(self._kw('TABLE'))
        if node.if_not_exists:
            create_part.append(self._kw('IF NOT EXISTS'))
        create_part.append(self._generate_node(node.table))
        parts.append(' '.join(create_part))
        
        if node.as_query:
            parts.append(f"{self._kw('AS')}")
            parts.append(self._generate_node(node.as_query))
        elif node.columns:
            col_defs = []
            for col in node.columns:
                col_defs.append(self._gen_ColumnDefinition(col))
            if node.constraints:
                for constraint in node.constraints:
                    col_defs.append(self._gen_TableConstraint(constraint))
            parts.append(f"({', '.join(col_defs)})")
        
        if node.location:
            parts.append(f"{self._kw('LOCATION')} '{node.location}'")
        if node.stored_as:
            parts.append(f"{self._kw('STORED AS')} {node.stored_as}")
        
        return self._newline().join(parts)
    
    def _gen_ColumnDefinition(self, node) -> str:
        """Génère une définition de colonne."""
        parts = [node.name, node.data_type]
        if not node.nullable:
            parts.append(self._kw('NOT NULL'))
        if node.default:
            parts.append(f"{self._kw('DEFAULT')} {self._generate_node(node.default)}")
        if node.primary_key:
            parts.append(self._kw('PRIMARY KEY'))
        if node.unique:
            parts.append(self._kw('UNIQUE'))
        if node.references:
            parts.append(f"{self._kw('REFERENCES')} {node.references}")
        return ' '.join(parts)
    
    def _gen_TableConstraint(self, node) -> str:
        """Génère une contrainte de table."""
        parts = []
        if node.name:
            parts.append(f"{self._kw('CONSTRAINT')} {node.name}")
        parts.append(self._kw(node.constraint_type))
        if node.columns:
            parts.append(f"({', '.join(node.columns)})")
        if node.references_table:
            parts.append(f"{self._kw('REFERENCES')} {node.references_table}")
            if node.references_columns:
                parts.append(f"({', '.join(node.references_columns)})")
        if node.check_expression:
            parts.append(f"({self._generate_node(node.check_expression)})")
        return ' '.join(parts)

    # ============== CREATE VIEW Statement ==============
    
    def _gen_CreateViewStatement(self, node) -> str:
        """Génère un statement CREATE VIEW."""
        parts = []
        create_part = [self._kw('CREATE')]
        if node.or_replace:
            create_part.append(self._kw('OR REPLACE'))
        create_part.append(self._kw('VIEW'))
        if node.if_not_exists:
            create_part.append(self._kw('IF NOT EXISTS'))
        create_part.append(self._generate_node(node.name))
        parts.append(' '.join(create_part))
        
        if node.columns:
            parts[0] += f" ({', '.join(node.columns)})"
        
        parts.append(self._kw('AS'))
        parts.append(self._generate_node(node.query))
        
        return self._newline().join(parts)

    # ============== DROP Statement ==============
    
    def _gen_DropStatement(self, node) -> str:
        """Génère un statement DROP."""
        parts = [self._kw('DROP'), self._kw(node.object_type)]
        if node.if_exists:
            parts.append(self._kw('IF EXISTS'))
        parts.append(self._generate_node(node.name))
        if node.cascade:
            parts.append(self._kw('CASCADE'))
        return ' '.join(parts)

    # ============== ALTER TABLE Statement ==============
    
    def _gen_AlterTableStatement(self, node) -> str:
        """Génère un statement ALTER TABLE."""
        parts = [f"{self._kw('ALTER TABLE')} {self._generate_node(node.table)}"]
        
        actions = []
        for action in node.actions:
            actions.append(self._gen_AlterTableAction(action))
        
        parts.append(', '.join(actions))
        return ' '.join(parts)
    
    def _gen_AlterTableAction(self, node) -> str:
        """Génère une action ALTER TABLE."""
        if node.action_type == "ADD COLUMN":
            return f"{self._kw('ADD COLUMN')} {self._gen_ColumnDefinition(node.column)}"
        elif node.action_type == "DROP COLUMN":
            return f"{self._kw('DROP COLUMN')} {node.old_name}"
        elif node.action_type == "RENAME COLUMN":
            return f"{self._kw('RENAME COLUMN')} {node.old_name} {self._kw('TO')} {node.new_name}"
        elif node.action_type == "RENAME TABLE":
            return f"{self._kw('RENAME TO')} {node.new_name}"
        elif node.action_type == "MODIFY COLUMN":
            return f"{self._kw('MODIFY COLUMN')} {self._gen_ColumnDefinition(node.column)}"
        elif node.action_type == "ADD CONSTRAINT":
            return f"{self._kw('ADD')} {self._gen_TableConstraint(node.constraint)}"
        elif node.action_type == "DROP CONSTRAINT":
            return f"{self._kw('DROP CONSTRAINT')} {node.old_name}"
        else:
            return node.action_type

    # ============== TRUNCATE Statement ==============
    
    def _gen_TruncateStatement(self, node) -> str:
        """Génère un statement TRUNCATE TABLE."""
        return f"{self._kw('TRUNCATE TABLE')} {self._generate_node(node.table)}"


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
