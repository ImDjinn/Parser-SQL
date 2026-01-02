"""
Parser SQL - Analyseur syntaxique.

Convertit une séquence de tokens en un AST (Abstract Syntax Tree).
Supporte plusieurs dialectes SQL: Standard, Presto, Athena, Trino.
"""

from typing import List, Optional, Union, Set
from .tokenizer import SQLTokenizer, Token, TokenType
from .dialects import SQLDialect, get_dialect_features, detect_dialect
from .ast_nodes import (
    Expression, Literal, Identifier, ColumnRef, Star, Parameter,
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


class SQLParserError(Exception):
    """Erreur de parsing SQL."""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.token = token
        if token:
            super().__init__(f"{message} at line {token.line}, column {token.column}")
        else:
            super().__init__(message)


class SQLParser:
    """Parser SQL qui construit un AST à partir de tokens."""
    
    # Fonctions d'agrégation connues
    AGGREGATE_FUNCTIONS = {
        'count', 'sum', 'avg', 'min', 'max', 'group_concat', 'string_agg',
        # Presto/Athena
        'approx_distinct', 'approx_percentile', 'arbitrary', 'array_agg',
        'bool_and', 'bool_or', 'checksum', 'count_if', 'every',
        'geometric_mean', 'histogram', 'map_agg', 'map_union',
        'max_by', 'min_by', 'multimap_agg'
    }
    
    def __init__(self, dialect: SQLDialect = None):
        """
        Initialise le parser.
        
        Args:
            dialect: Dialecte SQL à utiliser. Si None, auto-détection.
        """
        self.dialect = dialect
        self.dialect_features = None
        self.tokens: List[Token] = []
        self.pos: int = 0
        self.original_sql: str = ""
        self.warnings: List[str] = []
        
        # Métadonnées collectées pendant le parsing
        self.tables_referenced: Set[str] = set()
        self.columns_referenced: Set[str] = set()
        self.functions_used: Set[str] = set()
        self.has_aggregation: bool = False
        self.has_subquery: bool = False
        self.has_join: bool = False
        self.has_jinja: bool = False
    
    def _reset(self):
        """Réinitialise l'état du parser."""
        self.pos = 0
        self.warnings = []
        self.tables_referenced = set()
        self.columns_referenced = set()
        self.functions_used = set()
        self.has_aggregation = False
        self.has_subquery = False
        self.has_join = False
        self.has_jinja = False
    
    def _current(self) -> Token:
        """Retourne le token courant."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self.pos]
    
    def _peek(self, offset: int = 1) -> Token:
        """Regarde le token à offset positions devant."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[pos]
    
    def _advance(self) -> Token:
        """Avance au prochain token et retourne le précédent."""
        token = self._current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token
    
    def _match(self, *token_types: TokenType) -> bool:
        """Vérifie si le token courant est d'un des types spécifiés."""
        return self._current().type in token_types
    
    def _check(self, token_type: TokenType) -> bool:
        """Vérifie si le token courant est du type spécifié."""
        return self._current().type == token_type
    
    def _expect(self, token_type: TokenType, message: str = None) -> Token:
        """Attend un token d'un type spécifique, lève une erreur sinon."""
        if not self._check(token_type):
            msg = message or f"Expected {token_type.name}"
            raise SQLParserError(msg, self._current())
        return self._advance()
    
    def _consume_if(self, token_type: TokenType) -> bool:
        """Consomme le token si c'est du type spécifié."""
        if self._check(token_type):
            self._advance()
            return True
        return False
    
    def parse(self, sql: str) -> ParseResult:
        """
        Parse une requête SQL et retourne le résultat.
        
        Args:
            sql: La requête SQL à parser
            
        Returns:
            ParseResult contenant l'AST et les métadonnées
        """
        self.original_sql = sql
        self._reset()
        
        # Tokenization
        tokenizer = SQLTokenizer(sql, include_whitespace=False, include_comments=False)
        self.tokens = tokenizer.tokenize()
        
        # Auto-détection du dialecte si non spécifié
        if self.dialect is None:
            self.dialect = detect_dialect(sql)
        self.dialect_features = get_dialect_features(self.dialect)
        
        # Parsing
        statement = self._parse_statement()
        
        # Construction du résultat
        parse_info = ParseInfo(
            original_sql=sql,
            warnings=self.warnings,
            implicit_info=self._extract_implicit_info(statement)
        )
        
        result = ParseResult(
            statement=statement,
            parse_info=parse_info,
            tables_referenced=list(self.tables_referenced),
            columns_referenced=list(self.columns_referenced),
            functions_used=list(self.functions_used),
            has_aggregation=self.has_aggregation,
            has_subquery=self.has_subquery,
            has_join=self.has_join
        )
        
        # Ajout des métadonnées de dialecte
        result.parse_info.implicit_info["dialect"] = self.dialect.value
        if self.has_jinja:
            result.parse_info.implicit_info["has_jinja_templates"] = True
        
        return result
    
    def _extract_implicit_info(self, statement: SelectStatement) -> dict:
        """Extrait les informations implicites de la requête."""
        info = {}
        
        # Vérifie si toutes les colonnes non-agrégées sont dans GROUP BY
        if self.has_aggregation and not statement.group_by:
            info["implicit_grouping"] = "Query has aggregation without GROUP BY - will return single row"
        
        # Vérifie l'ordre de tri par défaut
        if statement.order_by:
            for item in statement.order_by:
                if item.direction == OrderDirection.ASC:
                    info.setdefault("implicit_defaults", []).append(
                        f"ORDER BY uses implicit ASC direction"
                    )
                    break
        
        # INNER JOIN implicite
        if self.has_join:
            info["join_info"] = "Explicit JOIN used"
        elif statement.from_clause and len(statement.from_clause.tables) > 1:
            info["implicit_cross_join"] = "Multiple tables without JOIN implies CROSS JOIN"
        
        return info
    
    def _parse_statement(self) -> SelectStatement:
        """Parse un statement SQL (pour l'instant, uniquement SELECT)."""
        # Consommer les templates Jinja de configuration au début (dbt)
        jinja_config = []
        while self._match(TokenType.JINJA_EXPR, TokenType.JINJA_STMT, TokenType.JINJA_COMMENT):
            token = self._advance()
            if token.type in (TokenType.JINJA_EXPR, TokenType.JINJA_STMT):
                jinja_config.append(token.value)
            self.has_jinja = True
        
        # Gestion des CTE (WITH clause)
        ctes = None
        if self._check(TokenType.WITH):
            ctes = self._parse_cte_list()
        
        if self._check(TokenType.SELECT):
            statement = self._parse_select()
            statement.ctes = ctes
            # Stocker les configurations Jinja comme métadonnées
            if jinja_config:
                statement.metadata = statement.metadata or {}
                statement.metadata["jinja_config"] = jinja_config
            return statement
        else:
            raise SQLParserError("Expected SELECT statement", self._current())
    
    def _parse_cte_list(self) -> List[CTEDefinition]:
        """Parse la liste des CTEs."""
        self._expect(TokenType.WITH)
        ctes = []
        
        recursive = self._consume_if(TokenType.RECURSIVE)
        
        while True:
            cte = self._parse_cte_definition(recursive)
            ctes.append(cte)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return ctes
    
    def _parse_cte_definition(self, recursive: bool) -> CTEDefinition:
        """Parse une définition de CTE."""
        name_token = self._expect(TokenType.IDENTIFIER, "Expected CTE name")
        name = name_token.value
        
        # Liste optionnelle de colonnes
        columns = None
        if self._check(TokenType.LPAREN):
            columns = self._parse_identifier_list()
        
        self._expect(TokenType.AS, "Expected AS in CTE")
        self._expect(TokenType.LPAREN, "Expected ( before CTE query")
        
        query = self._parse_select()
        
        self._expect(TokenType.RPAREN, "Expected ) after CTE query")
        
        return CTEDefinition(name=name, query=query, columns=columns, recursive=recursive)
    
    def _parse_identifier_list(self) -> List[str]:
        """Parse une liste d'identifiants entre parenthèses."""
        self._expect(TokenType.LPAREN)
        identifiers = []
        
        while True:
            token = self._expect(TokenType.IDENTIFIER, "Expected identifier")
            identifiers.append(token.value)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        self._expect(TokenType.RPAREN)
        return identifiers
    
    def _parse_select(self) -> SelectStatement:
        """Parse une requête SELECT."""
        self._expect(TokenType.SELECT)
        
        # DISTINCT
        distinct = self._consume_if(TokenType.DISTINCT)
        if not distinct:
            self._consume_if(TokenType.ALL)  # ALL est le comportement par défaut
        
        # Liste des colonnes
        select_items = self._parse_select_items()
        
        # FROM clause (optionnelle pour certains dialects)
        from_clause = None
        if self._check(TokenType.FROM):
            from_clause = self._parse_from_clause()
        
        # WHERE clause
        where_clause = None
        if self._check(TokenType.WHERE):
            self._advance()
            where_clause = self._parse_expression()
        
        # GROUP BY clause
        group_by = None
        if self._check(TokenType.GROUP):
            group_by = self._parse_group_by()
        
        # HAVING clause
        having_clause = None
        if self._check(TokenType.HAVING):
            self._advance()
            having_clause = self._parse_expression()
        
        # ORDER BY clause
        order_by = None
        if self._check(TokenType.ORDER):
            order_by = self._parse_order_by()
        
        # LIMIT clause
        limit = None
        offset = None
        if self._check(TokenType.LIMIT):
            self._advance()
            limit = self._parse_primary_expression()
            
            # OFFSET peut suivre LIMIT
            if self._check(TokenType.OFFSET):
                self._advance()
                offset = self._parse_primary_expression()
            # Syntaxe alternative: LIMIT x, y (offset, limit)
            elif self._consume_if(TokenType.COMMA):
                offset = limit
                limit = self._parse_primary_expression()
        
        # OFFSET peut aussi être seul après LIMIT
        if self._check(TokenType.OFFSET) and offset is None:
            self._advance()
            offset = self._parse_primary_expression()
        
        statement = SelectStatement(
            select_items=select_items,
            from_clause=from_clause,
            where_clause=where_clause,
            group_by=group_by,
            having_clause=having_clause,
            order_by=order_by,
            limit=limit,
            offset=offset,
            distinct=distinct
        )
        
        # UNION, INTERSECT, EXCEPT
        set_op = self._parse_set_operation()
        if set_op:
            statement.set_operation, statement.set_query = set_op
        
        return statement
    
    def _parse_set_operation(self) -> Optional[tuple]:
        """Parse les opérations ensemblistes."""
        if self._check(TokenType.UNION):
            self._advance()
            all_modifier = self._consume_if(TokenType.ALL)
            op_type = SetOperationType.UNION_ALL if all_modifier else SetOperationType.UNION
            query = self._parse_select()
            return (op_type, query)
        
        if self._check(TokenType.INTERSECT):
            self._advance()
            self._consume_if(TokenType.ALL)
            query = self._parse_select()
            return (SetOperationType.INTERSECT, query)
        
        if self._check(TokenType.EXCEPT):
            self._advance()
            self._consume_if(TokenType.ALL)
            query = self._parse_select()
            return (SetOperationType.EXCEPT, query)
        
        return None
    
    def _parse_select_items(self) -> List[SelectItem]:
        """Parse la liste des éléments SELECT."""
        items = []
        
        while True:
            item = self._parse_select_item()
            items.append(item)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return items
    
    def _parse_select_item(self) -> SelectItem:
        """Parse un élément SELECT (expression AS alias)."""
        # Cas spécial: *
        if self._check(TokenType.STAR):
            self._advance()
            return SelectItem(expression=Star())
        
        # Cas spécial: table.*
        if self._check(TokenType.IDENTIFIER) and self._peek().type == TokenType.DOT:
            if self._peek(2).type == TokenType.STAR:
                table_token = self._advance()
                self._advance()  # .
                self._advance()  # *
                return SelectItem(expression=Star(table=table_token.value))
        
        # Expression normale
        expr = self._parse_expression()
        
        # Alias optionnel
        alias = None
        if self._consume_if(TokenType.AS):
            alias_token = self._expect(TokenType.IDENTIFIER, "Expected alias name")
            alias = alias_token.value
        elif self._check(TokenType.IDENTIFIER):
            # Alias sans AS
            alias_token = self._advance()
            alias = alias_token.value
        
        return SelectItem(expression=expr, alias=alias)
    
    def _parse_from_clause(self) -> FromClause:
        """Parse la clause FROM."""
        self._expect(TokenType.FROM)
        tables = []
        
        # Première table
        table = self._parse_table_ref()
        tables.append(table)
        
        # JOINs ou tables supplémentaires
        while True:
            join = self._try_parse_join()
            if join:
                # Le JOIN référence la dernière table et la nouvelle
                tables.append(join)
                continue
            
            # Tables séparées par virgule (CROSS JOIN implicite)
            if self._consume_if(TokenType.COMMA):
                table = self._parse_table_ref()
                tables.append(table)
                continue
            
            break
        
        return FromClause(tables=tables)
    
    def _parse_table_ref(self) -> Union[TableRef, SubqueryRef, UnnestRef]:
        """Parse une référence de table, sous-requête, ou UNNEST."""
        
        # UNNEST (Presto/Athena)
        if self._check(TokenType.UNNEST):
            return self._parse_unnest_ref()
        
        # LATERAL (Presto/Athena)
        if self._check(TokenType.LATERAL):
            self._advance()
            return self._parse_table_ref()
        
        # Sous-requête
        if self._check(TokenType.LPAREN):
            self._advance()
            
            if self._check(TokenType.SELECT):
                query = self._parse_select()
                self._expect(TokenType.RPAREN)
                self.has_subquery = True
                
                # Alias obligatoire pour sous-requête
                alias = None
                self._consume_if(TokenType.AS)
                if self._check(TokenType.IDENTIFIER):
                    alias = self._advance().value
                else:
                    self.warnings.append("Subquery in FROM should have an alias")
                    alias = "_subquery"
                
                return SubqueryRef(query=query, alias=alias)
            else:
                raise SQLParserError("Expected SELECT in subquery", self._current())
        
        # Template Jinja comme référence de table (dbt: {{ ref('...') }}, {{ source(...) }})
        if self._match(TokenType.JINJA_EXPR, TokenType.JINJA_STMT):
            jinja_token = self._advance()
            self.has_jinja = True
            
            # Traiter comme une référence de table avec le template comme nom
            name = jinja_token.value
            self.tables_referenced.add(name)
            
            # Alias optionnel
            alias = None
            if self._consume_if(TokenType.AS):
                alias = self._expect(TokenType.IDENTIFIER, "Expected alias").value
            elif self._check(TokenType.IDENTIFIER) and not self._is_keyword(self._current()):
                alias = self._advance().value
            
            return TableRef(name=name, alias=alias, schema=None, is_jinja=True)
        
        # Table normale
        # Gestion du schéma: schema.table
        name_token = self._expect(TokenType.IDENTIFIER, "Expected table name")
        schema = None
        name = name_token.value
        
        if self._consume_if(TokenType.DOT):
            schema = name
            name_token = self._expect(TokenType.IDENTIFIER, "Expected table name after schema")
            name = name_token.value
        
        self.tables_referenced.add(name)
        
        # Alias
        alias = None
        if self._consume_if(TokenType.AS):
            alias = self._expect(TokenType.IDENTIFIER, "Expected alias").value
        elif self._check(TokenType.IDENTIFIER) and not self._is_keyword(self._current()):
            alias = self._advance().value
        
        return TableRef(name=name, alias=alias, schema=schema)
    
    def _parse_unnest_ref(self) -> UnnestRef:
        """Parse UNNEST(...) [WITH ORDINALITY] (Presto/Athena)."""
        self._expect(TokenType.UNNEST)
        self._expect(TokenType.LPAREN)
        
        expr = self._parse_expression()
        
        self._expect(TokenType.RPAREN)
        
        # WITH ORDINALITY
        with_ordinality = False
        ordinality_alias = None
        if self._check(TokenType.WITH):
            self._advance()
            if self._check(TokenType.ORDINALITY):
                self._advance()
                with_ordinality = True
        
        # Alias
        alias = None
        self._consume_if(TokenType.AS)
        if self._check(TokenType.IDENTIFIER):
            alias = self._advance().value
            
            # Alias pour ordinality: AS t(elem, ord)
            if self._check(TokenType.LPAREN):
                self._advance()
                # Ignore les noms de colonnes pour l'instant
                while not self._check(TokenType.RPAREN):
                    self._advance()
                    self._consume_if(TokenType.COMMA)
                self._expect(TokenType.RPAREN)
        
        return UnnestRef(
            expression=expr,
            alias=alias,
            with_ordinality=with_ordinality,
            ordinality_alias=ordinality_alias
        )
    
    def _is_keyword(self, token: Token) -> bool:
        """Vérifie si un token IDENTIFIER est en fait un mot-clé contextuel."""
        keywords = {
            TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT,
            TokenType.FULL, TokenType.OUTER, TokenType.CROSS, TokenType.NATURAL,
            TokenType.ON, TokenType.WHERE, TokenType.GROUP, TokenType.ORDER,
            TokenType.HAVING, TokenType.LIMIT, TokenType.OFFSET, TokenType.UNION,
            TokenType.INTERSECT, TokenType.EXCEPT, TokenType.LATERAL, TokenType.UNNEST
        }
        return token.type in keywords
    
    def _try_parse_join(self) -> Optional[JoinClause]:
        """Tente de parser une clause JOIN."""
        join_type = None
        
        # Détermine le type de JOIN
        if self._consume_if(TokenType.NATURAL):
            self._consume_if(TokenType.INNER)
            join_type = JoinType.NATURAL
        elif self._consume_if(TokenType.CROSS):
            self._expect(TokenType.JOIN)
            join_type = JoinType.CROSS
        elif self._consume_if(TokenType.INNER):
            self._expect(TokenType.JOIN)
            join_type = JoinType.INNER
        elif self._consume_if(TokenType.LEFT):
            self._consume_if(TokenType.OUTER)
            self._expect(TokenType.JOIN)
            join_type = JoinType.LEFT
        elif self._consume_if(TokenType.RIGHT):
            self._consume_if(TokenType.OUTER)
            self._expect(TokenType.JOIN)
            join_type = JoinType.RIGHT
        elif self._consume_if(TokenType.FULL):
            self._consume_if(TokenType.OUTER)
            self._expect(TokenType.JOIN)
            join_type = JoinType.FULL
        elif self._consume_if(TokenType.JOIN):
            join_type = JoinType.INNER  # JOIN seul = INNER JOIN
        
        if join_type is None:
            return None
        
        self.has_join = True
        
        # Table jointe
        table = self._parse_table_ref()
        
        # Condition de jointure
        condition = None
        using_columns = None
        
        if join_type != JoinType.CROSS and join_type != JoinType.NATURAL:
            if self._consume_if(TokenType.ON):
                condition = self._parse_expression()
            elif self._consume_if(TokenType.USING):
                using_columns = self._parse_identifier_list()
        
        return JoinClause(
            join_type=join_type,
            table=table,
            condition=condition,
            using_columns=using_columns
        )
    
    def _parse_group_by(self) -> List[Expression]:
        """Parse la clause GROUP BY."""
        self._expect(TokenType.GROUP)
        self._expect(TokenType.BY)
        
        expressions = []
        while True:
            expr = self._parse_expression()
            expressions.append(expr)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return expressions
    
    def _parse_order_by(self) -> List[OrderByItem]:
        """Parse la clause ORDER BY."""
        self._expect(TokenType.ORDER)
        self._expect(TokenType.BY)
        
        items = []
        while True:
            expr = self._parse_expression()
            
            # Direction
            direction = OrderDirection.ASC
            if self._consume_if(TokenType.ASC):
                direction = OrderDirection.ASC
            elif self._consume_if(TokenType.DESC):
                direction = OrderDirection.DESC
            
            # NULLS FIRST/LAST
            nulls_first = None
            if self._consume_if(TokenType.NULLS):
                if self._consume_if(TokenType.FIRST):
                    nulls_first = True
                elif self._consume_if(TokenType.LAST):
                    nulls_first = False
            
            items.append(OrderByItem(
                expression=expr,
                direction=direction,
                nulls_first=nulls_first
            ))
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return items
    
    # ============== Parsing des expressions ==============
    
    def _parse_expression(self) -> Expression:
        """Parse une expression (point d'entrée)."""
        return self._parse_or_expression()
    
    def _parse_or_expression(self) -> Expression:
        """Parse une expression OR."""
        left = self._parse_and_expression()
        
        while self._consume_if(TokenType.OR):
            right = self._parse_and_expression()
            left = BinaryOp(left=left, operator="OR", right=right)
        
        return left
    
    def _parse_and_expression(self) -> Expression:
        """Parse une expression AND."""
        left = self._parse_not_expression()
        
        while self._consume_if(TokenType.AND):
            right = self._parse_not_expression()
            left = BinaryOp(left=left, operator="AND", right=right)
        
        return left
    
    def _parse_not_expression(self) -> Expression:
        """Parse une expression NOT."""
        if self._consume_if(TokenType.NOT):
            operand = self._parse_not_expression()
            return UnaryOp(operator="NOT", operand=operand)
        
        return self._parse_comparison_expression()
    
    def _parse_comparison_expression(self) -> Expression:
        """Parse une expression de comparaison."""
        left = self._parse_additive_expression()
        
        # IS NULL / IS NOT NULL
        if self._consume_if(TokenType.IS):
            negated = self._consume_if(TokenType.NOT)
            if self._consume_if(TokenType.NULL):
                return IsNullExpression(expression=left, negated=negated)
            else:
                raise SQLParserError("Expected NULL after IS", self._current())
        
        # IN
        negated = False
        if self._check(TokenType.NOT) and self._peek().type == TokenType.IN:
            self._advance()
            negated = True
        
        if self._consume_if(TokenType.IN):
            return self._parse_in_expression(left, negated)
        
        # BETWEEN
        if self._check(TokenType.NOT) and self._peek().type == TokenType.BETWEEN:
            self._advance()
            negated = True
        
        if self._consume_if(TokenType.BETWEEN):
            low = self._parse_additive_expression()
            self._expect(TokenType.AND, "Expected AND in BETWEEN")
            high = self._parse_additive_expression()
            return BetweenExpression(expression=left, low=low, high=high, negated=negated)
        
        # LIKE
        if self._check(TokenType.NOT) and self._peek().type == TokenType.LIKE:
            self._advance()
            negated = True
        
        if self._consume_if(TokenType.LIKE):
            pattern = self._parse_additive_expression()
            return LikeExpression(expression=left, pattern=pattern, negated=negated)
        
        # Opérateurs de comparaison standards
        if self._match(TokenType.EQUALS, TokenType.NOT_EQUALS, TokenType.LESS_THAN,
                       TokenType.GREATER_THAN, TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL):
            op_token = self._advance()
            op_map = {
                TokenType.EQUALS: "=",
                TokenType.NOT_EQUALS: "<>",
                TokenType.LESS_THAN: "<",
                TokenType.GREATER_THAN: ">",
                TokenType.LESS_EQUAL: "<=",
                TokenType.GREATER_EQUAL: ">="
            }
            operator = op_map[op_token.type]
            right = self._parse_additive_expression()
            return BinaryOp(left=left, operator=operator, right=right)
        
        return left
    
    def _parse_in_expression(self, left: Expression, negated: bool) -> InExpression:
        """Parse une expression IN."""
        self._expect(TokenType.LPAREN)
        
        # Sous-requête ou liste de valeurs?
        if self._check(TokenType.SELECT):
            self.has_subquery = True
            subquery = self._parse_select()
            self._expect(TokenType.RPAREN)
            return InExpression(expression=left, subquery=subquery, negated=negated)
        
        # Liste de valeurs
        values = []
        while True:
            value = self._parse_expression()
            values.append(value)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        self._expect(TokenType.RPAREN)
        return InExpression(expression=left, values=values, negated=negated)
    
    def _parse_additive_expression(self) -> Expression:
        """Parse une expression additive (+, -, ||)."""
        left = self._parse_multiplicative_expression()
        
        while self._match(TokenType.PLUS, TokenType.MINUS, TokenType.CONCAT):
            op_token = self._advance()
            op_map = {
                TokenType.PLUS: "+",
                TokenType.MINUS: "-",
                TokenType.CONCAT: "||"
            }
            operator = op_map[op_token.type]
            right = self._parse_multiplicative_expression()
            left = BinaryOp(left=left, operator=operator, right=right)
        
        return left
    
    def _parse_multiplicative_expression(self) -> Expression:
        """Parse une expression multiplicative (*, /, %)."""
        left = self._parse_unary_expression()
        
        while self._match(TokenType.STAR, TokenType.DIVIDE, TokenType.MODULO):
            op_token = self._advance()
            op_map = {
                TokenType.STAR: "*",
                TokenType.DIVIDE: "/",
                TokenType.MODULO: "%"
            }
            operator = op_map[op_token.type]
            right = self._parse_unary_expression()
            left = BinaryOp(left=left, operator=operator, right=right)
        
        return left
    
    def _parse_unary_expression(self) -> Expression:
        """Parse une expression unaire (-, +)."""
        if self._match(TokenType.MINUS, TokenType.PLUS):
            op_token = self._advance()
            operand = self._parse_unary_expression()
            return UnaryOp(operator=op_token.value, operand=operand)
        
        # Parse expression primaire puis expressions postfixes
        expr = self._parse_primary_expression()
        return self._parse_postfix_expression(expr)
    
    def _parse_primary_expression(self) -> Expression:
        """Parse une expression primaire."""
        token = self._current()
        
        # Jinja templates (dbt)
        if self._match(TokenType.JINJA_EXPR, TokenType.JINJA_STMT, TokenType.JINJA_COMMENT):
            return self._parse_jinja_expression()
        
        # EXISTS
        if self._consume_if(TokenType.EXISTS):
            self._expect(TokenType.LPAREN)
            self.has_subquery = True
            subquery = self._parse_select()
            self._expect(TokenType.RPAREN)
            return ExistsExpression(subquery=subquery)
        
        # CASE expression
        if self._check(TokenType.CASE):
            return self._parse_case_expression()
        
        # ARRAY[] (Presto/Athena)
        if self._check(TokenType.ARRAY):
            return self._parse_array_expression()
        
        # MAP() (Presto/Athena)
        if self._check(TokenType.MAP):
            return self._parse_map_expression()
        
        # ROW() (Presto/Athena)
        if self._check(TokenType.ROW):
            return self._parse_row_expression()
        
        # INTERVAL (Presto/Athena)
        if self._check(TokenType.INTERVAL):
            return self._parse_interval_expression()
        
        # TRY() ou TRY_CAST() (Presto/Athena)
        if self._check(TokenType.TRY):
            return self._parse_try_expression()
        
        # IF() (Presto/Athena)
        if self._check(TokenType.IF):
            return self._parse_if_expression()
        
        # Littéraux typés: DATE 'valeur', TIMESTAMP 'valeur', TIME 'valeur'
        if self._match(TokenType.DATE, TokenType.TIMESTAMP, TokenType.TIME):
            type_token = self._advance()
            type_name = type_token.value.upper()
            
            # Peut être suivi d'une string littérale ou d'un template Jinja
            if self._check(TokenType.STRING):
                raw_value = self._advance().value
                value = raw_value[1:-1] if len(raw_value) >= 2 else raw_value
                return Literal(value=value, literal_type=type_name.lower())
            elif self._match(TokenType.JINJA_EXPR, TokenType.JINJA_STMT):
                jinja_expr = self._parse_jinja_expression()
                # Créer un CAST implicite du Jinja vers le type
                return CastExpression(expression=jinja_expr, target_type=type_name)
            else:
                # DATE sans littéral - traiter comme identifiant
                return Identifier(name=type_name)
        
        # Parenthèses ou sous-requête
        if self._consume_if(TokenType.LPAREN):
            if self._check(TokenType.SELECT):
                self.has_subquery = True
                subquery = self._parse_select()
                self._expect(TokenType.RPAREN)
                return SubqueryExpression(query=subquery)
            else:
                expr = self._parse_expression()
                self._expect(TokenType.RPAREN)
                return expr
        
        # Littéraux
        if self._check(TokenType.INTEGER):
            value = int(self._advance().value)
            return Literal(value=value, literal_type="integer")
        
        if self._check(TokenType.FLOAT):
            value = float(self._advance().value)
            return Literal(value=value, literal_type="float")
        
        if self._check(TokenType.STRING):
            raw_value = self._advance().value
            # Retire les guillemets
            value = raw_value[1:-1] if len(raw_value) >= 2 else raw_value
            return Literal(value=value, literal_type="string")
        
        if self._consume_if(TokenType.NULL):
            return Literal(value=None, literal_type="null")
        
        if self._consume_if(TokenType.TRUE):
            return Literal(value=True, literal_type="boolean")
        
        if self._consume_if(TokenType.FALSE):
            return Literal(value=False, literal_type="boolean")
        
        # Paramètre
        if self._consume_if(TokenType.QUESTION):
            return Parameter()
        
        if self._consume_if(TokenType.COLON):
            name_token = self._expect(TokenType.IDENTIFIER)
            return Parameter(name=name_token.value)
        
        # Fonctions d'agrégation ou autres
        if self._match(TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX):
            return self._parse_aggregate_function()
        
        # Identifiant ou appel de fonction
        if self._check(TokenType.IDENTIFIER) or self._check(TokenType.QUOTED_IDENTIFIER):
            return self._parse_identifier_or_function()
        
        # Mots-clés qui peuvent être utilisés comme noms de fonction en Presto/Athena
        # (filter, left, right, etc.) - si suivi par LPAREN
        if self._is_callable_keyword():
            return self._parse_identifier_or_function()
        
        # STAR seul (dans certains contextes)
        if self._check(TokenType.STAR):
            self._advance()
            return Star()
        
        raise SQLParserError(f"Unexpected token: {token.type.name}", token)
    
    def _is_callable_keyword(self) -> bool:
        """Vérifie si le token courant est un mot-clé utilisable comme fonction."""
        # Vérifie si le token actuel est suivi par LPAREN (appel de fonction)
        if self._peek().type != TokenType.LPAREN:
            return False
            
        # Liste des mots-clés qui sont aussi des fonctions en Presto/Athena
        callable_keywords = {
            TokenType.FILTER,      # filter(array, x -> ...)
            TokenType.LEFT,        # left(string, n)
            TokenType.RIGHT,       # right(string, n)
        }
        return self._current().type in callable_keywords
    
    def _parse_aggregate_function(self) -> FunctionCall:
        """Parse une fonction d'agrégation."""
        func_token = self._advance()
        func_name = func_token.value.upper()
        
        self._expect(TokenType.LPAREN)
        
        self.has_aggregation = True
        self.functions_used.add(func_name)
        
        distinct = self._consume_if(TokenType.DISTINCT)
        
        args = []
        if self._check(TokenType.STAR):
            self._advance()
            args.append(Star())
        elif not self._check(TokenType.RPAREN):
            while True:
                arg = self._parse_expression()
                args.append(arg)
                if not self._consume_if(TokenType.COMMA):
                    break
        
        self._expect(TokenType.RPAREN)
        
        return FunctionCall(name=func_name, args=args, distinct=distinct)
    
    def _parse_identifier_or_function(self) -> Expression:
        """Parse un identifiant ou un appel de fonction."""
        token = self._advance()
        name = token.value
        quoted = token.type == TokenType.QUOTED_IDENTIFIER
        
        if quoted:
            # Retire les guillemets
            name = name[1:-1]
        
        # Appel de fonction?
        if self._check(TokenType.LPAREN):
            self._advance()
            
            func_name = name.upper()
            self.functions_used.add(func_name)
            
            # CAST a une syntaxe spéciale: CAST(expr AS type)
            if func_name == 'CAST':
                expr = self._parse_expression()
                self._expect(TokenType.AS, "Expected AS in CAST expression")
                target_type = self._parse_type_name()
                self._expect(TokenType.RPAREN)
                return CastExpression(expression=expr, target_type=target_type)
            
            if func_name.lower() in self.AGGREGATE_FUNCTIONS:
                self.has_aggregation = True
            
            distinct = self._consume_if(TokenType.DISTINCT)
            
            args = []
            if not self._check(TokenType.RPAREN):
                if self._check(TokenType.STAR):
                    self._advance()
                    args.append(Star())
                else:
                    while True:
                        arg = self._parse_expression()
                        args.append(arg)
                        if not self._consume_if(TokenType.COMMA):
                            break
            
            self._expect(TokenType.RPAREN)
            return FunctionCall(name=func_name, args=args, distinct=distinct)
        
        # Référence de colonne avec table/schema
        if self._consume_if(TokenType.DOT):
            # C'est une référence table.colonne ou schema.table.colonne
            second_name_token = self._advance()
            second_name = second_name_token.value
            
            if self._consume_if(TokenType.DOT):
                # schema.table.colonne
                third_name_token = self._advance()
                column = third_name_token.value
                self.columns_referenced.add(column)
                return ColumnRef(column=column, table=second_name, schema=name, quoted=quoted)
            else:
                # table.colonne
                self.columns_referenced.add(second_name)
                return ColumnRef(column=second_name, table=name, quoted=quoted)
        
        # Simple identifiant (colonne)
        self.columns_referenced.add(name)
        return ColumnRef(column=name, quoted=quoted)
    
    def _parse_case_expression(self) -> CaseExpression:
        """Parse une expression CASE."""
        self._expect(TokenType.CASE)
        
        # CASE simple (CASE expr WHEN ...) ou CASE recherché (CASE WHEN cond ...)
        operand = None
        if not self._check(TokenType.WHEN):
            operand = self._parse_expression()
        
        when_clauses = []
        while self._consume_if(TokenType.WHEN):
            condition = self._parse_expression()
            self._expect(TokenType.THEN)
            result = self._parse_expression()
            when_clauses.append((condition, result))
        
        else_clause = None
        if self._consume_if(TokenType.ELSE):
            else_clause = self._parse_expression()
        
        self._expect(TokenType.END)
        
        return CaseExpression(
            operand=operand,
            when_clauses=when_clauses,
            else_clause=else_clause
        )
    
    def _parse_type_name(self) -> str:
        """Parse un nom de type SQL (ex: DOUBLE, VARCHAR(255), ARRAY<INT>)."""
        # Types Presto/Athena: DOUBLE, VARCHAR, BIGINT, ARRAY<T>, MAP<K,V>, ROW(...)
        type_parts = []
        
        # Nom de type principal - accepter identifiants et certains mots-clés
        token = self._current()
        if token.type in (TokenType.IDENTIFIER, TokenType.ARRAY, TokenType.MAP, 
                          TokenType.ROW, TokenType.INTERVAL, TokenType.DATE,
                          TokenType.TIMESTAMP, TokenType.TIME):
            type_parts.append(self._advance().value.upper())
        else:
            # Accepter d'autres tokens comme noms de type (DOUBLE, INTEGER, etc.)
            type_parts.append(self._advance().value.upper())
        
        # Précision/échelle optionnelle: (10) ou (10, 2)
        if self._check(TokenType.LPAREN):
            self._advance()
            type_parts.append('(')
            
            # Lire jusqu'à RPAREN, en gérant les types imbriqués
            depth = 1
            while depth > 0 and not self._check(TokenType.EOF):
                token = self._current()
                if token.type == TokenType.LPAREN:
                    depth += 1
                elif token.type == TokenType.RPAREN:
                    depth -= 1
                    if depth == 0:
                        break
                type_parts.append(token.value)
                self._advance()
            
            type_parts.append(')')
            self._expect(TokenType.RPAREN)
        
        # Types génériques: ARRAY<INT>, MAP<STRING, INT>
        if self._check(TokenType.LESS_THAN):
            self._advance()
            type_parts.append('<')
            
            depth = 1
            while depth > 0 and not self._check(TokenType.EOF):
                token = self._current()
                if token.type == TokenType.LESS_THAN:
                    depth += 1
                elif token.type == TokenType.GREATER_THAN:
                    depth -= 1
                    if depth == 0:
                        break
                type_parts.append(token.value)
                self._advance()
            
            type_parts.append('>')
            self._expect(TokenType.GREATER_THAN)
        
        return ''.join(type_parts)
    
    # ============== Presto/Athena specific parsing ==============
    
    def _parse_jinja_expression(self) -> JinjaExpression:
        """Parse une expression Jinja (dbt)."""
        token = self._advance()
        self.has_jinja = True
        
        jinja_type_map = {
            TokenType.JINJA_EXPR: 'expression',
            TokenType.JINJA_STMT: 'statement', 
            TokenType.JINJA_COMMENT: 'comment'
        }
        
        return JinjaExpression(
            content=token.value,
            jinja_type=jinja_type_map.get(token.type, 'expression')
        )
    
    def _parse_array_expression(self) -> ArrayExpression:
        """Parse ARRAY[...] (Presto/Athena)."""
        self._expect(TokenType.ARRAY)
        self._expect(TokenType.LBRACKET)
        
        elements = []
        if not self._check(TokenType.RBRACKET):
            while True:
                elem = self._parse_expression()
                elements.append(elem)
                if not self._consume_if(TokenType.COMMA):
                    break
        
        self._expect(TokenType.RBRACKET)
        return ArrayExpression(elements=elements)
    
    def _parse_map_expression(self) -> MapExpression:
        """Parse MAP(...) (Presto/Athena)."""
        self._expect(TokenType.MAP)
        self._expect(TokenType.LPAREN)
        
        keys = []
        values = []
        
        if not self._check(TokenType.RPAREN):
            # MAP(ARRAY[k1, k2], ARRAY[v1, v2]) ou MAP(k1, v1, k2, v2)
            first_arg = self._parse_expression()
            
            if isinstance(first_arg, ArrayExpression):
                # Format MAP(ARRAY[], ARRAY[])
                keys_array = first_arg
                self._expect(TokenType.COMMA)
                values_array = self._parse_expression()
                
                if isinstance(values_array, ArrayExpression):
                    keys = keys_array.elements
                    values = values_array.elements
            else:
                # Format MAP(k1, v1, k2, v2, ...)
                args = [first_arg]
                while self._consume_if(TokenType.COMMA):
                    args.append(self._parse_expression())
                
                # Pairs of key, value
                for i in range(0, len(args), 2):
                    keys.append(args[i])
                    if i + 1 < len(args):
                        values.append(args[i + 1])
        
        self._expect(TokenType.RPAREN)
        return MapExpression(keys=keys, values=values)
    
    def _parse_row_expression(self) -> RowExpression:
        """Parse ROW(...) (Presto/Athena)."""
        self._expect(TokenType.ROW)
        self._expect(TokenType.LPAREN)
        
        fields = []
        if not self._check(TokenType.RPAREN):
            while True:
                field = self._parse_expression()
                fields.append(field)
                if not self._consume_if(TokenType.COMMA):
                    break
        
        self._expect(TokenType.RPAREN)
        return RowExpression(fields=fields)
    
    def _parse_interval_expression(self) -> IntervalExpression:
        """Parse INTERVAL '1' DAY (Presto/Athena)."""
        self._expect(TokenType.INTERVAL)
        
        value = self._parse_primary_expression()
        
        # Unit (DAY, HOUR, MINUTE, SECOND, MONTH, YEAR)
        unit_token = self._current()
        if unit_token.type == TokenType.IDENTIFIER:
            unit = self._advance().value.upper()
        else:
            unit = "DAY"  # Default
        
        return IntervalExpression(value=value, unit=unit)
    
    def _parse_try_expression(self) -> Expression:
        """Parse TRY(...) ou TRY_CAST(...) (Presto/Athena)."""
        self._expect(TokenType.TRY)
        
        # Check for TRY_CAST
        if self._current().type == TokenType.IDENTIFIER and self._current().value.upper() == '_CAST':
            # Actually it would be parsed as TRY identifier, need different approach
            pass
        
        self._expect(TokenType.LPAREN)
        expr = self._parse_expression()
        self._expect(TokenType.RPAREN)
        
        return TryExpression(expression=expr)
    
    def _parse_if_expression(self) -> IfExpression:
        """Parse IF(cond, then, else) (Presto/Athena)."""
        self._expect(TokenType.IF)
        self._expect(TokenType.LPAREN)
        
        condition = self._parse_expression()
        self._expect(TokenType.COMMA)
        
        then_expr = self._parse_expression()
        
        else_expr = None
        if self._consume_if(TokenType.COMMA):
            else_expr = self._parse_expression()
        
        self._expect(TokenType.RPAREN)
        
        return IfExpression(
            condition=condition,
            then_expr=then_expr,
            else_expr=else_expr
        )
    
    def _parse_lambda_expression(self, first_param: str) -> LambdaExpression:
        """Parse une lambda: x -> x + 1 (Presto/Athena)."""
        # Le premier paramètre a déjà été parsé
        parameters = [first_param]
        
        # Si on a des parenthèses, il peut y avoir plusieurs paramètres
        # (x, y) -> x + y
        
        self._expect(TokenType.ARROW)
        body = self._parse_expression()
        
        return LambdaExpression(parameters=parameters, body=body)
    
    def _parse_postfix_expression(self, expr: Expression) -> Expression:
        """Parse les expressions postfixes: array[i], expr AT TIME ZONE, etc."""
        while True:
            # Array subscript: arr[i]
            if self._check(TokenType.LBRACKET):
                self._advance()
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET)
                expr = ArraySubscript(array=expr, index=index)
                continue
            
            # AT TIME ZONE
            if self._check(TokenType.AT):
                if self._peek().type == TokenType.TIME:
                    self._advance()  # AT
                    self._advance()  # TIME
                    self._expect(TokenType.ZONE)
                    timezone = self._parse_primary_expression()
                    expr = AtTimeZone(expression=expr, timezone=timezone)
                    continue
            
            # Lambda arrow (dans contexte de fonction comme transform, filter)
            if self._check(TokenType.ARROW):
                if isinstance(expr, ColumnRef):
                    return self._parse_lambda_expression(expr.column)
            
            break
        
        return expr


def parse(sql: str, dialect: SQLDialect = None) -> ParseResult:
    """
    Fonction utilitaire pour parser du SQL.
    
    Args:
        sql: Le code SQL à parser
        dialect: Dialecte SQL (auto-détecté si None)
        
    Returns:
        ParseResult contenant l'AST et les métadonnées
    """
    parser = SQLParser(dialect=dialect)
    return parser.parse(sql)
