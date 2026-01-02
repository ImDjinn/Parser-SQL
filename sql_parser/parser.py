"""
Parser SQL - Analyseur syntaxique.

Convertit une séquence de tokens en un AST (Abstract Syntax Tree).
"""

from typing import List, Optional, Union, Set
from .tokenizer import SQLTokenizer, Token, TokenType
from .ast_nodes import (
    Expression, Literal, Identifier, ColumnRef, Star, Parameter,
    BinaryOp, UnaryOp, FunctionCall, CaseExpression,
    InExpression, BetweenExpression, LikeExpression, IsNullExpression,
    ExistsExpression, SubqueryExpression, CastExpression,
    SelectItem, TableRef, SubqueryRef, JoinClause, FromClause,
    JoinType, OrderDirection, SetOperationType, OrderByItem,
    CTEDefinition, SelectStatement, ParseInfo, ParseResult
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
    AGGREGATE_FUNCTIONS = {'count', 'sum', 'avg', 'min', 'max', 'group_concat', 'string_agg'}
    
    def __init__(self):
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
        
        # Parsing
        statement = self._parse_statement()
        
        # Construction du résultat
        parse_info = ParseInfo(
            original_sql=sql,
            warnings=self.warnings,
            implicit_info=self._extract_implicit_info(statement)
        )
        
        return ParseResult(
            statement=statement,
            parse_info=parse_info,
            tables_referenced=list(self.tables_referenced),
            columns_referenced=list(self.columns_referenced),
            functions_used=list(self.functions_used),
            has_aggregation=self.has_aggregation,
            has_subquery=self.has_subquery,
            has_join=self.has_join
        )
    
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
        # Gestion des CTE (WITH clause)
        ctes = None
        if self._check(TokenType.WITH):
            ctes = self._parse_cte_list()
        
        if self._check(TokenType.SELECT):
            statement = self._parse_select()
            statement.ctes = ctes
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
    
    def _parse_table_ref(self) -> Union[TableRef, SubqueryRef]:
        """Parse une référence de table ou sous-requête."""
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
    
    def _is_keyword(self, token: Token) -> bool:
        """Vérifie si un token IDENTIFIER est en fait un mot-clé contextuel."""
        keywords = {
            TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT,
            TokenType.FULL, TokenType.OUTER, TokenType.CROSS, TokenType.NATURAL,
            TokenType.ON, TokenType.WHERE, TokenType.GROUP, TokenType.ORDER,
            TokenType.HAVING, TokenType.LIMIT, TokenType.OFFSET, TokenType.UNION,
            TokenType.INTERSECT, TokenType.EXCEPT
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
        
        return self._parse_primary_expression()
    
    def _parse_primary_expression(self) -> Expression:
        """Parse une expression primaire."""
        token = self._current()
        
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
        
        # STAR seul (dans certains contextes)
        if self._check(TokenType.STAR):
            self._advance()
            return Star()
        
        raise SQLParserError(f"Unexpected token: {token.type.name}", token)
    
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


def parse(sql: str) -> ParseResult:
    """
    Fonction utilitaire pour parser du SQL.
    
    Args:
        sql: Le code SQL à parser
        
    Returns:
        ParseResult contenant l'AST et les métadonnées
    """
    parser = SQLParser()
    return parser.parse(sql)
