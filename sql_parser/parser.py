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
    BinaryOp, UnaryOp, FunctionCall, CaseExpression, NamedArgument, KeyValuePair,
    InExpression, BetweenExpression, LikeExpression, IsNullExpression,
    ExistsExpression, SubqueryExpression, CastExpression, ExtractExpression,
    QuantifiedComparison,
    SelectItem, TableRef, SubqueryRef, JoinClause, FromClause,
    JoinType, OrderDirection, SetOperationType, OrderByItem,
    CTEDefinition, SelectStatement, ParseInfo, ParseResult,
    GroupingSets, Cube, Rollup, WindowDefinition,
    # Presto/Athena specific
    ArrayExpression, MapExpression, RowExpression, ArraySubscript,
    LambdaExpression, IntervalExpression, AtTimeZone, TryExpression,
    IfExpression, JinjaExpression, WindowFunction, UnnestRef, TableSample,
    # DML statements
    InsertStatement, UpdateStatement, DeleteStatement, MergeStatement,
    Assignment, OnConflictClause, MergeWhenClause,
    # DDL statements
    CreateTableStatement, CreateViewStatement, DropStatement,
    AlterTableStatement, AlterTableAction, TruncateStatement,
    ColumnDefinition, TableConstraint
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
    
    def _extract_implicit_info(self, statement) -> dict:
        """Extrait les informations implicites de la requête."""
        info = {}
        
        # Uniquement pour SELECT
        if not isinstance(statement, SelectStatement):
            info["statement_type"] = type(statement).__name__
            return info
        
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
    
    def _parse_statement(self):
        """Parse un statement SQL (SELECT, INSERT, UPDATE, DELETE, MERGE, DDL)."""
        # Consommer les templates Jinja de configuration au début (dbt)
        jinja_config_raw = []
        parsed_config = {}
        
        while self._match(TokenType.JINJA_EXPR, TokenType.JINJA_STMT, TokenType.JINJA_COMMENT):
            token = self._advance()
            if token.type in (TokenType.JINJA_EXPR, TokenType.JINJA_STMT):
                jinja_config_raw.append(token.value)
                # Parser la config dbt
                config = self._parse_dbt_config(token.value)
                if config:
                    parsed_config.update(config)
            self.has_jinja = True
        
        # Gestion des CTE (WITH clause) - uniquement pour SELECT
        ctes = None
        if self._check(TokenType.WITH):
            ctes = self._parse_cte_list()
        
        # Déterminer le type de statement
        current = self._current()
        
        if self._check(TokenType.SELECT):
            statement = self._parse_select()
            statement.ctes = ctes
            # Stocker les configurations Jinja comme métadonnées
            if jinja_config_raw or parsed_config:
                statement.metadata = statement.metadata or {}
                if jinja_config_raw:
                    statement.metadata["jinja_config_raw"] = jinja_config_raw
                if parsed_config:
                    statement.metadata["dbt_config"] = parsed_config
            return statement
        
        elif self._check(TokenType.INSERT):
            return self._parse_insert()
        
        elif self._check(TokenType.UPDATE):
            return self._parse_update()
        
        elif self._check(TokenType.DELETE):
            return self._parse_delete()
        
        elif self._check(TokenType.MERGE):
            return self._parse_merge()
        
        elif self._check(TokenType.CREATE):
            return self._parse_create()
        
        elif self._check(TokenType.DROP):
            return self._parse_drop()
        
        elif self._check(TokenType.ALTER):
            return self._parse_alter()
        
        elif self._check(TokenType.TRUNCATE):
            return self._parse_truncate()
        
        else:
            raise SQLParserError(f"Unexpected statement type: {current.type.name}", current)
    
    def _parse_dbt_config(self, jinja_content: str) -> dict:
        """
        Parse le contenu d'un template Jinja dbt pour extraire la configuration.
        
        Supporte:
        - {{ config(...) }}
        - {{ ref('table') }}
        - {{ source('schema', 'table') }}
        - {{ var('name') }}
        
        Returns:
            dict avec les paramètres extraits ou None
        """
        import re
        
        result = {}
        
        # Nettoyer le contenu (retirer {{ }} ou {% %})
        content = jinja_content.strip()
        if content.startswith('{{') and content.endswith('}}'):
            content = content[2:-2].strip()
        elif content.startswith('{%') and content.endswith('%}'):
            content = content[2:-2].strip()
        
        # Parser config(...)
        config_match = re.match(r'config\s*\((.*)\)\s*$', content, re.DOTALL)
        if config_match:
            config_content = config_match.group(1).strip()
            result = self._parse_python_kwargs(config_content)
            return result
        
        # Parser ref('table') ou ref('schema', 'table')
        ref_match = re.match(r"ref\s*\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*['\"]([^'\"]+)['\"])?\s*\)", content)
        if ref_match:
            if ref_match.group(2):
                return {"_type": "ref", "schema": ref_match.group(1), "table": ref_match.group(2)}
            return {"_type": "ref", "table": ref_match.group(1)}
        
        # Parser source('schema', 'table')
        source_match = re.match(r"source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)", content)
        if source_match:
            return {"_type": "source", "schema": source_match.group(1), "table": source_match.group(2)}
        
        # Parser var('name')
        var_match = re.match(r"var\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", content)
        if var_match:
            return {"_type": "var", "name": var_match.group(1)}
        
        return None
    
    def _parse_python_kwargs(self, content: str) -> dict:
        """
        Parse des kwargs Python simples: key='value', key=True, key=['a', 'b']
        
        Returns:
            dict avec les paramètres parsés
        """
        import re
        
        result = {}
        
        # Regex pour capturer key=value patterns
        # Supporte: string, bool, int, list simple
        patterns = [
            # key='string' ou key="string"
            (r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", lambda m: (m.group(1), m.group(2))),
            # key=True ou key=False
            (r"(\w+)\s*=\s*(True|False)", lambda m: (m.group(1), m.group(2) == 'True')),
            # key=123
            (r"(\w+)\s*=\s*(\d+)", lambda m: (m.group(1), int(m.group(2)))),
            # key=None
            (r"(\w+)\s*=\s*None", lambda m: (m.group(1), None)),
        ]
        
        for pattern, extractor in patterns:
            for match in re.finditer(pattern, content):
                key, value = extractor(match)
                result[key] = value
        
        # Parser les listes: key=['a', 'b', 'c']
        list_pattern = r"(\w+)\s*=\s*\[([^\]]*)\]"
        for match in re.finditer(list_pattern, content):
            key = match.group(1)
            list_content = match.group(2)
            # Extraire les éléments de la liste
            items = re.findall(r"['\"]([^'\"]*)['\"]", list_content)
            if items:
                result[key] = items
        
        # Parser les dicts simples: key={'a': 'b'}
        dict_pattern = r"(\w+)\s*=\s*\{([^\}]*)\}"
        for match in re.finditer(dict_pattern, content):
            key = match.group(1)
            dict_content = match.group(2)
            # Extraire les paires clé-valeur
            pairs = re.findall(r"['\"]([^'\"]*)['\"]:\s*['\"]([^'\"]*)['\"]", dict_content)
            if pairs:
                result[key] = dict(pairs)
        
        return result
    
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
        
        # DISTINCT / DISTINCT ON (PostgreSQL)
        distinct = False
        distinct_on = None
        
        if self._consume_if(TokenType.DISTINCT):
            distinct = True
            # Check for DISTINCT ON (col1, col2, ...) - PostgreSQL specific
            if self._check(TokenType.ON):
                self._advance()  # consume ON
                self._expect(TokenType.LPAREN)
                distinct_on = []
                while True:
                    distinct_on.append(self._parse_expression())
                    if not self._consume_if(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
        elif not self._consume_if(TokenType.ALL):
            pass  # ALL est le comportement par défaut
        
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
        
        # WINDOW clause (WINDOW w AS (...))
        window_clause = None
        if self._check(TokenType.WINDOW):
            window_clause = self._parse_window_clause()
        
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
            distinct=distinct,
            distinct_on=distinct_on,
            window_clause=window_clause
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
        # Cas spécial: * ou * EXCEPT(...) ou * REPLACE(...)
        if self._check(TokenType.STAR):
            self._advance()
            star = self._parse_star_modifiers(Star())
            return SelectItem(expression=star)
        
        # Cas spécial: table.* ou table.* EXCEPT(...)
        if self._check(TokenType.IDENTIFIER) and self._peek().type == TokenType.DOT:
            if self._peek(2).type == TokenType.STAR:
                table_token = self._advance()
                self._advance()  # .
                self._advance()  # *
                star = self._parse_star_modifiers(Star(table=table_token.value))
                return SelectItem(expression=star)
        
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
    
    def _parse_star_modifiers(self, star: Star) -> Star:
        """Parse les modificateurs EXCEPT/REPLACE après *."""
        # EXCEPT(col1, col2)
        if self._consume_if(TokenType.EXCEPT):
            self._expect(TokenType.LPAREN)
            except_cols = []
            while True:
                col_token = self._expect(TokenType.IDENTIFIER, "Expected column name")
                except_cols.append(col_token.value)
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
            star.except_columns = except_cols
        
        # REPLACE(expr AS alias, ...)
        if self._consume_if(TokenType.REPLACE):
            self._expect(TokenType.LPAREN)
            replace_cols = []
            while True:
                expr = self._parse_expression()
                self._expect(TokenType.AS, "Expected AS in REPLACE")
                alias_token = self._expect(TokenType.IDENTIFIER, "Expected alias in REPLACE")
                replace_cols.append((expr, alias_token.value))
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
            star.replace_columns = replace_cols
        
        return star
    
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
        # Accepte IDENTIFIER ou QUOTED_IDENTIFIER
        if not self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
            raise SQLParserError("Expected table name", self._current())
        
        name_token = self._advance()
        schema = None
        schema_quoted = False
        name = name_token.value
        quoted = name_token.type == TokenType.QUOTED_IDENTIFIER
        
        # Retirer les guillemets si c'est un identifiant quoté
        if quoted and len(name) >= 2:
            name = name[1:-1]
        
        if self._consume_if(TokenType.DOT):
            # Ce qu'on avait comme "name" est en fait le schema
            schema = name
            schema_quoted = quoted
            if not self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
                raise SQLParserError("Expected table name after schema", self._current())
            name_token = self._advance()
            name = name_token.value
            quoted = name_token.type == TokenType.QUOTED_IDENTIFIER
            if quoted and len(name) >= 2:
                name = name[1:-1]
        
        self.tables_referenced.add(name)
        
        # Alias
        alias = None
        if self._consume_if(TokenType.AS):
            if self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
                alias = self._advance().value
            else:
                raise SQLParserError("Expected alias", self._current())
        elif self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER) and not self._is_keyword(self._current()):
            alias = self._advance().value
        
        return TableRef(name=name, alias=alias, schema=schema, quoted=quoted, schema_quoted=schema_quoted)
    
    def _parse_table_name_only(self) -> TableRef:
        """Parse un nom de table sans alias (pour CREATE TABLE, DROP TABLE, etc.)."""
        if not self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
            raise SQLParserError("Expected table name", self._current())
        
        name_token = self._advance()
        schema = None
        schema_quoted = False
        name = name_token.value
        quoted = name_token.type == TokenType.QUOTED_IDENTIFIER
        
        # Retirer les guillemets si c'est un identifiant quoté
        if quoted and len(name) >= 2:
            name = name[1:-1]
        
        # Gestion du schéma: schema.table
        if self._consume_if(TokenType.DOT):
            schema = name
            schema_quoted = quoted
            if not self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
                raise SQLParserError("Expected table name after schema", self._current())
            name_token = self._advance()
            name = name_token.value
            quoted = name_token.type == TokenType.QUOTED_IDENTIFIER
            if quoted and len(name) >= 2:
                name = name[1:-1]
        
        return TableRef(name=name, alias=None, schema=schema, quoted=quoted, schema_quoted=schema_quoted)
    
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
    
    def _parse_simple_table_ref(self) -> TableRef:
        """Parse un nom de table simple (schema.table) sans alias."""
        if not self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
            raise SQLParserError("Expected table name", self._current())
        
        name_token = self._advance()
        schema = None
        name = name_token.value
        quoted = name_token.type == TokenType.QUOTED_IDENTIFIER
        
        if quoted and len(name) >= 2:
            name = name[1:-1]
        
        if self._consume_if(TokenType.DOT):
            schema = name
            if not self._match(TokenType.IDENTIFIER, TokenType.QUOTED_IDENTIFIER):
                raise SQLParserError("Expected table name after schema", self._current())
            name_token = self._advance()
            name = name_token.value
            if name_token.type == TokenType.QUOTED_IDENTIFIER and len(name) >= 2:
                name = name[1:-1]
        
        return TableRef(name=name, schema=schema, quoted=quoted)
    
    def _is_keyword(self, token: Token) -> bool:
        """Vérifie si un token IDENTIFIER est en fait un mot-clé contextuel."""
        keyword_types = {
            TokenType.JOIN, TokenType.INNER, TokenType.LEFT, TokenType.RIGHT,
            TokenType.FULL, TokenType.OUTER, TokenType.CROSS, TokenType.NATURAL,
            TokenType.ON, TokenType.WHERE, TokenType.GROUP, TokenType.ORDER,
            TokenType.HAVING, TokenType.LIMIT, TokenType.OFFSET, TokenType.UNION,
            TokenType.INTERSECT, TokenType.EXCEPT, TokenType.LATERAL, TokenType.UNNEST,
            TokenType.SET, TokenType.VALUES, TokenType.DROP, TokenType.WHEN
        }
        if token.type in keyword_types:
            return True
        # Mots-clés reconnus comme IDENTIFIER mais qui ne devraient pas être des alias
        keyword_values = {'ADD', 'MODIFY', 'RENAME', 'ALTER', 'USING', 'MATCHED', 'THEN'}
        if token.type == TokenType.IDENTIFIER and token.value.upper() in keyword_values:
            return True
        return False
    
    def _try_parse_join(self) -> Optional[JoinClause]:
        """Tente de parser une clause JOIN."""
        join_type = None
        
        # Détermine le type de JOIN
        if self._consume_if(TokenType.NATURAL):
            # NATURAL [INNER|LEFT|RIGHT|FULL] JOIN
            if self._consume_if(TokenType.LEFT):
                self._consume_if(TokenType.OUTER)
                join_type = JoinType.LEFT
            elif self._consume_if(TokenType.RIGHT):
                self._consume_if(TokenType.OUTER)
                join_type = JoinType.RIGHT
            elif self._consume_if(TokenType.FULL):
                self._consume_if(TokenType.OUTER)
                join_type = JoinType.FULL
            else:
                self._consume_if(TokenType.INNER)
                join_type = JoinType.NATURAL
            self._expect(TokenType.JOIN)
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
    
    def _parse_group_by(self) -> List:
        """Parse la clause GROUP BY avec support GROUPING SETS, CUBE, ROLLUP."""
        self._expect(TokenType.GROUP)
        self._expect(TokenType.BY)
        
        elements = []
        while True:
            element = self._parse_grouping_element()
            elements.append(element)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return elements
    
    def _parse_grouping_element(self):
        """Parse un élément de GROUP BY (expression, GROUPING SETS, CUBE, ROLLUP)."""
        # GROUPING SETS
        if self._consume_if(TokenType.GROUPING):
            self._expect(TokenType.SETS)
            self._expect(TokenType.LPAREN)
            sets = []
            while True:
                if self._consume_if(TokenType.LPAREN):
                    # (a, b)
                    exprs = []
                    if not self._check(TokenType.RPAREN):
                        while True:
                            exprs.append(self._parse_expression())
                            if not self._consume_if(TokenType.COMMA):
                                break
                    self._expect(TokenType.RPAREN)
                    sets.append(exprs)
                else:
                    # Expression simple
                    sets.append([self._parse_expression()])
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
            return GroupingSets(sets=sets)
        
        # CUBE
        if self._consume_if(TokenType.CUBE):
            self._expect(TokenType.LPAREN)
            exprs = []
            while True:
                exprs.append(self._parse_expression())
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
            return Cube(expressions=exprs)
        
        # ROLLUP
        if self._consume_if(TokenType.ROLLUP):
            self._expect(TokenType.LPAREN)
            exprs = []
            while True:
                exprs.append(self._parse_expression())
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
            return Rollup(expressions=exprs)
        
        # Expression simple
        return self._parse_expression()
    
    def _parse_order_by(self) -> List[OrderByItem]:
        """Parse la clause ORDER BY."""
        self._expect(TokenType.ORDER)
        self._expect(TokenType.BY)
        
        items = []
        while True:
            item = self._parse_order_by_item()
            items.append(item)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return items
    
    def _parse_order_by_item(self) -> OrderByItem:
        """Parse un seul élément ORDER BY."""
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
        
        return OrderByItem(
            expression=expr,
            direction=direction,
            nulls_first=nulls_first
        )
    
    def _parse_window_clause(self) -> List[WindowDefinition]:
        """Parse la clause WINDOW (WINDOW w AS (...), ...)."""
        self._expect(TokenType.WINDOW)
        
        definitions = []
        while True:
            # Nom de la fenêtre
            name_token = self._expect(TokenType.IDENTIFIER, "Expected window name")
            self._expect(TokenType.AS)
            self._expect(TokenType.LPAREN)
            
            # Contenu de la fenêtre (similaire à OVER)
            partition_by = None
            order_by = None
            frame_type = None
            frame_start = None
            frame_end = None
            
            # PARTITION BY
            if self._check(TokenType.PARTITION):
                self._advance()
                self._expect(TokenType.BY)
                partition_by = []
                while True:
                    partition_by.append(self._parse_expression())
                    if not self._consume_if(TokenType.COMMA):
                        break
            
            # ORDER BY
            if self._check(TokenType.ORDER):
                self._advance()
                self._expect(TokenType.BY)
                order_by = []
                while True:
                    order_by.append(self._parse_order_by_item())
                    if not self._consume_if(TokenType.COMMA):
                        break
            
            # Frame specification
            if self._check(TokenType.ROWS) or self._check(TokenType.RANGE) or self._check(TokenType.GROUPS):
                frame_type = self._advance().value.upper()
                if self._consume_if(TokenType.BETWEEN):
                    frame_start = self._parse_frame_bound()
                    self._expect(TokenType.AND)
                    frame_end = self._parse_frame_bound()
                else:
                    frame_start = self._parse_frame_bound()
            
            self._expect(TokenType.RPAREN)
            
            definitions.append(WindowDefinition(
                name=name_token.value,
                partition_by=partition_by,
                order_by=order_by,
                frame_type=frame_type,
                frame_start=frame_start,
                frame_end=frame_end
            ))
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        return definitions
    
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
            
            # Check for quantified comparison: = ALL/ANY/SOME (subquery)
            if self._match(TokenType.ALL, TokenType.ANY, TokenType.SOME):
                quantifier = self._advance().value.upper()
                self._expect(TokenType.LPAREN)
                self.has_subquery = True
                subquery = self._parse_select()
                self._expect(TokenType.RPAREN)
                return QuantifiedComparison(
                    expression=left,
                    operator=operator,
                    quantifier=quantifier,
                    subquery=subquery
                )
            
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
        
        # TRY_CAST (Presto/Athena) - must check before TRY
        if self._check(TokenType.TRY_CAST):
            return self._parse_try_cast_expression()
        
        # TRY() (Presto/Athena)
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
        
        # Parenthèses ou sous-requête ou lambda multi-paramètres ou tuple
        if self._consume_if(TokenType.LPAREN):
            if self._check(TokenType.SELECT):
                self.has_subquery = True
                subquery = self._parse_select()
                self._expect(TokenType.RPAREN)
                return SubqueryExpression(query=subquery)
            else:
                # Check for multi-parameter lambda: (x, y) -> expr
                # Look ahead to see if this could be a lambda parameter list
                if self._is_lambda_params():
                    return self._parse_multi_param_lambda()
                
                expr = self._parse_expression()
                
                # Check for tuple: (expr, expr, ...)
                if self._consume_if(TokenType.COMMA):
                    elements = [expr]
                    while True:
                        elements.append(self._parse_expression())
                        if not self._consume_if(TokenType.COMMA):
                            break
                    self._expect(TokenType.RPAREN)
                    return RowExpression(fields=elements)
                
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
    
    def _parse_aggregate_function(self) -> Expression:
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
        
        func = FunctionCall(name=func_name, args=args, distinct=distinct)
        
        # Check for OVER clause (window function)
        if self._check(TokenType.OVER):
            return self._parse_window_function(func)
        
        return func
    
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
            
            # EXTRACT a une syntaxe spéciale: EXTRACT(field FROM expr)
            if func_name == 'EXTRACT':
                # Le champ est un mot-clé comme YEAR, MONTH, DAY, etc.
                field_token = self._advance()
                field = field_token.value.upper()
                self._expect(TokenType.FROM, "Expected FROM in EXTRACT expression")
                expr = self._parse_expression()
                self._expect(TokenType.RPAREN)
                return ExtractExpression(field=field, expression=expr)
            
            if func_name.lower() in self.AGGREGATE_FUNCTIONS:
                self.has_aggregation = True
            
            distinct = self._consume_if(TokenType.DISTINCT)
            
            # STRUCT et ROW supportent les arguments nommés (expr AS name)
            supports_named_args = func_name in ('STRUCT', 'ROW')
            # JSON_OBJECT supporte key: value syntax
            supports_key_value = func_name in ('JSON_OBJECT',)
            
            args = []
            if not self._check(TokenType.RPAREN):
                if self._check(TokenType.STAR):
                    self._advance()
                    args.append(Star())
                else:
                    while True:
                        arg = self._parse_expression()
                        # Check for key:value syntax (JSON_OBJECT)
                        if supports_key_value and self._consume_if(TokenType.COLON):
                            value = self._parse_expression()
                            arg = KeyValuePair(key=arg, value=value)
                        # Check for named argument (expr AS name)
                        elif supports_named_args and self._consume_if(TokenType.AS):
                            arg_name_token = self._expect(TokenType.IDENTIFIER, "Expected field name after AS")
                            arg = NamedArgument(expression=arg, name=arg_name_token.value)
                        args.append(arg)
                        if not self._consume_if(TokenType.COMMA):
                            break
            
            self._expect(TokenType.RPAREN)
            func = FunctionCall(name=func_name, args=args, distinct=distinct)
            
            # Check for OVER clause (window function)
            if self._check(TokenType.OVER):
                return self._parse_window_function(func)
            
            return func
        
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
    
    def _parse_window_function(self, func: FunctionCall) -> WindowFunction:
        """Parse une window function avec OVER clause."""
        self._expect(TokenType.OVER)
        
        # Check for window name reference (OVER window_name) vs inline spec (OVER (...))
        if not self._check(TokenType.LPAREN):
            # Window name reference
            window_name = self._expect(TokenType.IDENTIFIER).value
            return WindowFunction(
                function=func,
                partition_by=None,
                order_by=None,
                frame_type=None,
                frame_start=None,
                frame_end=None,
                window_name=window_name
            )
        
        self._expect(TokenType.LPAREN)
        
        partition_by = None
        order_by = None
        frame_type = None
        frame_start = None
        frame_end = None
        
        # PARTITION BY
        if self._check(TokenType.PARTITION):
            self._advance()
            self._expect(TokenType.BY)
            partition_by = []
            while True:
                expr = self._parse_expression()
                partition_by.append(expr)
                if not self._consume_if(TokenType.COMMA):
                    break
        
        # ORDER BY
        if self._check(TokenType.ORDER):
            self._advance()
            self._expect(TokenType.BY)
            order_by = []
            while True:
                order_item = self._parse_order_by_item()
                order_by.append(order_item)
                if not self._consume_if(TokenType.COMMA):
                    break
        
        # Frame specification: ROWS/RANGE/GROUPS BETWEEN ... AND ...
        if self._check(TokenType.ROWS) or self._check(TokenType.RANGE) or self._check(TokenType.GROUPS):
            frame_type = self._advance().value.upper()
            
            if self._consume_if(TokenType.BETWEEN):
                frame_start = self._parse_frame_bound()
                self._expect(TokenType.AND)
                frame_end = self._parse_frame_bound()
            else:
                # Just frame start (e.g., ROWS UNBOUNDED PRECEDING)
                frame_start = self._parse_frame_bound()
            
            # EXCLUDE clause (optional)
            if self._consume_if(TokenType.EXCLUDE):
                if self._check_keyword('CURRENT') and self._peek().value.upper() == 'ROW':
                    self._advance()  # CURRENT
                    self._advance()  # ROW
                    # Store in metadata or extend WindowFunction
                elif self._check_keyword('NO') and self._peek().value.upper() == 'OTHERS':
                    self._advance()
                    self._advance()
        
        self._expect(TokenType.RPAREN)
        
        return WindowFunction(
            function=func,
            partition_by=partition_by,
            order_by=order_by,
            frame_type=frame_type,
            frame_start=frame_start,
            frame_end=frame_end
        )
    
    def _parse_frame_bound(self) -> str:
        """Parse a window frame bound (UNBOUNDED PRECEDING, CURRENT ROW, etc.)."""
        if self._check(TokenType.UNBOUNDED):
            self._advance()
            if self._check(TokenType.PRECEDING):
                self._advance()
                return "UNBOUNDED PRECEDING"
            elif self._check(TokenType.FOLLOWING):
                self._advance()
                return "UNBOUNDED FOLLOWING"
        elif self._check(TokenType.CURRENT):
            self._advance()
            self._expect(TokenType.ROW)
            return "CURRENT ROW"
        else:
            # N PRECEDING or N FOLLOWING
            expr = self._parse_expression()
            if self._check(TokenType.PRECEDING):
                self._advance()
                return f"{expr} PRECEDING"
            elif self._check(TokenType.FOLLOWING):
                self._advance()
                return f"{expr} FOLLOWING"
        return "CURRENT ROW"
    
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
    
    def _parse_try_cast_expression(self) -> CastExpression:
        """Parse TRY_CAST(expr AS type) (Presto/Athena)."""
        self._expect(TokenType.TRY_CAST)
        self._expect(TokenType.LPAREN)
        
        self.functions_used.add("TRY_CAST")
        
        expr = self._parse_expression()
        self._expect(TokenType.AS, "Expected AS in TRY_CAST expression")
        target_type = self._parse_type_name()
        self._expect(TokenType.RPAREN)
        
        return CastExpression(expression=expr, target_type=target_type, is_try_cast=True)
    
    def _parse_try_expression(self) -> Expression:
        """Parse TRY(...) (Presto/Athena)."""
        self._expect(TokenType.TRY)
        
        self._expect(TokenType.LPAREN)
        expr = self._parse_expression()
        self._expect(TokenType.RPAREN)
        
        self.functions_used.add("TRY")
        
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
    
    def _is_lambda_params(self) -> bool:
        """Check if current position looks like lambda parameters: x, y) ->"""
        # Save position for lookahead
        save_pos = self.pos
        
        try:
            # Must start with identifier
            if not self._check(TokenType.IDENTIFIER):
                return False
            
            # Scan forward to find ) followed by ->
            depth = 1
            while self.pos < len(self.tokens) - 1:
                token = self._current()
                
                if token.type == TokenType.LPAREN:
                    depth += 1
                elif token.type == TokenType.RPAREN:
                    depth -= 1
                    if depth == 0:
                        # Found matching ), check for ->
                        self._advance()
                        is_lambda = self._check(TokenType.ARROW)
                        return is_lambda
                elif token.type not in (TokenType.IDENTIFIER, TokenType.COMMA):
                    # Lambda params should only have identifiers and commas
                    return False
                
                self._advance()
            
            return False
        finally:
            self.pos = save_pos
    
    def _parse_multi_param_lambda(self) -> LambdaExpression:
        """Parse a multi-parameter lambda: (x, y) -> x + y"""
        parameters = []
        
        # Parse parameter list
        while True:
            if not self._check(TokenType.IDENTIFIER):
                raise SQLParserError("Expected parameter name in lambda", self._current())
            
            param = self._advance().value
            parameters.append(param)
            
            if not self._consume_if(TokenType.COMMA):
                break
        
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.ARROW)
        
        body = self._parse_expression()
        
        return LambdaExpression(parameters=parameters, body=body)
    
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

    # ============== INSERT Statement ==============
    
    def _parse_insert(self) -> InsertStatement:
        """Parse un statement INSERT INTO."""
        self._expect(TokenType.INSERT)
        self._expect(TokenType.INTO)
        
        # Table cible
        table = self._parse_table_ref()
        
        # Colonnes optionnelles
        columns = None
        if self._check(TokenType.LPAREN):
            self._advance()
            columns = []
            while True:
                col_token = self._expect(TokenType.IDENTIFIER, "Expected column name")
                columns.append(col_token.value)
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
        
        # VALUES ou SELECT
        values = None
        query = None
        
        if self._check(TokenType.VALUES):
            self._advance()
            values = []
            while True:
                self._expect(TokenType.LPAREN)
                row = []
                while True:
                    if self._check(TokenType.DEFAULT):
                        self._advance()
                        row.append(Literal(value="DEFAULT", literal_type="keyword"))
                    else:
                        expr = self._parse_expression()
                        row.append(expr)
                    if not self._consume_if(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
                values.append(row)
                if not self._consume_if(TokenType.COMMA):
                    break
        elif self._check(TokenType.SELECT):
            query = self._parse_select()
        else:
            raise SQLParserError("Expected VALUES or SELECT after INSERT INTO", self._current())
        
        # ON CONFLICT (PostgreSQL UPSERT)
        on_conflict = None
        if self._check(TokenType.ON) and self._peek().type == TokenType.CONFLICT:
            on_conflict = self._parse_on_conflict()
        
        return InsertStatement(
            table=table,
            columns=columns,
            values=values,
            query=query,
            on_conflict=on_conflict
        )
    
    def _parse_on_conflict(self) -> OnConflictClause:
        """Parse ON CONFLICT clause pour PostgreSQL."""
        self._expect(TokenType.ON)
        self._expect(TokenType.CONFLICT)
        
        # Conflict target (colonnes)
        conflict_target = None
        if self._check(TokenType.LPAREN):
            self._advance()
            conflict_target = []
            while True:
                col = self._expect(TokenType.IDENTIFIER, "Expected column name")
                conflict_target.append(col.value)
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
        
        self._expect(TokenType.DO)
        
        action = "NOTHING"
        update_assignments = None
        
        if self._check(TokenType.NOTHING):
            self._advance()
        elif self._check(TokenType.UPDATE):
            self._advance()
            action = "UPDATE"
            self._expect(TokenType.SET)
            update_assignments = self._parse_assignments()
        
        return OnConflictClause(
            conflict_target=conflict_target,
            action=action,
            update_assignments=update_assignments
        )

    # ============== UPDATE Statement ==============
    
    def _parse_update(self) -> UpdateStatement:
        """Parse un statement UPDATE."""
        self._expect(TokenType.UPDATE)
        
        # Table cible
        table = self._parse_table_ref()
        
        # SET clause
        self._expect(TokenType.SET)
        assignments = self._parse_assignments()
        
        # FROM clause optionnelle (PostgreSQL)
        from_clause = None
        if self._check(TokenType.FROM):
            from_clause = self._parse_from_clause()
        
        # WHERE clause optionnelle
        where_clause = None
        if self._check(TokenType.WHERE):
            self._advance()
            where_clause = self._parse_expression()
        
        return UpdateStatement(
            table=table,
            assignments=assignments,
            from_clause=from_clause,
            where_clause=where_clause
        )
    
    def _parse_assignments(self) -> List[Assignment]:
        """Parse une liste d'assignations col = valeur ou table.col = valeur."""
        assignments = []
        while True:
            # Colonne (peut être préfixée par table: t.column)
            col_name = self._expect(TokenType.IDENTIFIER, "Expected column name").value
            if self._consume_if(TokenType.DOT):
                # Ignorer le préfixe de table et prendre juste le nom de colonne
                col_name = self._expect(TokenType.IDENTIFIER, "Expected column name after dot").value
            
            self._expect(TokenType.EQUALS)
            value = self._parse_expression()
            assignments.append(Assignment(column=col_name, value=value))
            if not self._consume_if(TokenType.COMMA):
                break
        return assignments

    # ============== DELETE Statement ==============
    
    def _parse_delete(self) -> DeleteStatement:
        """Parse un statement DELETE."""
        self._expect(TokenType.DELETE)
        self._expect(TokenType.FROM)
        
        # Table cible
        table = self._parse_table_ref()
        
        # USING clause optionnelle (PostgreSQL)
        using = None
        if self._check(TokenType.USING):
            self._advance()
            using = []
            while True:
                t = self._parse_table_ref()
                using.append(t)
                if not self._consume_if(TokenType.COMMA):
                    break
        
        # WHERE clause optionnelle
        where_clause = None
        if self._check(TokenType.WHERE):
            self._advance()
            where_clause = self._parse_expression()
        
        return DeleteStatement(
            table=table,
            using=using,
            where_clause=where_clause
        )

    # ============== MERGE Statement ==============
    
    def _parse_merge(self) -> MergeStatement:
        """Parse un statement MERGE INTO."""
        self._expect(TokenType.MERGE)
        self._consume_if(TokenType.INTO)
        
        # Table cible
        target = self._parse_table_ref()
        
        # USING clause
        self._expect(TokenType.USING)
        if self._check(TokenType.LPAREN):
            # Sous-requête
            self._advance()
            subquery = self._parse_select()
            self._expect(TokenType.RPAREN)
            alias = None
            self._consume_if(TokenType.AS)
            if self._check(TokenType.IDENTIFIER):
                alias = self._advance().value
            source = SubqueryRef(query=subquery, alias=alias or "_source")
        else:
            source = self._parse_table_ref()
        
        # ON condition
        self._expect(TokenType.ON)
        on_condition = self._parse_expression()
        
        # WHEN clauses
        when_clauses = []
        while self._check(TokenType.WHEN):
            when_clause = self._parse_merge_when_clause()
            when_clauses.append(when_clause)
        
        return MergeStatement(
            target=target,
            source=source,
            on_condition=on_condition,
            when_clauses=when_clauses
        )
    
    def _parse_merge_when_clause(self) -> MergeWhenClause:
        """Parse une clause WHEN pour MERGE."""
        self._expect(TokenType.WHEN)
        
        # MATCHED ou NOT MATCHED
        matched = True
        if self._check(TokenType.NOT):
            self._advance()
            matched = False
        self._expect(TokenType.MATCHED)
        
        # BY TARGET ou BY SOURCE optionnel (SQL Server)
        if self._check(TokenType.BY):
            self._advance()
            self._advance()  # TARGET ou SOURCE
        
        # Condition AND optionnelle
        condition = None
        if self._check(TokenType.AND):
            self._advance()
            condition = self._parse_expression()
        
        self._expect(TokenType.THEN)
        
        # Action: UPDATE, DELETE, INSERT
        action = "UPDATE"
        assignments = None
        insert_columns = None
        insert_values = None
        
        if self._check(TokenType.UPDATE):
            self._advance()
            self._expect(TokenType.SET)
            assignments = self._parse_assignments()
        elif self._check(TokenType.DELETE):
            self._advance()
            action = "DELETE"
        elif self._check(TokenType.INSERT):
            self._advance()
            action = "INSERT"
            
            # Colonnes optionnelles
            if self._check(TokenType.LPAREN):
                self._advance()
                insert_columns = []
                while True:
                    col = self._expect(TokenType.IDENTIFIER, "Expected column name")
                    insert_columns.append(col.value)
                    if not self._consume_if(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
            
            # VALUES
            self._expect(TokenType.VALUES)
            self._expect(TokenType.LPAREN)
            insert_values = []
            while True:
                if self._check(TokenType.DEFAULT):
                    self._advance()
                    insert_values.append(Literal(value="DEFAULT", literal_type="keyword"))
                else:
                    expr = self._parse_expression()
                    insert_values.append(expr)
                if not self._consume_if(TokenType.COMMA):
                    break
            self._expect(TokenType.RPAREN)
        
        return MergeWhenClause(
            matched=matched,
            condition=condition,
            action=action,
            assignments=assignments,
            insert_columns=insert_columns,
            insert_values=insert_values
        )

    # ============== CREATE Statement ==============
    
    def _parse_create(self) -> Union[CreateTableStatement, CreateViewStatement]:
        """Parse un statement CREATE TABLE ou CREATE VIEW."""
        self._expect(TokenType.CREATE)
        
        or_replace = False
        if self._check(TokenType.OR):
            self._advance()
            if self._check(TokenType.REPLACE):
                self._advance()
                or_replace = True
        
        temporary = False
        if self._match(TokenType.TEMPORARY, TokenType.TEMP):
            self._advance()
            temporary = True
        
        external = False
        if self._check(TokenType.EXTERNAL):
            self._advance()
            external = True
        
        if self._check(TokenType.TABLE):
            return self._parse_create_table(temporary, external)
        elif self._check(TokenType.VIEW):
            return self._parse_create_view(or_replace)
        elif self._check(TokenType.INDEX):
            # Pour l'instant, on renvoie une erreur
            raise SQLParserError("CREATE INDEX not yet supported", self._current())
        else:
            raise SQLParserError("Expected TABLE or VIEW after CREATE", self._current())
    
    def _parse_create_table(self, temporary: bool = False, external: bool = False) -> CreateTableStatement:
        """Parse CREATE TABLE."""
        self._expect(TokenType.TABLE)
        
        if_not_exists = False
        if self._check(TokenType.IF):
            self._advance()
            self._expect(TokenType.NOT)
            self._expect(TokenType.EXISTS)
            if_not_exists = True
        
        # Nom de la table (sans alias - utiliser méthode spécifique)
        table = self._parse_table_name_only()
        
        columns = []
        constraints = []
        as_query = None
        location = None
        stored_as = None
        row_format = None
        table_properties = None
        
        # CREATE TABLE AS SELECT
        if self._check(TokenType.AS):
            self._advance()
            as_query = self._parse_select()
        elif self._check(TokenType.LPAREN):
            self._advance()
            
            # Colonnes et contraintes
            while True:
                if self._match(TokenType.PRIMARY, TokenType.FOREIGN, TokenType.UNIQUE, 
                              TokenType.CHECK, TokenType.CONSTRAINT):
                    constraint = self._parse_table_constraint()
                    constraints.append(constraint)
                else:
                    column = self._parse_column_definition()
                    columns.append(column)
                
                if not self._consume_if(TokenType.COMMA):
                    break
            
            self._expect(TokenType.RPAREN)
        
        # Options Athena/Hive
        while not self._check(TokenType.EOF) and not self._check(TokenType.SEMICOLON):
            if self._current().value.upper() == 'LOCATION':
                self._advance()
                location = self._expect(TokenType.STRING, "Expected location string").value
            elif self._current().value.upper() == 'STORED':
                self._advance()
                if self._check(TokenType.AS):
                    self._advance()
                stored_as = self._advance().value
            elif self._current().value.upper() == 'ROW':
                self._advance()
                if self._current().value.upper() == 'FORMAT':
                    self._advance()
                row_format = self._advance().value
            elif self._current().value.upper() == 'TBLPROPERTIES':
                self._advance()
                table_properties = self._parse_table_properties()
            else:
                break
        
        return CreateTableStatement(
            table=table,
            columns=columns,
            constraints=constraints if constraints else None,
            if_not_exists=if_not_exists,
            temporary=temporary,
            as_query=as_query,
            external=external,
            location=location,
            stored_as=stored_as,
            row_format=row_format,
            table_properties=table_properties
        )
    
    def _parse_column_definition(self) -> ColumnDefinition:
        """Parse une définition de colonne."""
        # Nom de colonne - accepte aussi certains mots-clés courants comme noms de colonnes
        # KEY, VALUE, DATA, TYPE, NAME, etc. sont souvent utilisés comme noms de colonnes
        if self._check(TokenType.IDENTIFIER):
            name_token = self._advance()
            name = name_token.value
        elif self._current().type in (TokenType.KEY, TokenType.DATE,
                                       TokenType.TIME, TokenType.TIMESTAMP, TokenType.COMMENT):
            # Accepter ces mots-clés comme noms de colonnes
            name_token = self._advance()
            name = name_token.value
        elif self._check(TokenType.QUOTED_IDENTIFIER):
            name_token = self._advance()
            name = name_token.value[1:-1] if len(name_token.value) >= 2 else name_token.value
        else:
            raise SQLParserError("Expected column name", self._current())
        
        # Type de données
        data_type = self._parse_type_name()
        
        # Modificateurs
        nullable = True
        default = None
        primary_key = False
        unique = False
        references = None
        
        while True:
            if self._check(TokenType.NOT):
                self._advance()
                self._expect(TokenType.NULL)
                nullable = False
            elif self._check(TokenType.NULL):
                self._advance()
                nullable = True
            elif self._check(TokenType.DEFAULT):
                self._advance()
                default = self._parse_expression()
            elif self._check(TokenType.PRIMARY):
                self._advance()
                self._expect(TokenType.KEY)
                primary_key = True
            elif self._check(TokenType.UNIQUE):
                self._advance()
                unique = True
            elif self._check(TokenType.REFERENCES):
                self._advance()
                ref_table = self._advance().value
                if self._check(TokenType.LPAREN):
                    self._advance()
                    ref_col = self._advance().value
                    self._expect(TokenType.RPAREN)
                    references = f"{ref_table}({ref_col})"
                else:
                    references = ref_table
            else:
                break
        
        return ColumnDefinition(
            name=name,
            data_type=data_type,
            nullable=nullable,
            default=default,
            primary_key=primary_key,
            unique=unique,
            references=references
        )
    
    def _parse_table_constraint(self) -> TableConstraint:
        """Parse une contrainte de table."""
        name = None
        if self._check(TokenType.CONSTRAINT):
            self._advance()
            name = self._expect(TokenType.IDENTIFIER, "Expected constraint name").value
        
        if self._check(TokenType.PRIMARY):
            self._advance()
            self._expect(TokenType.KEY)
            columns = self._parse_column_list()
            return TableConstraint(
                constraint_type="PRIMARY KEY",
                name=name,
                columns=columns
            )
        elif self._check(TokenType.UNIQUE):
            self._advance()
            columns = self._parse_column_list()
            return TableConstraint(
                constraint_type="UNIQUE",
                name=name,
                columns=columns
            )
        elif self._check(TokenType.FOREIGN):
            self._advance()
            self._expect(TokenType.KEY)
            columns = self._parse_column_list()
            self._expect(TokenType.REFERENCES)
            ref_table = self._advance().value
            ref_columns = self._parse_column_list()
            return TableConstraint(
                constraint_type="FOREIGN KEY",
                name=name,
                columns=columns,
                references_table=ref_table,
                references_columns=ref_columns
            )
        elif self._check(TokenType.CHECK):
            self._advance()
            self._expect(TokenType.LPAREN)
            check_expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return TableConstraint(
                constraint_type="CHECK",
                name=name,
                check_expression=check_expr
            )
        else:
            raise SQLParserError("Expected constraint type", self._current())
    
    def _parse_column_list(self) -> List[str]:
        """Parse une liste de colonnes entre parenthèses."""
        self._expect(TokenType.LPAREN)
        columns = []
        while True:
            col = self._expect(TokenType.IDENTIFIER, "Expected column name")
            columns.append(col.value)
            if not self._consume_if(TokenType.COMMA):
                break
        self._expect(TokenType.RPAREN)
        return columns
    
    def _parse_table_properties(self) -> dict:
        """Parse TBLPROPERTIES pour Athena/Hive."""
        self._expect(TokenType.LPAREN)
        props = {}
        while True:
            key = self._expect(TokenType.STRING, "Expected property key").value
            self._expect(TokenType.EQUALS)
            value = self._expect(TokenType.STRING, "Expected property value").value
            props[key.strip("'")] = value.strip("'")
            if not self._consume_if(TokenType.COMMA):
                break
        self._expect(TokenType.RPAREN)
        return props
    
    def _parse_create_view(self, or_replace: bool = False) -> CreateViewStatement:
        """Parse CREATE VIEW."""
        self._expect(TokenType.VIEW)
        
        if_not_exists = False
        if self._check(TokenType.IF):
            self._advance()
            self._expect(TokenType.NOT)
            self._expect(TokenType.EXISTS)
            if_not_exists = True
        
        # Nom de la vue (sans alias)
        name = self._parse_simple_table_ref()
        
        # Colonnes optionnelles
        columns = None
        if self._check(TokenType.LPAREN):
            columns = self._parse_column_list()
        
        self._expect(TokenType.AS)
        query = self._parse_select()
        
        return CreateViewStatement(
            name=name,
            query=query,
            columns=columns,
            or_replace=or_replace,
            if_not_exists=if_not_exists
        )

    # ============== DROP Statement ==============
    
    def _parse_drop(self) -> DropStatement:
        """Parse un statement DROP."""
        self._expect(TokenType.DROP)
        
        # Type d'objet
        object_type = None
        if self._check(TokenType.TABLE):
            self._advance()
            object_type = "TABLE"
        elif self._check(TokenType.VIEW):
            self._advance()
            object_type = "VIEW"
        elif self._check(TokenType.INDEX):
            self._advance()
            object_type = "INDEX"
        elif self._check(TokenType.SCHEMA):
            self._advance()
            object_type = "SCHEMA"
        elif self._check(TokenType.DATABASE):
            self._advance()
            object_type = "DATABASE"
        else:
            raise SQLParserError("Expected TABLE, VIEW, INDEX, SCHEMA or DATABASE after DROP", self._current())
        
        if_exists = False
        if self._check(TokenType.IF):
            self._advance()
            self._expect(TokenType.EXISTS)
            if_exists = True
        
        # Nom de l'objet
        name = self._parse_table_ref()
        
        cascade = False
        if self._check(TokenType.CASCADE):
            self._advance()
            cascade = True
        elif self._check(TokenType.RESTRICT):
            self._advance()
        
        return DropStatement(
            object_type=object_type,
            name=name,
            if_exists=if_exists,
            cascade=cascade
        )

    # ============== ALTER Statement ==============
    
    def _parse_alter(self) -> AlterTableStatement:
        """Parse un statement ALTER TABLE."""
        self._expect(TokenType.ALTER)
        self._expect(TokenType.TABLE)
        
        table = self._parse_table_ref()
        
        actions = []
        while True:
            action = self._parse_alter_action()
            actions.append(action)
            if not self._consume_if(TokenType.COMMA):
                break
        
        return AlterTableStatement(table=table, actions=actions)
    
    def _parse_alter_action(self) -> AlterTableAction:
        """Parse une action ALTER TABLE."""
        current = self._current()
        
        if current.value.upper() == 'ADD':
            self._advance()
            if self._check(TokenType.COLUMN) or self._check(TokenType.IDENTIFIER):
                if self._current().value.upper() == 'COLUMN':
                    self._advance()
                column = self._parse_column_definition()
                return AlterTableAction(action_type="ADD COLUMN", column=column)
            elif self._match(TokenType.CONSTRAINT, TokenType.PRIMARY, TokenType.FOREIGN, TokenType.UNIQUE, TokenType.CHECK):
                constraint = self._parse_table_constraint()
                return AlterTableAction(action_type="ADD CONSTRAINT", constraint=constraint)
        
        elif self._check(TokenType.DROP):
            self._advance()
            if self._current().value.upper() == 'COLUMN':
                self._advance()
                col_name = self._expect(TokenType.IDENTIFIER, "Expected column name").value
                return AlterTableAction(action_type="DROP COLUMN", old_name=col_name)
            elif self._check(TokenType.CONSTRAINT):
                self._advance()
                constraint_name = self._expect(TokenType.IDENTIFIER, "Expected constraint name").value
                return AlterTableAction(action_type="DROP CONSTRAINT", old_name=constraint_name)
        
        elif current.value.upper() == 'RENAME':
            self._advance()
            if self._current().value.upper() == 'COLUMN':
                self._advance()
                old_name = self._expect(TokenType.IDENTIFIER, "Expected old column name").value
                if self._current().value.upper() == 'TO':
                    self._advance()
                new_name = self._expect(TokenType.IDENTIFIER, "Expected new column name").value
                return AlterTableAction(action_type="RENAME COLUMN", old_name=old_name, new_name=new_name)
            elif self._current().value.upper() == 'TO':
                self._advance()
                new_name = self._expect(TokenType.IDENTIFIER, "Expected new table name").value
                return AlterTableAction(action_type="RENAME TABLE", new_name=new_name)
        
        elif current.value.upper() in ('MODIFY', 'ALTER'):
            self._advance()
            if self._current().value.upper() == 'COLUMN':
                self._advance()
            column = self._parse_column_definition()
            return AlterTableAction(action_type="MODIFY COLUMN", column=column)
        
        raise SQLParserError(f"Unknown ALTER action: {current.value}", current)

    # ============== TRUNCATE Statement ==============
    
    def _parse_truncate(self) -> TruncateStatement:
        """Parse un statement TRUNCATE TABLE."""
        self._expect(TokenType.TRUNCATE)
        self._consume_if(TokenType.TABLE)
        
        table = self._parse_table_ref()
        
        return TruncateStatement(table=table)


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
