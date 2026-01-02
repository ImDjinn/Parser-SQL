"""
Tokenizer (Analyseur Lexical) pour SQL.

Convertit une chaîne SQL en une séquence de tokens identifiables.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Iterator


class TokenType(Enum):
    """Types de tokens SQL."""
    
    # Mots-clés DML
    SELECT = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    MERGE = auto()
    TRUNCATE = auto()
    
    # Mots-clés DDL
    CREATE = auto()
    ALTER = auto()
    DROP = auto()
    TABLE = auto()
    VIEW = auto()
    INDEX = auto()
    SCHEMA = auto()
    DATABASE = auto()
    TEMPORARY = auto()
    TEMP = auto()
    EXTERNAL = auto()
    
    # Clauses DML
    INTO = auto()
    VALUES = auto()
    SET = auto()
    MATCHED = auto()
    
    # Contraintes
    PRIMARY = auto()
    KEY = auto()
    FOREIGN = auto()
    REFERENCES = auto()
    UNIQUE = auto()
    CHECK = auto()
    CONSTRAINT = auto()
    DEFAULT = auto()
    CASCADE = auto()
    RESTRICT = auto()
    COLUMN = auto()
    
    # Modificateurs
    REPLACE = auto()
    CONFLICT = auto()
    NOTHING = auto()
    DO = auto()
    
    # Clauses communes
    FROM = auto()
    WHERE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    BETWEEN = auto()
    LIKE = auto()
    IS = auto()
    NULL = auto()
    TRUE = auto()
    FALSE = auto()
    AS = auto()
    ON = auto()
    USING = auto()
    
    # Jointures
    JOIN = auto()
    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()
    OUTER = auto()
    CROSS = auto()
    NATURAL = auto()
    
    # Clauses
    GROUP = auto()
    BY = auto()
    HAVING = auto()
    ORDER = auto()
    ASC = auto()
    DESC = auto()
    LIMIT = auto()
    OFFSET = auto()
    DISTINCT = auto()
    ALL = auto()
    
    # Fonctions d'agrégation
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    
    # Sous-requêtes
    EXISTS = auto()
    ANY = auto()
    SOME = auto()
    
    # Opérations ensemblistes
    UNION = auto()
    INTERSECT = auto()
    EXCEPT = auto()
    
    # CASE
    CASE = auto()
    WHEN = auto()
    THEN = auto()
    ELSE = auto()
    END = auto()
    
    # Autres mots-clés
    WITH = auto()
    RECURSIVE = auto()
    OVER = auto()
    PARTITION = auto()
    ROWS = auto()
    RANGE = auto()
    NULLS = auto()
    FIRST = auto()
    LAST = auto()
    
    # Presto/Athena spécifiques
    UNNEST = auto()
    ORDINALITY = auto()
    LATERAL = auto()
    TABLESAMPLE = auto()
    BERNOULLI = auto()
    SYSTEM = auto()
    TRY = auto()
    TRY_CAST = auto()
    ARRAY = auto()
    MAP = auto()
    ROW = auto()
    INTERVAL = auto()
    AT = auto()
    ZONE = auto()
    TIME = auto()
    TIMESTAMP = auto()
    DATE = auto()
    GROUPING = auto()
    SETS = auto()
    CUBE = auto()
    ROLLUP = auto()
    FILTER = auto()
    WITHIN = auto()
    PRECEDING = auto()
    FOLLOWING = auto()
    UNBOUNDED = auto()
    CURRENT = auto()
    IF = auto()
    
    # Littéraux
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    IDENTIFIER = auto()
    QUOTED_IDENTIFIER = auto()
    
    # Opérateurs
    EQUALS = auto()           # =
    NOT_EQUALS = auto()       # <> ou !=
    LESS_THAN = auto()        # <
    GREATER_THAN = auto()     # >
    LESS_EQUAL = auto()       # <=
    GREATER_EQUAL = auto()    # >=
    PLUS = auto()             # +
    MINUS = auto()            # -
    MULTIPLY = auto()         # *
    DIVIDE = auto()           # /
    MODULO = auto()           # %
    CONCAT = auto()           # ||
    
    # Ponctuation
    COMMA = auto()            # ,
    DOT = auto()              # .
    SEMICOLON = auto()        # ;
    LPAREN = auto()           # (
    RPAREN = auto()           # )
    STAR = auto()             # * (contexte sélection)
    COLON = auto()            # :
    DOUBLE_COLON = auto()     # ::
    QUESTION = auto()         # ?
    LBRACKET = auto()         # [
    RBRACKET = auto()         # ]
    ARROW = auto()            # ->
    DOUBLE_ARROW = auto()     # =>
    LAMBDA = auto()           # -> (dans contexte lambda)
    
    # Jinja/dbt templates
    JINJA_EXPR = auto()       # {{ ... }}
    JINJA_STMT = auto()       # {% ... %}
    JINJA_COMMENT = auto()    # {# ... #}
    
    # Spéciaux
    COMMENT = auto()
    WHITESPACE = auto()
    NEWLINE = auto()
    EOF = auto()
    UNKNOWN = auto()


@dataclass
class Token:
    """Représente un token SQL."""
    type: TokenType
    value: str
    line: int
    column: int
    position: int  # Position absolue dans le texte
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


class SQLTokenizer:
    """Analyseur lexical pour SQL."""
    
    # Mots-clés SQL (insensible à la casse)
    KEYWORDS = {
        # DML
        'select': TokenType.SELECT,
        'insert': TokenType.INSERT,
        'update': TokenType.UPDATE,
        'delete': TokenType.DELETE,
        'merge': TokenType.MERGE,
        'truncate': TokenType.TRUNCATE,
        # DDL
        'create': TokenType.CREATE,
        'alter': TokenType.ALTER,
        'drop': TokenType.DROP,
        'table': TokenType.TABLE,
        'view': TokenType.VIEW,
        'index': TokenType.INDEX,
        'schema': TokenType.SCHEMA,
        'database': TokenType.DATABASE,
        'temporary': TokenType.TEMPORARY,
        'temp': TokenType.TEMP,
        'external': TokenType.EXTERNAL,
        # DML clauses
        'into': TokenType.INTO,
        'values': TokenType.VALUES,
        'set': TokenType.SET,
        'matched': TokenType.MATCHED,
        # Constraints
        'primary': TokenType.PRIMARY,
        'key': TokenType.KEY,
        'foreign': TokenType.FOREIGN,
        'references': TokenType.REFERENCES,
        'unique': TokenType.UNIQUE,
        'check': TokenType.CHECK,
        'constraint': TokenType.CONSTRAINT,
        'default': TokenType.DEFAULT,
        'cascade': TokenType.CASCADE,
        'restrict': TokenType.RESTRICT,
        # Modifiers
        'replace': TokenType.REPLACE,
        'conflict': TokenType.CONFLICT,
        'nothing': TokenType.NOTHING,
        'do': TokenType.DO,
        # Clauses communes
        'from': TokenType.FROM,
        'where': TokenType.WHERE,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'in': TokenType.IN,
        'between': TokenType.BETWEEN,
        'like': TokenType.LIKE,
        'is': TokenType.IS,
        'null': TokenType.NULL,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'as': TokenType.AS,
        'on': TokenType.ON,
        'using': TokenType.USING,
        'column': TokenType.COLUMN,
        'join': TokenType.JOIN,
        'inner': TokenType.INNER,
        'left': TokenType.LEFT,
        'right': TokenType.RIGHT,
        'full': TokenType.FULL,
        'outer': TokenType.OUTER,
        'cross': TokenType.CROSS,
        'natural': TokenType.NATURAL,
        'group': TokenType.GROUP,
        'by': TokenType.BY,
        'having': TokenType.HAVING,
        'order': TokenType.ORDER,
        'asc': TokenType.ASC,
        'desc': TokenType.DESC,
        'limit': TokenType.LIMIT,
        'offset': TokenType.OFFSET,
        'distinct': TokenType.DISTINCT,
        'all': TokenType.ALL,
        'count': TokenType.COUNT,
        'sum': TokenType.SUM,
        'avg': TokenType.AVG,
        'min': TokenType.MIN,
        'max': TokenType.MAX,
        'exists': TokenType.EXISTS,
        'any': TokenType.ANY,
        'some': TokenType.SOME,
        'union': TokenType.UNION,
        'intersect': TokenType.INTERSECT,
        'except': TokenType.EXCEPT,
        'case': TokenType.CASE,
        'when': TokenType.WHEN,
        'then': TokenType.THEN,
        'else': TokenType.ELSE,
        'end': TokenType.END,
        'with': TokenType.WITH,
        'recursive': TokenType.RECURSIVE,
        'over': TokenType.OVER,
        'partition': TokenType.PARTITION,
        'rows': TokenType.ROWS,
        'range': TokenType.RANGE,
        'nulls': TokenType.NULLS,
        'first': TokenType.FIRST,
        'last': TokenType.LAST,
        # Presto/Athena
        'unnest': TokenType.UNNEST,
        'ordinality': TokenType.ORDINALITY,
        'lateral': TokenType.LATERAL,
        'tablesample': TokenType.TABLESAMPLE,
        'bernoulli': TokenType.BERNOULLI,
        'system': TokenType.SYSTEM,
        'try': TokenType.TRY,
        'try_cast': TokenType.TRY_CAST,
        'array': TokenType.ARRAY,
        'map': TokenType.MAP,
        'row': TokenType.ROW,
        'interval': TokenType.INTERVAL,
        'at': TokenType.AT,
        'zone': TokenType.ZONE,
        'time': TokenType.TIME,
        'timestamp': TokenType.TIMESTAMP,
        'date': TokenType.DATE,
        'grouping': TokenType.GROUPING,
        'sets': TokenType.SETS,
        'cube': TokenType.CUBE,
        'rollup': TokenType.ROLLUP,
        'filter': TokenType.FILTER,
        'within': TokenType.WITHIN,
        'preceding': TokenType.PRECEDING,
        'following': TokenType.FOLLOWING,
        'unbounded': TokenType.UNBOUNDED,
        'current': TokenType.CURRENT,
        'if': TokenType.IF,
    }
    
    def __init__(self, sql: str, include_whitespace: bool = False, include_comments: bool = True):
        """
        Initialise le tokenizer.
        
        Args:
            sql: Le code SQL à tokenizer
            include_whitespace: Inclure les tokens d'espaces blancs
            include_comments: Inclure les tokens de commentaires
        """
        self.sql = sql
        self.include_whitespace = include_whitespace
        self.include_comments = include_comments
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def _current_char(self) -> Optional[str]:
        """Retourne le caractère courant ou None si fin de chaîne."""
        if self.pos >= len(self.sql):
            return None
        return self.sql[self.pos]
    
    def _peek(self, offset: int = 1) -> Optional[str]:
        """Regarde le caractère à offset positions devant."""
        pos = self.pos + offset
        if pos >= len(self.sql):
            return None
        return self.sql[pos]
    
    def _advance(self, count: int = 1) -> str:
        """Avance de count caractères et retourne les caractères consommés."""
        result = self.sql[self.pos:self.pos + count]
        for char in result:
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
        return result
    
    def _make_token(self, token_type: TokenType, value: str, start_line: int, start_col: int, start_pos: int) -> Token:
        """Crée un token."""
        return Token(token_type, value, start_line, start_col, start_pos)
    
    def _skip_whitespace(self) -> Optional[Token]:
        """Consomme les espaces blancs."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = ""
        
        while self._current_char() and self._current_char() in ' \t\r\n':
            value += self._advance()
        
        if value and self.include_whitespace:
            return self._make_token(TokenType.WHITESPACE, value, start_line, start_col, start_pos)
        return None
    
    def _read_single_line_comment(self) -> Token:
        """Lit un commentaire sur une seule ligne (-- ...)."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = self._advance(2)  # Consomme --
        
        while self._current_char() and self._current_char() != '\n':
            value += self._advance()
        
        return self._make_token(TokenType.COMMENT, value, start_line, start_col, start_pos)
    
    def _read_multi_line_comment(self) -> Token:
        """Lit un commentaire multi-ligne (/* ... */)."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = self._advance(2)  # Consomme /*
        
        while self._current_char():
            if self._current_char() == '*' and self._peek() == '/':
                value += self._advance(2)
                break
            value += self._advance()
        
        return self._make_token(TokenType.COMMENT, value, start_line, start_col, start_pos)
    
    def _read_jinja_template(self) -> Token:
        """Lit un template Jinja ({{ }}, {% %}, {# #})."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        
        opening = self._advance(2)  # Consomme {{ ou {% ou {#
        value = opening
        
        # Détermine le type et la fermeture attendue
        if opening == '{{':
            close = '}}'
            token_type = TokenType.JINJA_EXPR
        elif opening == '{%':
            close = '%}'
            token_type = TokenType.JINJA_STMT
        else:  # {#
            close = '#}'
            token_type = TokenType.JINJA_COMMENT
        
        # Lit jusqu'à la fermeture
        depth = 1  # Pour gérer les Jinja imbriqués
        while self._current_char() and depth > 0:
            # Vérifie les ouvertures imbriquées
            two_char = (self._current_char() or '') + (self._peek() or '')
            
            if two_char == close:
                value += self._advance(2)
                depth -= 1
            elif two_char in ('{{', '{%', '{#'):
                value += self._advance(2)
                depth += 1
            else:
                value += self._advance()
        
        return self._make_token(token_type, value, start_line, start_col, start_pos)
    
    def _read_string(self, quote_char: str) -> Token:
        """Lit une chaîne de caractères (entre ' ou ")."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = self._advance()  # Consomme le premier guillemet
        
        while self._current_char():
            char = self._current_char()
            
            # Échappement par doublement du guillemet
            if char == quote_char:
                value += self._advance()
                if self._current_char() == quote_char:
                    value += self._advance()
                else:
                    break
            else:
                value += self._advance()
        
        return self._make_token(TokenType.STRING, value, start_line, start_col, start_pos)
    
    def _read_quoted_identifier(self) -> Token:
        """Lit un identifiant entre guillemets doubles ou backticks."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        quote_char = self._current_char()
        value = self._advance()  # Consomme le premier caractère
        
        while self._current_char() and self._current_char() != quote_char:
            value += self._advance()
        
        if self._current_char() == quote_char:
            value += self._advance()
        
        return self._make_token(TokenType.QUOTED_IDENTIFIER, value, start_line, start_col, start_pos)
    
    def _read_bracket_identifier(self) -> Token:
        """Lit un identifiant entre crochets (style T-SQL: [identifier])."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = self._advance()  # Consomme le '['
        
        while self._current_char() and self._current_char() != ']':
            value += self._advance()
        
        if self._current_char() == ']':
            value += self._advance()
        
        return self._make_token(TokenType.QUOTED_IDENTIFIER, value, start_line, start_col, start_pos)
    
    def _read_number(self) -> Token:
        """Lit un nombre (entier ou flottant)."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = ""
        is_float = False
        
        # Partie entière
        while self._current_char() and self._current_char().isdigit():
            value += self._advance()
        
        # Partie décimale
        if self._current_char() == '.' and self._peek() and self._peek().isdigit():
            is_float = True
            value += self._advance()  # Consomme le point
            while self._current_char() and self._current_char().isdigit():
                value += self._advance()
        
        # Notation scientifique
        if self._current_char() and self._current_char().lower() == 'e':
            next_char = self._peek()
            if next_char and (next_char.isdigit() or next_char in '+-'):
                is_float = True
                value += self._advance()  # Consomme 'e'
                if self._current_char() in '+-':
                    value += self._advance()
                while self._current_char() and self._current_char().isdigit():
                    value += self._advance()
        
        token_type = TokenType.FLOAT if is_float else TokenType.INTEGER
        return self._make_token(token_type, value, start_line, start_col, start_pos)
    
    def _read_identifier_or_keyword(self) -> Token:
        """Lit un identifiant ou un mot-clé."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        value = ""
        
        while self._current_char() and (self._current_char().isalnum() or self._current_char() == '_'):
            value += self._advance()
        
        # Vérifie si c'est un mot-clé
        lower_value = value.lower()
        if lower_value in self.KEYWORDS:
            token_type = self.KEYWORDS[lower_value]
        else:
            token_type = TokenType.IDENTIFIER
        
        return self._make_token(token_type, value, start_line, start_col, start_pos)
    
    def _read_operator_or_punctuation(self) -> Token:
        """Lit un opérateur ou un signe de ponctuation."""
        start_line, start_col, start_pos = self.line, self.column, self.pos
        char = self._current_char()
        next_char = self._peek()
        
        # Opérateurs à deux caractères
        two_char = char + (next_char or '')
        
        if two_char == '<>':
            self._advance(2)
            return self._make_token(TokenType.NOT_EQUALS, two_char, start_line, start_col, start_pos)
        elif two_char == '!=':
            self._advance(2)
            return self._make_token(TokenType.NOT_EQUALS, two_char, start_line, start_col, start_pos)
        elif two_char == '<=':
            self._advance(2)
            return self._make_token(TokenType.LESS_EQUAL, two_char, start_line, start_col, start_pos)
        elif two_char == '>=':
            self._advance(2)
            return self._make_token(TokenType.GREATER_EQUAL, two_char, start_line, start_col, start_pos)
        elif two_char == '||':
            self._advance(2)
            return self._make_token(TokenType.CONCAT, two_char, start_line, start_col, start_pos)
        elif two_char == '::':
            self._advance(2)
            return self._make_token(TokenType.DOUBLE_COLON, two_char, start_line, start_col, start_pos)
        elif two_char == '->':
            self._advance(2)
            return self._make_token(TokenType.ARROW, two_char, start_line, start_col, start_pos)
        elif two_char == '=>':
            self._advance(2)
            return self._make_token(TokenType.DOUBLE_ARROW, two_char, start_line, start_col, start_pos)
        
        # Opérateurs et ponctuation à un caractère
        self._advance()
        
        operators = {
            '=': TokenType.EQUALS,
            '<': TokenType.LESS_THAN,
            '>': TokenType.GREATER_THAN,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            ';': TokenType.SEMICOLON,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ':': TokenType.COLON,
            '?': TokenType.QUESTION,
        }
        
        token_type = operators.get(char, TokenType.UNKNOWN)
        return self._make_token(token_type, char, start_line, start_col, start_pos)
    
    def tokenize(self) -> List[Token]:
        """
        Tokenize la chaîne SQL complète.
        
        Returns:
            Liste de tokens
        """
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.column = 1
        
        while self.pos < len(self.sql):
            char = self._current_char()
            
            # Espaces blancs
            if char in ' \t\r\n':
                token = self._skip_whitespace()
                if token:
                    self.tokens.append(token)
                continue
            
            # Templates Jinja (dbt)
            if char == '{' and self._peek() in ('{', '%', '#'):
                self.tokens.append(self._read_jinja_template())
                continue
            
            # Commentaires
            if char == '-' and self._peek() == '-':
                token = self._read_single_line_comment()
                if self.include_comments:
                    self.tokens.append(token)
                continue
            
            if char == '/' and self._peek() == '*':
                token = self._read_multi_line_comment()
                if self.include_comments:
                    self.tokens.append(token)
                continue
            
            # Chaînes de caractères
            if char == "'":
                self.tokens.append(self._read_string("'"))
                continue
            
            # Identifiants entre guillemets ou crochets (T-SQL: [identifier])
            if char == '"' or char == '`':
                self.tokens.append(self._read_quoted_identifier())
                continue
            
            # Identifiants entre crochets (T-SQL style) ou subscript (ARRAY[...])
            # Si le token précédent est ARRAY, MAP, ou un identifiant, c'est un subscript
            if char == '[':
                # Check if this is an array subscript or a T-SQL identifier
                last_token = self.tokens[-1] if self.tokens else None
                if last_token and last_token.type in (
                    TokenType.ARRAY, TokenType.MAP, TokenType.IDENTIFIER,
                    TokenType.QUOTED_IDENTIFIER, TokenType.RBRACKET, TokenType.RPAREN
                ):
                    # This is a subscript like ARRAY[...] or arr[0]
                    self.tokens.append(self._read_operator_or_punctuation())
                else:
                    # This is a T-SQL bracket identifier like [Column Name]
                    self.tokens.append(self._read_bracket_identifier())
                continue
            
            # Nombres
            if char.isdigit():
                self.tokens.append(self._read_number())
                continue
            
            # Nombres négatifs ou opérateur moins
            if char == '-' and self._peek() and self._peek().isdigit():
                # On laisse le parser gérer le signe négatif
                self.tokens.append(self._read_operator_or_punctuation())
                continue
            
            # Identifiants ou mots-clés
            if char.isalpha() or char == '_':
                self.tokens.append(self._read_identifier_or_keyword())
                continue
            
            # Opérateurs et ponctuation
            self.tokens.append(self._read_operator_or_punctuation())
        
        # Token de fin
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column, self.pos))
        
        return self.tokens
    
    def __iter__(self) -> Iterator[Token]:
        """Permet d'itérer sur les tokens."""
        if not self.tokens:
            self.tokenize()
        return iter(self.tokens)


def tokenize(sql: str, include_whitespace: bool = False, include_comments: bool = True) -> List[Token]:
    """
    Fonction utilitaire pour tokenizer du SQL.
    
    Args:
        sql: Le code SQL à tokenizer
        include_whitespace: Inclure les tokens d'espaces blancs
        include_comments: Inclure les tokens de commentaires
        
    Returns:
        Liste de tokens
    """
    tokenizer = SQLTokenizer(sql, include_whitespace, include_comments)
    return tokenizer.tokenize()
