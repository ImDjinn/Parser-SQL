"""
Tests exhaustifs pour le tokenizer SQL.
"""

import pytest
import sys
sys.path.insert(0, '..')

from sql_parser.tokenizer import SQLTokenizer, TokenType, Token, tokenize


# ============================================================
# SECTION 1: TOKENS DE BASE
# ============================================================

class TestBasicTokens:
    """Tests des tokens de base."""
    
    def test_tokenize_select(self):
        tokens = tokenize("SELECT")
        assert len(tokens) == 2  # SELECT + EOF
        assert tokens[0].type == TokenType.SELECT
    
    def test_tokenize_keywords_case_insensitive(self):
        tokens1 = tokenize("SELECT")
        tokens2 = tokenize("select")
        tokens3 = tokenize("SeLeCt")
        assert tokens1[0].type == tokens2[0].type == tokens3[0].type == TokenType.SELECT
    
    def test_tokenize_identifier(self):
        tokens = tokenize("my_table")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "my_table"
    
    def test_tokenize_identifier_with_numbers(self):
        tokens = tokenize("table123")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "table123"
    
    def test_tokenize_identifier_starting_underscore(self):
        tokens = tokenize("_private")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "_private"


class TestLiterals:
    """Tests des littéraux."""
    
    def test_integer(self):
        tokens = tokenize("42")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == "42"
    
    def test_negative_looking_integer(self):
        # Le signe moins est un token séparé
        tokens = tokenize("-42")
        assert tokens[0].type == TokenType.MINUS
        assert tokens[1].type == TokenType.INTEGER
    
    def test_float(self):
        tokens = tokenize("3.14")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == "3.14"
    
    def test_float_scientific(self):
        tokens = tokenize("1.5e10")
        assert tokens[0].type == TokenType.FLOAT
    
    def test_float_scientific_negative_exponent(self):
        tokens = tokenize("1.5e-10")
        assert tokens[0].type == TokenType.FLOAT
    
    def test_string_single_quotes(self):
        tokens = tokenize("'hello world'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "'hello world'"
    
    def test_string_with_escaped_quote(self):
        tokens = tokenize("'it''s working'")
        assert tokens[0].type == TokenType.STRING
    
    def test_string_empty(self):
        tokens = tokenize("''")
        assert tokens[0].type == TokenType.STRING
    
    def test_null_literal(self):
        tokens = tokenize("NULL")
        assert tokens[0].type == TokenType.NULL
    
    def test_true_literal(self):
        tokens = tokenize("TRUE")
        assert tokens[0].type == TokenType.TRUE
    
    def test_false_literal(self):
        tokens = tokenize("FALSE")
        assert tokens[0].type == TokenType.FALSE


class TestQuotedIdentifiers:
    """Tests des identifiants quotés."""
    
    def test_double_quoted_identifier(self):
        tokens = tokenize('"Column Name"')
        assert tokens[0].type == TokenType.QUOTED_IDENTIFIER
        assert tokens[0].value == '"Column Name"'
    
    def test_backtick_identifier(self):
        tokens = tokenize('`table`')
        assert tokens[0].type == TokenType.QUOTED_IDENTIFIER
        assert tokens[0].value == '`table`'
    
    def test_bracket_identifier_standalone(self):
        # When [ is not preceded by ARRAY/MAP/identifier, it's a bracket identifier
        tokens = tokenize('[Column Name]')
        assert tokens[0].type == TokenType.QUOTED_IDENTIFIER
        assert tokens[0].value == '[Column Name]'
    
    def test_bracket_after_array(self):
        # ARRAY[...] should tokenize [ as LBRACKET
        tokens = tokenize('ARRAY[1, 2]')
        assert tokens[0].type == TokenType.ARRAY
        assert tokens[1].type == TokenType.LBRACKET
        assert tokens[2].type == TokenType.INTEGER


# ============================================================
# SECTION 2: OPÉRATEURS
# ============================================================

class TestOperators:
    """Tests des opérateurs."""
    
    def test_equals(self):
        tokens = tokenize("=")
        assert tokens[0].type == TokenType.EQUALS
    
    def test_not_equals_angle(self):
        tokens = tokenize("<>")
        assert tokens[0].type == TokenType.NOT_EQUALS
    
    def test_not_equals_bang(self):
        tokens = tokenize("!=")
        assert tokens[0].type == TokenType.NOT_EQUALS
    
    def test_less_than(self):
        tokens = tokenize("<")
        assert tokens[0].type == TokenType.LESS_THAN
    
    def test_greater_than(self):
        tokens = tokenize(">")
        assert tokens[0].type == TokenType.GREATER_THAN
    
    def test_less_equal(self):
        tokens = tokenize("<=")
        assert tokens[0].type == TokenType.LESS_EQUAL
    
    def test_greater_equal(self):
        tokens = tokenize(">=")
        assert tokens[0].type == TokenType.GREATER_EQUAL
    
    def test_plus(self):
        tokens = tokenize("+")
        assert tokens[0].type == TokenType.PLUS
    
    def test_minus(self):
        tokens = tokenize("-")
        assert tokens[0].type == TokenType.MINUS
    
    def test_multiply(self):
        tokens = tokenize("*")
        assert tokens[0].type == TokenType.STAR
    
    def test_divide(self):
        tokens = tokenize("/")
        assert tokens[0].type == TokenType.DIVIDE
    
    def test_modulo(self):
        tokens = tokenize("%")
        assert tokens[0].type == TokenType.MODULO
    
    def test_concat(self):
        tokens = tokenize("||")
        assert tokens[0].type == TokenType.CONCAT
    
    def test_double_colon(self):
        tokens = tokenize("::")
        assert tokens[0].type == TokenType.DOUBLE_COLON
    
    def test_arrow(self):
        tokens = tokenize("->")
        assert tokens[0].type == TokenType.ARROW
    
    def test_double_arrow(self):
        tokens = tokenize("=>")
        assert tokens[0].type == TokenType.DOUBLE_ARROW


class TestPunctuation:
    """Tests de la ponctuation."""
    
    def test_comma(self):
        tokens = tokenize(",")
        assert tokens[0].type == TokenType.COMMA
    
    def test_dot(self):
        tokens = tokenize(".")
        assert tokens[0].type == TokenType.DOT
    
    def test_semicolon(self):
        tokens = tokenize(";")
        assert tokens[0].type == TokenType.SEMICOLON
    
    def test_lparen(self):
        tokens = tokenize("(")
        assert tokens[0].type == TokenType.LPAREN
    
    def test_rparen(self):
        tokens = tokenize(")")
        assert tokens[0].type == TokenType.RPAREN
    
    def test_colon(self):
        tokens = tokenize(":")
        assert tokens[0].type == TokenType.COLON
    
    def test_question(self):
        tokens = tokenize("?")
        assert tokens[0].type == TokenType.QUESTION


# ============================================================
# SECTION 3: COMMENTAIRES
# ============================================================

class TestComments:
    """Tests des commentaires."""
    
    def test_single_line_comment(self):
        tokens = tokenize("SELECT -- comment\n* FROM t")
        # Comments are included by default
        comment_tokens = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comment_tokens) == 1
    
    def test_multi_line_comment(self):
        tokens = tokenize("SELECT /* multi\nline */ * FROM t")
        comment_tokens = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comment_tokens) == 1
    
    def test_comment_exclude(self):
        tokenizer = SQLTokenizer("SELECT -- comment\n*", include_comments=False)
        tokens = tokenizer.tokenize()
        comment_tokens = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comment_tokens) == 0


# ============================================================
# SECTION 4: JINJA TEMPLATES
# ============================================================

class TestJinjaTemplates:
    """Tests des templates Jinja."""
    
    def test_jinja_expression(self):
        tokens = tokenize("SELECT {{ column }}")
        jinja_tokens = [t for t in tokens if t.type == TokenType.JINJA_EXPR]
        assert len(jinja_tokens) == 1
    
    def test_jinja_statement(self):
        tokens = tokenize("{% if condition %}SELECT{% endif %}")
        jinja_tokens = [t for t in tokens if t.type == TokenType.JINJA_STMT]
        assert len(jinja_tokens) == 2
    
    def test_jinja_comment(self):
        tokens = tokenize("{# This is a comment #}")
        jinja_tokens = [t for t in tokens if t.type == TokenType.JINJA_COMMENT]
        assert len(jinja_tokens) == 1
    
    def test_jinja_ref(self):
        tokens = tokenize("SELECT * FROM {{ ref('model') }}")
        jinja_tokens = [t for t in tokens if t.type == TokenType.JINJA_EXPR]
        assert len(jinja_tokens) == 1
        assert "ref('model')" in jinja_tokens[0].value
    
    def test_jinja_var(self):
        tokens = tokenize("WHERE date > {{ var('start_date') }}")
        jinja_tokens = [t for t in tokens if t.type == TokenType.JINJA_EXPR]
        assert len(jinja_tokens) == 1


# ============================================================
# SECTION 5: MOTS-CLÉS SQL
# ============================================================

class TestSQLKeywords:
    """Tests des mots-clés SQL."""
    
    def test_dml_keywords(self):
        assert tokenize("SELECT")[0].type == TokenType.SELECT
        assert tokenize("INSERT")[0].type == TokenType.INSERT
        assert tokenize("UPDATE")[0].type == TokenType.UPDATE
        assert tokenize("DELETE")[0].type == TokenType.DELETE
        assert tokenize("MERGE")[0].type == TokenType.MERGE
    
    def test_ddl_keywords(self):
        assert tokenize("CREATE")[0].type == TokenType.CREATE
        assert tokenize("ALTER")[0].type == TokenType.ALTER
        assert tokenize("DROP")[0].type == TokenType.DROP
        assert tokenize("TRUNCATE")[0].type == TokenType.TRUNCATE
    
    def test_clause_keywords(self):
        assert tokenize("FROM")[0].type == TokenType.FROM
        assert tokenize("WHERE")[0].type == TokenType.WHERE
        assert tokenize("GROUP")[0].type == TokenType.GROUP
        assert tokenize("HAVING")[0].type == TokenType.HAVING
        assert tokenize("ORDER")[0].type == TokenType.ORDER
        assert tokenize("LIMIT")[0].type == TokenType.LIMIT
    
    def test_join_keywords(self):
        assert tokenize("JOIN")[0].type == TokenType.JOIN
        assert tokenize("INNER")[0].type == TokenType.INNER
        assert tokenize("LEFT")[0].type == TokenType.LEFT
        assert tokenize("RIGHT")[0].type == TokenType.RIGHT
        assert tokenize("FULL")[0].type == TokenType.FULL
        assert tokenize("CROSS")[0].type == TokenType.CROSS
    
    def test_logical_keywords(self):
        assert tokenize("AND")[0].type == TokenType.AND
        assert tokenize("OR")[0].type == TokenType.OR
        assert tokenize("NOT")[0].type == TokenType.NOT
        assert tokenize("IN")[0].type == TokenType.IN
        assert tokenize("BETWEEN")[0].type == TokenType.BETWEEN
        assert tokenize("LIKE")[0].type == TokenType.LIKE
    
    def test_presto_keywords(self):
        assert tokenize("UNNEST")[0].type == TokenType.UNNEST
        assert tokenize("ARRAY")[0].type == TokenType.ARRAY
        assert tokenize("MAP")[0].type == TokenType.MAP
        assert tokenize("TRY")[0].type == TokenType.TRY
        assert tokenize("TRY_CAST")[0].type == TokenType.TRY_CAST


# ============================================================
# SECTION 6: POSITION ET LIGNE
# ============================================================

class TestTokenPosition:
    """Tests des positions des tokens."""
    
    def test_single_line_positions(self):
        tokens = tokenize("SELECT id FROM t")
        assert tokens[0].line == 1
        assert tokens[0].column == 1
        assert tokens[1].column == 8  # "id"
    
    def test_multi_line_positions(self):
        tokens = tokenize("SELECT\nid\nFROM t")
        assert tokens[0].line == 1  # SELECT
        assert tokens[1].line == 2  # id
        assert tokens[2].line == 3  # FROM
    
    def test_position_absolute(self):
        tokens = tokenize("SELECT id")
        assert tokens[0].position == 0
        assert tokens[1].position == 7


# ============================================================
# SECTION 7: REQUÊTES COMPLÈTES
# ============================================================

class TestCompleteQueries:
    """Tests de tokenization de requêtes complètes."""
    
    def test_simple_select(self):
        sql = "SELECT id, name FROM users WHERE active = TRUE"
        tokens = tokenize(sql)
        token_types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.SELECT in token_types
        assert TokenType.FROM in token_types
        assert TokenType.WHERE in token_types
    
    def test_select_with_join(self):
        sql = "SELECT * FROM a INNER JOIN b ON a.id = b.a_id"
        tokens = tokenize(sql)
        token_types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.INNER in token_types
        assert TokenType.JOIN in token_types
        assert TokenType.ON in token_types
    
    def test_insert_values(self):
        sql = "INSERT INTO users (id, name) VALUES (1, 'John')"
        tokens = tokenize(sql)
        token_types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.INSERT in token_types
        assert TokenType.INTO in token_types
        assert TokenType.VALUES in token_types
    
    def test_create_table(self):
        sql = "CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(100))"
        tokens = tokenize(sql)
        token_types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.CREATE in token_types
        assert TokenType.TABLE in token_types
        assert TokenType.PRIMARY in token_types
        assert TokenType.KEY in token_types


# ============================================================
# SECTION 8: CAS LIMITES
# ============================================================

class TestEdgeCases:
    """Tests des cas limites."""
    
    def test_empty_string(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_whitespace_only(self):
        tokens = tokenize("   \n\t   ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_unicode_in_string(self):
        tokens = tokenize("'héllo wörld'")
        assert tokens[0].type == TokenType.STRING
    
    def test_very_long_identifier(self):
        long_id = "a" * 1000
        tokens = tokenize(long_id)
        assert tokens[0].type == TokenType.IDENTIFIER
        assert len(tokens[0].value) == 1000
    
    def test_consecutive_operators(self):
        tokens = tokenize("+-*/")
        assert tokens[0].type == TokenType.PLUS
        assert tokens[1].type == TokenType.MINUS
        assert tokens[2].type == TokenType.STAR
        assert tokens[3].type == TokenType.DIVIDE
    
    def test_keyword_as_part_of_identifier(self):
        # "selected" should be an identifier, not SELECT + ed
        tokens = tokenize("selected")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "selected"
    
    def test_number_followed_by_identifier(self):
        # Should be two tokens
        tokens = tokenize("123abc")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[1].type == TokenType.IDENTIFIER


# ============================================================
# SECTION 9: TOKENIZER CLASS
# ============================================================

class TestTokenizerClass:
    """Tests de la classe SQLTokenizer."""
    
    def test_iteration(self):
        tokenizer = SQLTokenizer("SELECT *")
        tokens = list(tokenizer)
        assert len(tokens) >= 2
    
    def test_include_whitespace(self):
        tokenizer = SQLTokenizer("SELECT *", include_whitespace=True)
        tokens = tokenizer.tokenize()
        ws_tokens = [t for t in tokens if t.type == TokenType.WHITESPACE]
        assert len(ws_tokens) >= 1
    
    def test_exclude_whitespace_default(self):
        tokenizer = SQLTokenizer("SELECT *", include_whitespace=False)
        tokens = tokenizer.tokenize()
        ws_tokens = [t for t in tokens if t.type == TokenType.WHITESPACE]
        assert len(ws_tokens) == 0
    
    def test_token_repr(self):
        token = Token(TokenType.SELECT, "SELECT", 1, 1, 0)
        repr_str = repr(token)
        assert "SELECT" in repr_str
        assert "line=1" in repr_str


# ============================================================
# SECTION 10: TOKENS SPÉCIAUX PRESTO/ATHENA
# ============================================================

class TestPrestoTokens:
    """Tests des tokens spécifiques Presto/Athena."""
    
    def test_interval(self):
        tokens = tokenize("INTERVAL '1' DAY")
        assert tokens[0].type == TokenType.INTERVAL
    
    def test_at_time_zone(self):
        tokens = tokenize("AT TIME ZONE")
        assert tokens[0].type == TokenType.AT
        assert tokens[1].type == TokenType.TIME
        assert tokens[2].type == TokenType.ZONE
    
    def test_row(self):
        tokens = tokenize("ROW(1, 2)")
        assert tokens[0].type == TokenType.ROW
    
    def test_ordinality(self):
        tokens = tokenize("WITH ORDINALITY")
        assert tokens[0].type == TokenType.WITH
        assert tokens[1].type == TokenType.ORDINALITY
    
    def test_tablesample(self):
        tokens = tokenize("TABLESAMPLE BERNOULLI(10)")
        assert tokens[0].type == TokenType.TABLESAMPLE
        assert tokens[1].type == TokenType.BERNOULLI


# ============================================================
# SECTION 11: TOKENS T-SQL
# ============================================================

class TestTSQLTokens:
    """Tests des tokens spécifiques T-SQL."""
    
    def test_bracket_identifier_with_spaces(self):
        tokens = tokenize("[First Name]")
        assert tokens[0].type == TokenType.QUOTED_IDENTIFIER
        assert "[First Name]" in tokens[0].value
    
    def test_bracket_identifier_with_keyword(self):
        tokens = tokenize("[SELECT]")
        assert tokens[0].type == TokenType.QUOTED_IDENTIFIER
    
    def test_mixed_brackets_and_array(self):
        # [col] then ARRAY[1] - should differentiate
        tokens = tokenize("[col] ARRAY[1]")
        assert tokens[0].type == TokenType.QUOTED_IDENTIFIER  # [col]
        assert tokens[1].type == TokenType.ARRAY
        assert tokens[2].type == TokenType.LBRACKET  # [ from ARRAY[


# ============================================================
# SECTION 12: WINDOW FUNCTION TOKENS
# ============================================================

class TestWindowFunctionTokens:
    """Tests des tokens pour window functions."""
    
    def test_over(self):
        assert tokenize("OVER")[0].type == TokenType.OVER
    
    def test_partition(self):
        assert tokenize("PARTITION")[0].type == TokenType.PARTITION
    
    def test_rows(self):
        assert tokenize("ROWS")[0].type == TokenType.ROWS
    
    def test_range(self):
        assert tokenize("RANGE")[0].type == TokenType.RANGE
    
    def test_preceding(self):
        assert tokenize("PRECEDING")[0].type == TokenType.PRECEDING
    
    def test_following(self):
        assert tokenize("FOLLOWING")[0].type == TokenType.FOLLOWING
    
    def test_unbounded(self):
        assert tokenize("UNBOUNDED")[0].type == TokenType.UNBOUNDED
    
    def test_current(self):
        assert tokenize("CURRENT")[0].type == TokenType.CURRENT
