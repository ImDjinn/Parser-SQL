"""
Dialectes SQL supportés par le parser.

Ce module définit les différents dialectes SQL et leurs spécificités.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional


class SQLDialect(Enum):
    """Dialectes SQL supportés."""
    STANDARD = "standard"      # SQL standard (ANSI)
    PRESTO = "presto"          # Presto SQL
    ATHENA = "athena"          # AWS Athena (basé sur Presto/Trino)
    TRINO = "trino"            # Trino (anciennement PrestoSQL)
    POSTGRESQL = "postgresql"  # PostgreSQL
    MYSQL = "mysql"            # MySQL
    BIGQUERY = "bigquery"      # Google BigQuery
    SNOWFLAKE = "snowflake"    # Snowflake
    SPARK = "spark"            # Spark SQL
    TSQL = "tsql"              # T-SQL (SQL Server)


@dataclass
class DialectFeatures:
    """Caractéristiques d'un dialecte SQL."""
    
    # Nom du dialecte
    name: str
    
    # Fonctions spécifiques au dialecte
    functions: Set[str] = field(default_factory=set)
    
    # Types de données spécifiques
    data_types: Set[str] = field(default_factory=set)
    
    # Mots-clés additionnels
    keywords: Set[str] = field(default_factory=set)
    
    # Supporte les lambdas (x -> x + 1)
    supports_lambda: bool = False
    
    # Supporte UNNEST
    supports_unnest: bool = False
    
    # Supporte les tableaux []
    supports_array_subscript: bool = False
    
    # Supporte les structs/maps avec .field
    supports_struct_access: bool = False
    
    # Supporte TABLESAMPLE
    supports_tablesample: bool = False
    
    # Supporte les commentaires de type dbt ({{ }}, {# #}, {% %})
    supports_jinja: bool = False
    
    # Supporte CROSS JOIN UNNEST
    supports_cross_join_unnest: bool = False
    
    # Supporte AT TIME ZONE
    supports_at_time_zone: bool = False
    
    # Supporte TRY_CAST
    supports_try_cast: bool = False
    
    # Supporte FORMAT pour les dates
    supports_format: bool = False
    
    # Supporte les types ROW
    supports_row_type: bool = False
    
    # Supporte GROUPING SETS, CUBE, ROLLUP
    supports_grouping_sets: bool = False
    
    # Caractère d'échappement pour les identifiants
    identifier_quote: str = '"'
    
    # Supporte les backticks pour les identifiants
    supports_backtick_identifiers: bool = False
    
    # Supporte INTERVAL
    supports_interval: bool = False
    
    # Supporte les window functions avec OVER
    supports_window_functions: bool = True
    
    # Supporte TOP N (T-SQL)
    supports_top: bool = False
    
    # Supporte les crochets pour identifiants [column]
    supports_bracket_identifiers: bool = False
    
    # Supporte les variables @variable
    supports_at_variables: bool = False
    
    # Supporte CONVERT (T-SQL)
    supports_convert: bool = False
    
    # Supporte IIF (T-SQL)
    supports_iif: bool = False


# Fonctions T-SQL (SQL Server)
TSQL_FUNCTIONS = {
    # Fonctions d'agrégation
    'avg', 'count', 'count_big', 'max', 'min', 'sum', 'stdev', 'stdevp',
    'var', 'varp', 'grouping', 'grouping_id', 'string_agg',
    
    # Fonctions de chaîne
    'ascii', 'char', 'charindex', 'concat', 'concat_ws', 'datalength',
    'difference', 'format', 'left', 'len', 'lower', 'ltrim', 'nchar',
    'patindex', 'quotename', 'replace', 'replicate', 'reverse', 'right',
    'rtrim', 'soundex', 'space', 'str', 'string_escape', 'string_split',
    'stuff', 'substring', 'translate', 'trim', 'unicode', 'upper',
    
    # Fonctions de date/heure
    'current_timestamp', 'dateadd', 'datediff', 'datediff_big', 'datefromparts',
    'datename', 'datepart', 'datetime2fromparts', 'datetimefromparts',
    'datetimeoffsetfromparts', 'day', 'eomonth', 'getdate', 'getutcdate',
    'isdate', 'month', 'smalldatetimefromparts', 'switchoffset', 'sysdatetime',
    'sysdatetimeoffset', 'sysutcdatetime', 'timefromparts', 'todatetimeoffset',
    'year',
    
    # Fonctions de conversion
    'cast', 'convert', 'parse', 'try_cast', 'try_convert', 'try_parse',
    
    # Fonctions conditionnelles
    'coalesce', 'iif', 'isnull', 'nullif', 'choose',
    
    # Fonctions mathématiques
    'abs', 'acos', 'asin', 'atan', 'atn2', 'ceiling', 'cos', 'cot',
    'degrees', 'exp', 'floor', 'log', 'log10', 'pi', 'power', 'radians',
    'rand', 'round', 'sign', 'sin', 'sqrt', 'square', 'tan',
    
    # Fonctions de fenêtre
    'row_number', 'rank', 'dense_rank', 'ntile', 'lag', 'lead',
    'first_value', 'last_value', 'cume_dist', 'percent_rank',
    'percentile_cont', 'percentile_disc',
    
    # Fonctions JSON
    'isjson', 'json_value', 'json_query', 'json_modify', 'json_path_exists',
    
    # Autres
    'newid', 'newsequentialid', 'rowcount_big', 'scope_identity',
    'serverproperty', 'sessionproperty', 'session_user', 'system_user',
    'user_name', 'host_name', 'db_name', 'object_id', 'object_name',
}

# Types de données T-SQL
TSQL_DATA_TYPES = {
    'bigint', 'int', 'smallint', 'tinyint', 'bit',
    'decimal', 'numeric', 'money', 'smallmoney',
    'float', 'real',
    'date', 'time', 'datetime', 'datetime2', 'smalldatetime', 'datetimeoffset',
    'char', 'varchar', 'text', 'nchar', 'nvarchar', 'ntext',
    'binary', 'varbinary', 'image',
    'uniqueidentifier', 'xml', 'sql_variant',
    'geography', 'geometry', 'hierarchyid',
}

# Mots-clés T-SQL
TSQL_KEYWORDS = {
    'top', 'percent', 'with', 'ties', 'offset', 'fetch', 'next', 'only',
    'identity', 'rowguidcol', 'filestream', 'sparse', 'not', 'for', 'replication',
    'go', 'use', 'exec', 'execute', 'sp_executesql', 'print', 'raiserror',
    'throw', 'try', 'catch', 'begin', 'end', 'if', 'else', 'while',
    'break', 'continue', 'return', 'goto', 'waitfor', 'declare', 'set',
    'output', 'inserted', 'deleted', 'merge', 'matched', 'pivot', 'unpivot',
    'cross', 'apply', 'outer', 'openrowset', 'openquery', 'opendatasource',
}


# Fonctions communes Presto/Athena/Trino
PRESTO_FUNCTIONS = {
    # Fonctions d'agrégation
    'approx_distinct', 'approx_percentile', 'arbitrary', 'array_agg',
    'avg', 'bool_and', 'bool_or', 'checksum', 'count', 'count_if',
    'every', 'geometric_mean', 'histogram', 'map_agg', 'map_union',
    'max', 'max_by', 'min', 'min_by', 'multimap_agg', 'sum',
    
    # Fonctions de tableau
    'array_distinct', 'array_except', 'array_intersect', 'array_join',
    'array_max', 'array_min', 'array_position', 'array_remove',
    'array_sort', 'array_union', 'cardinality', 'concat', 'contains',
    'element_at', 'filter', 'flatten', 'reduce', 'repeat', 'reverse',
    'sequence', 'shuffle', 'slice', 'transform', 'zip', 'zip_with',
    
    # Fonctions de map
    'map', 'map_concat', 'map_entries', 'map_filter', 'map_from_entries',
    'map_keys', 'map_values', 'map_zip_with', 'transform_keys',
    'transform_values',
    
    # Fonctions de chaîne
    'chr', 'codepoint', 'concat_ws', 'format', 'from_utf8',
    'length', 'levenshtein_distance', 'lower', 'lpad', 'ltrim',
    'normalize', 'position', 'replace', 'reverse', 'rpad', 'rtrim',
    'soundex', 'split', 'split_part', 'split_to_map', 'strpos',
    'substr', 'substring', 'to_utf8', 'translate', 'trim', 'upper',
    'word_stem',
    
    # Fonctions de date/heure
    'current_date', 'current_time', 'current_timestamp', 'current_timezone',
    'date', 'date_add', 'date_diff', 'date_format', 'date_parse',
    'date_trunc', 'day', 'day_of_month', 'day_of_week', 'day_of_year',
    'dow', 'doy', 'extract', 'from_iso8601_date', 'from_iso8601_timestamp',
    'from_unixtime', 'hour', 'last_day_of_month', 'localtimestamp',
    'millisecond', 'minute', 'month', 'now', 'parse_datetime', 'quarter',
    'second', 'timezone_hour', 'timezone_minute', 'to_iso8601',
    'to_milliseconds', 'to_unixtime', 'week', 'week_of_year', 'year',
    'year_of_week', 'yow', 'at_timezone',
    
    # Fonctions de conversion
    'cast', 'try_cast', 'typeof', 'format',
    
    # Fonctions conditionnelles
    'coalesce', 'if', 'nullif', 'try',
    
    # Fonctions JSON
    'is_json_scalar', 'json_array_contains', 'json_array_get',
    'json_array_length', 'json_extract', 'json_extract_scalar',
    'json_format', 'json_parse', 'json_size',
    
    # Fonctions de fenêtre
    'cume_dist', 'dense_rank', 'first_value', 'lag', 'last_value',
    'lead', 'nth_value', 'ntile', 'percent_rank', 'rank', 'row_number',
    
    # Fonctions URL
    'url_decode', 'url_encode', 'url_extract_fragment',
    'url_extract_host', 'url_extract_parameter', 'url_extract_path',
    'url_extract_port', 'url_extract_protocol', 'url_extract_query',
    
    # Fonctions géospatiales
    'st_area', 'st_centroid', 'st_contains', 'st_crosses',
    'st_difference', 'st_dimension', 'st_disjoint', 'st_distance',
    'st_endpoint', 'st_envelope', 'st_equals', 'st_exteriorring',
    
    # Autres
    'approx_distinct', 'arbitrary', 'uuid', 'regexp_extract',
    'regexp_extract_all', 'regexp_like', 'regexp_replace', 'regexp_split',
}

# Types de données Presto/Athena
PRESTO_DATA_TYPES = {
    'boolean', 'tinyint', 'smallint', 'integer', 'int', 'bigint',
    'real', 'double', 'decimal', 'varchar', 'char', 'varbinary',
    'json', 'date', 'time', 'timestamp', 'interval',
    'array', 'map', 'row', 'ipaddress', 'uuid',
    'hyperloglog', 'p4hyperloglog', 'qdigest',
}

# Mots-clés additionnels Presto/Athena
PRESTO_KEYWORDS = {
    'unnest', 'ordinality', 'tablesample', 'bernoulli', 'system',
    'lateral', 'array', 'map', 'row', 'try', 'lambda',
    'filter', 'transform', 'reduce', 'format', 'at', 'zone',
    'grouping', 'sets', 'cube', 'rollup', 'interval',
    'preceding', 'following', 'unbounded', 'current',
}


def get_dialect_features(dialect: SQLDialect) -> DialectFeatures:
    """Retourne les caractéristiques d'un dialecte."""
    
    if dialect == SQLDialect.STANDARD:
        return DialectFeatures(
            name="SQL Standard",
            supports_window_functions=True,
        )
    
    elif dialect in (SQLDialect.PRESTO, SQLDialect.ATHENA, SQLDialect.TRINO):
        return DialectFeatures(
            name=dialect.value.title(),
            functions=PRESTO_FUNCTIONS,
            data_types=PRESTO_DATA_TYPES,
            keywords=PRESTO_KEYWORDS,
            supports_lambda=True,
            supports_unnest=True,
            supports_array_subscript=True,
            supports_struct_access=True,
            supports_tablesample=True,
            supports_jinja=(dialect == SQLDialect.ATHENA),  # dbt-athena utilise Jinja
            supports_cross_join_unnest=True,
            supports_at_time_zone=True,
            supports_try_cast=True,
            supports_format=True,
            supports_row_type=True,
            supports_grouping_sets=True,
            identifier_quote='"',
            supports_backtick_identifiers=False,
            supports_interval=True,
            supports_window_functions=True,
        )
    
    elif dialect == SQLDialect.POSTGRESQL:
        return DialectFeatures(
            name="PostgreSQL",
            functions={'array_agg', 'string_agg', 'jsonb_agg', 'generate_series'},
            supports_array_subscript=True,
            supports_window_functions=True,
            identifier_quote='"',
            supports_interval=True,
        )
    
    elif dialect == SQLDialect.MYSQL:
        return DialectFeatures(
            name="MySQL",
            functions={'group_concat', 'json_extract', 'json_object'},
            identifier_quote='`',
            supports_backtick_identifiers=True,
        )
    
    elif dialect == SQLDialect.BIGQUERY:
        return DialectFeatures(
            name="BigQuery",
            functions={'array_agg', 'struct', 'safe_cast', 'parse_date'},
            supports_array_subscript=True,
            supports_struct_access=True,
            supports_unnest=True,
            identifier_quote='`',
            supports_backtick_identifiers=True,
        )
    
    elif dialect == SQLDialect.SNOWFLAKE:
        return DialectFeatures(
            name="Snowflake",
            functions={'array_agg', 'object_construct', 'flatten', 'parse_json'},
            supports_array_subscript=True,
            supports_struct_access=True,
            identifier_quote='"',
        )
    
    elif dialect == SQLDialect.SPARK:
        return DialectFeatures(
            name="Spark SQL",
            functions=PRESTO_FUNCTIONS,  # Spark est très similaire à Presto
            supports_lambda=True,
            supports_unnest=True,
            supports_array_subscript=True,
            supports_struct_access=True,
            identifier_quote='`',
            supports_backtick_identifiers=True,
        )
    
    elif dialect == SQLDialect.TSQL:
        return DialectFeatures(
            name="T-SQL (SQL Server)",
            functions=TSQL_FUNCTIONS,
            data_types=TSQL_DATA_TYPES,
            keywords=TSQL_KEYWORDS,
            supports_window_functions=True,
            supports_grouping_sets=True,
            identifier_quote='"',
            supports_bracket_identifiers=True,
            supports_at_variables=True,
            supports_convert=True,
            supports_iif=True,
            supports_top=True,
        )
    
    return DialectFeatures(name="Unknown")


def detect_dialect(sql: str) -> SQLDialect:
    """
    Tente de détecter le dialecte SQL à partir du code.
    
    Args:
        sql: Code SQL à analyser
        
    Returns:
        Dialecte détecté
    """
    sql_lower = sql.lower()
    
    # Détection Jinja (dbt)
    if '{{' in sql or '{%' in sql or '{#' in sql:
        # Probablement dbt, vérifier si c'est Athena/Presto
        if any(f in sql_lower for f in ['unnest', 'try_cast', 'array_agg', 'map_agg']):
            return SQLDialect.ATHENA
    
    # Détection Presto/Athena/Trino
    presto_indicators = [
        'unnest', 'try_cast', 'array[', 'map(', 'row(',
        'at time zone', 'with ordinality', 'cross join unnest',
        'approx_distinct', 'approx_percentile', 'transform(',
        'filter(', 'reduce(', 'tablesample',
    ]
    if any(ind in sql_lower for ind in presto_indicators):
        return SQLDialect.ATHENA  # Athena est le plus courant avec dbt
    
    # Détection BigQuery
    if 'safe_cast' in sql_lower or '`' in sql and 'struct' in sql_lower:
        return SQLDialect.BIGQUERY
    
    # Détection Snowflake
    if 'flatten(' in sql_lower or 'object_construct' in sql_lower:
        return SQLDialect.SNOWFLAKE
    
    # Détection MySQL
    if 'group_concat' in sql_lower or sql.count('`') > 2:
        return SQLDialect.MYSQL
    
    # Détection PostgreSQL
    if '::' in sql or 'generate_series' in sql_lower:
        return SQLDialect.POSTGRESQL
    
    # Détection T-SQL (SQL Server)
    tsql_indicators = [
        'getdate()', 'isnull(', 'charindex(', 'len(', 'dateadd(',
        'datediff(', 'convert(', 'iif(', 'top ', 'with (nolock)',
        'identity(', 'scope_identity', 'newid()', '@@',
    ]
    if any(ind in sql_lower for ind in tsql_indicators):
        return SQLDialect.TSQL
    # Also check for [bracket] identifiers
    if '[' in sql and ']' in sql:
        return SQLDialect.TSQL
    
    return SQLDialect.STANDARD
