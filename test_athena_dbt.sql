-- Modèle dbt typique pour Athena (Presto SQL)
-- Avec templates Jinja, UNNEST, ARRAY, fonctions Presto, etc.

{{ config(
    materialized='table',
    partitioned_by=['year', 'month']
) }}

WITH 
-- Extraction des événements avec UNNEST
events_exploded AS (
    SELECT 
        event_id,
        user_id,
        event_timestamp,
        event_type,
        property_key,
        property_value
    FROM {{ ref('raw_events') }}
    CROSS JOIN UNNEST(properties) AS t(property_key, property_value)
    WHERE event_date >= DATE '{{ var("start_date") }}'
      AND event_date < DATE '{{ var("end_date") }}'
),

-- Utilisation de fonctions Presto/Athena spécifiques
user_sessions AS (
    SELECT 
        user_id,
        date_trunc('hour', event_timestamp) AS session_hour,
        array_agg(DISTINCT event_type) AS event_types,
        count(DISTINCT event_id) AS event_count,
        approx_percentile(CAST(json_extract_scalar(property_value, '$.duration') AS DOUBLE), 0.95) AS p95_duration,
        arbitrary(property_value) AS sample_property
    FROM events_exploded
    WHERE property_key = 'session_data'
    GROUP BY user_id, date_trunc('hour', event_timestamp)
),

-- Transformation avec ARRAY et MAP
aggregated_data AS (
    SELECT 
        user_id,
        session_hour,
        cardinality(event_types) AS unique_event_types,
        event_count,
        p95_duration,
        -- Utilisation de transform et filter sur arrays
        transform(event_types, x -> upper(x)) AS event_types_upper,
        filter(event_types, x -> x LIKE 'click%') AS click_events,
        -- Construction de MAP
        map_agg(
            'total_events', 
            CAST(event_count AS VARCHAR)
        ) AS metrics_map,
        -- Condition avec IF
        IF(event_count > 100, 'high_activity', 'normal') AS activity_level,
        -- TRY pour gérer les erreurs
        TRY(CAST(sample_property AS JSON)) AS parsed_json
    FROM user_sessions
),

-- Données de référence avec Jinja
user_segments AS (
    SELECT 
        user_id,
        segment_name,
        segment_score
    FROM {{ ref('user_segments') }}
    WHERE is_active = true
)

SELECT 
    a.user_id,
    a.session_hour,
    a.unique_event_types,
    a.event_count,
    a.p95_duration,
    a.activity_level,
    s.segment_name,
    s.segment_score,
    -- CASE avec BETWEEN
    CASE 
        WHEN s.segment_score BETWEEN 0 AND 25 THEN 'low'
        WHEN s.segment_score BETWEEN 26 AND 75 THEN 'medium'
        ELSE 'high'
    END AS score_category,
    -- Conversion de timestamp avec AT TIME ZONE
    a.session_hour AT TIME ZONE 'Europe/Paris' AS session_hour_paris,
    -- Extraction année/mois pour partitionnement
    year(a.session_hour) AS year,
    month(a.session_hour) AS month,
    -- Génération d'un identifiant unique
    {{ dbt_utils.generate_surrogate_key(['a.user_id', 'a.session_hour']) }} AS session_key
FROM aggregated_data a
LEFT JOIN user_segments s ON a.user_id = s.user_id
WHERE a.event_count > 0
  AND a.session_hour >= TIMESTAMP '2025-01-01 00:00:00'
  AND s.segment_name IN (
      SELECT DISTINCT segment_name 
      FROM {{ ref('active_segments') }}
  )
ORDER BY a.event_count DESC
LIMIT 10000
