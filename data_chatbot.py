import os
import re
import json
import requests
import psycopg2
from decimal import Decimal
from contextlib import contextmanager
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

DB_CONFIG = {
    "host": os.environ.get("DWH_DB_HOST"),
    "port": int(os.environ.get("DWH_DB_PORT")),
    "dbname": os.environ.get("DWH_DB_NAME"),
    "user": os.environ.get("DWH_DB_USER"),
    "password": os.environ.get("DWH_DB_PASSWORD"),
}

SUPERSET_URL = os.environ.get("SUPERSET_URL", "http://localhost:8088")
# Ayrı bir "genel/tarayıcı" adresi -- SUPERSET_URL container'ın Superset'e API
# çağrıları için kullandığı adres (örn. host.docker.internal), bu ise
# kullanıcıya gösterilen linkte kullanılan, tarayıcıdan erişilebilen adres
# (örn. localhost). Verilmezse SUPERSET_URL ile aynı kabul edilir.
SUPERSET_PUBLIC_URL = os.environ.get("SUPERSET_PUBLIC_URL", SUPERSET_URL)
SUPERSET_USERNAME = os.environ.get("SUPERSET_USERNAME")
SUPERSET_PASSWORD = os.environ.get("SUPERSET_PASSWORD")
DATASET_ID = int(os.environ.get("SUPERSET_DATASET_ID"))

# Tables the generated SQL is allowed to touch. Anything else gets rejected.
ALLOWED_TABLES = {
    "dim_customer", "dim_product", "dim_seller", "dim_date", "fact_order_items"
}

# Superset payload shape for each supported chart type.
CHART_SPECS = {
    "line": {
        "viz_type": "echarts_timeseries_line",
        "allowed_axis_roles": {"temporal"},
        "metric_as_list": True,
        "supports_contribution": True,
    },
    "bar": {
        "viz_type": "echarts_timeseries_bar",
        "allowed_axis_roles": {"categorical", "temporal"},
        "metric_as_list": True,
        "supports_contribution": True,
    },
    "pie": {
        "viz_type": "pie",
        "allowed_axis_roles": {"categorical"},
        "metric_as_list": False,
        "supports_contribution": False,
    },
}

VALID_AGGREGATES = {"SUM", "AVG", "COUNT", "MIN", "MAX"}
VALID_FILTER_OPERATORS = {"==", "!=", ">", "<", ">=", "<="}
VALID_TIME_GRAINS = {"P1D", "P1W", "P1M", "P3M", "P1Y"}

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


# ============================================================
# Database helpers
# ============================================================

@contextmanager
def get_db_cursor():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        conn.close()


def fetch_data_catalog():
    """Loads the metadata table that describes every question-able column."""
    with get_db_cursor() as cursor:
        cursor.execute(
            "SELECT friendly_name, source_table, source_column, description, join_info, dataset_id FROM data_catalog;"
        )
        return cursor.fetchall()


def run_sql(sql):
    with get_db_cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


# ============================================================
# Schema retrieval
# ============================================================

def get_all_columns(catalog):
    """Converts raw data_catalog rows into a list of dicts the rest of the code uses."""
    return [
        {
            "friendly_name": row[0],
            "source_table": row[1],
            "source_column": row[2],
            "description": row[3],
            "join_info": row[4],
            "dataset_id": row[5],
        }
        for row in catalog
    ]


def build_schema_context(relevant_columns):
    """Turns the column list into the text block the LLM prompts embed."""
    column_lines = [
        f"- Table: {c['source_table']} | Column: {c['source_column']} | Meaning: {c['friendly_name']} | {c['description']}"
        for c in relevant_columns
    ]
    join_lines = {c["join_info"] for c in relevant_columns if c["join_info"]}
    return "\n".join(column_lines), "\n".join(join_lines) if join_lines else "None"


# ============================================================
# Column type enrichment (used for chart building)
# ============================================================

def get_column_types(relevant_columns):
    """Looks up each column's real PostgreSQL data type from information_schema."""
    tables_cols = {}
    for c in relevant_columns:
        tables_cols.setdefault(c["source_table"], set()).add(c["source_column"])

    type_map = {}
    with get_db_cursor() as cursor:
        for table, cols in tables_cols.items():
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = ANY(%s)
                """,
                (table, list(cols)),
            )
            for col_name, data_type in cursor.fetchall():
                type_map[(table, col_name)] = data_type
    return type_map


NUMERIC_SQL_TYPES = ("integer", "bigint", "numeric", "double precision", "real", "smallint")

# Integer columns that are actually categories/groupings, not summable values
# (e.g. dim_date.quarter is 1/2/3/4 -- useful as an axis, meaningless to SUM).
CATEGORICAL_INT_COLUMNS = {"year", "quarter", "month", "day", "day_of_week"}


def classify_column(data_type, column_name=None):
    """Maps a SQL type + column name to one of: temporal / numeric / categorical."""
    if data_type in ("date", "timestamp", "timestamp without time zone", "timestamptz"):
        return "temporal"
    if column_name in CATEGORICAL_INT_COLUMNS:
        return "categorical"
    if data_type in NUMERIC_SQL_TYPES:
        return "numeric"
    return "categorical"


def enrich_columns(relevant_columns):
    """Adds data_type and role (temporal/numeric/categorical) to each column."""
    type_map = get_column_types(relevant_columns)
    enriched = []
    for c in relevant_columns:
        data_type = type_map.get((c["source_table"], c["source_column"]), "unknown")
        role = classify_column(data_type, c["source_column"])
        enriched.append({**c, "data_type": data_type, "role": role})
    return enriched


# ============================================================
# Time range validation
# ============================================================

def is_valid_time_range(time_range):
    """
    Accepted format: "START : END", where START/END are each either empty
    or YYYY-MM-DD (both empty is not allowed). Matches Superset's
    TEMPORAL_RANGE filter format.
    """
    if " : " not in time_range:
        return False
    start, end = time_range.split(" : ", 1)
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    start_ok = start == "" or date_pattern.match(start)
    end_ok = end == "" or date_pattern.match(end)
    return bool(start_ok and end_ok and (start != "" or end != ""))


# ============================================================
# SQL safety
# ============================================================

def is_select_only(sql):
    return sql.strip().upper().startswith("SELECT")


def extract_table_names(sql):
    pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    return {m.lower() for m in re.findall(pattern, sql, re.IGNORECASE)}


def is_sql_safe(sql, allowed_tables=ALLOWED_TABLES):
    """Rejects anything that isn't a plain SELECT over the whitelisted tables."""
    if not is_select_only(sql):
        return False, "Only SELECT queries are allowed."
    disallowed = extract_table_names(sql) - allowed_tables
    if disallowed:
        return False, f"Disallowed table(s) used: {', '.join(disallowed)}"
    return True, "Safe."


# ============================================================
# Result formatting
# ============================================================

def format_results(rows, limit=10):
    lines = []
    for row in rows[:limit]:
        parts = []
        for value in row:
            if isinstance(value, Decimal):
                parts.append(f"{float(value):,.2f}")
            elif isinstance(value, str):
                parts.append(value.title())
            else:
                parts.append(str(value))
        lines.append(" -> ".join(parts))
    return "\n".join(lines)


# ============================================================
# SQL answer path: user question -> SQL -> result text
# ============================================================

def answer_with_sql(user_question, relevant_columns):
    columns_summary, joins_summary = build_schema_context(relevant_columns)

    prompt = f"""
Based on the database schema below, translate the user's question into a PostgreSQL query.

RULES:
- ONLY use the tables and columns listed below, never invent columns.
- When joining tables, ONLY use the joins listed in "Join info".
- If the question cannot be answered with the columns below, respond with:
  SQL: NONE
  EXPLANATION: No relevant data found for this question.
- Otherwise, respond in EXACTLY this format, nothing else:

SQL: <SQL code here>
EXPLANATION: <1-2 sentence explanation in Turkish of what this SQL does>

Columns:
{columns_summary}

Join info:
{joins_summary}

Question: {user_question}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()

    if "SQL:" in answer and "EXPLANATION:" in answer:
        sql = answer.split("SQL:")[1].split("EXPLANATION:")[0].strip()
        explanation = answer.split("EXPLANATION:")[1].strip()
    else:
        sql, explanation = answer, ""

    if sql.strip().upper() == "NONE":
        return explanation or "Bu soru için ilgili veri bulunamadı."

    is_safe, message = is_sql_safe(sql)
    if not is_safe:
        return f"⚠️ Query rejected: {message}"

    rows = run_sql(sql)
    return f"{explanation}\n\n{format_results(rows)}"


# ============================================================
# Chart answer path: user question -> chart type -> chart params -> Superset + PNG
# ============================================================

def determine_chart_type(user_question):
    """Asks the LLM to pick "bar", "line" or "pie" for the question."""
    prompt = f"""
Determine the most appropriate chart type for the user's request.
Respond ONLY with one word: "bar", "line", or "pie".

Use "line" for trends over time. Use "bar" for comparing categories.
Use "pie" for showing proportions/distribution of a whole.

User request: {user_question}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    chart_type = response.choices[0].message.content.strip().lower()
    return chart_type if chart_type in CHART_SPECS else "bar"


def determine_chart_params(user_question, chart_type, relevant_columns):
    """
    Asks the LLM to pick axis/metric/filter columns and settings for the
    given chart_type, constrained to the columns actually available.
    Returns (chart_params dict, enriched columns) -- chart_params is
    untrusted until it passes is_chart_params_safe.
    """
    if chart_type not in CHART_SPECS:
        raise ValueError(f"Bilinmeyen chart_type: {chart_type}")

    spec = CHART_SPECS[chart_type]
    enriched = enrich_columns(relevant_columns)

    columns_desc = "\n".join(
        f"- {c['source_column']} (tablo: {c['source_table']}, tip: {c['data_type']}, rol: {c['role']}) "
        f"- {c['friendly_name']}"
        for c in enriched
    )

    axis_role_tr = " veya ".join(
        "temporal (tarih/zaman)" if r == "temporal" else "categorical (kategori)"
        for r in sorted(spec["allowed_axis_roles"])
    )

    if spec.get("supports_contribution"):
        contribution_rule = (
            '- contribution_mode: Kullanıcı "yüzde", "pay", "oranı" gibi bir şey istiyorsa '
            '(örn. "kategorilerin satıştaki payı"), "share_of_total" kullan -- bu her axis '
            'değerinin toplam içindeki payını hesaplar. Kullanıcı ham sayı/toplam '
            'istiyorsa (varsayılan durum) "none" kullan.\n'
        )
        contribution_format = ',\n  "contribution_mode": "none" veya "share_of_total"'
    else:
        contribution_rule = ""
        contribution_format = ""

    prompt = f"""
Bir {chart_type} chart için parametre üret.

Kurallar:
- axis_column SADECE rolü {axis_role_tr} olan bir kolon olabilir.
- Eğer axis_column'un rolü temporal ise (tarih/zaman kolonuysa), time_grain
  ZORUNLU ve şu değerlerden biri olmalı: P1D, P1W, P1M, P3M, P1Y
  (kullanıcı "aylık" derse P1M, "haftalık" derse P1W, "yıllık" derse P1Y, vb.).
- Eğer axis_column'un rolü categorical ise, time_grain null olmalı.
- time_filter_column SADECE rolü "temporal" olan bir kolon olmalı. Bu, grafikteki
  eksenden BAĞIMSIZ bir zaman aralığı filtresi için kullanılır (genelde ana tarih
  kolonu, örn. sipariş tarihi). axis_column temporal ise time_filter_column ile
  aynı kolon olabilir.
  ÖNEMLİ: time_filter_column HER ZAMAN dolu olmalı, ASLA null olamaz --
  kullanıcı hiçbir tarih/zaman ifadesi belirtmemiş olsa bile (bu durumda
  time_range = "No filter" olur ama time_filter_column yine de rolü temporal
  olan en uygun kolona, genelde "order_date"e, eşit olmalıdır). time_range ile
  time_filter_column birbirinden BAĞIMSIZ kararlardır: biri "hangi kolona
  filtre uygulanacak" (her zaman gerekli), diğeri "o kolonda hangi aralık"
  (opsiyonel, yoksa "No filter").
- Kullanıcı satır sayısı/adet/kaç tane gibi bir şey soruyorsa:
  metric_type = "count", metric_column = null, aggregate = null
- Kullanıcı bir sayısal kolonun toplamı/ortalaması/vb. istiyorsa:
  metric_type = "column", metric_column rolü "numeric" olan bir kolon,
  aggregate SUM/AVG/COUNT/MIN/MAX içinden biri.
- sort_ascending: axis_column temporal ise kronolojik yön (true = eskiden yeniye);
  axis_column categorical ise metrik büyüklüğüne göre yön (true = küçükten büyüğe,
  false = büyükten küçüğe, örn. "en çok satan kategori önce" için false).
- time_range: Kullanıcı isteğinde AÇIKÇA bir yıl, çeyrek veya tarih aralığı
  geçiyorsa (örn. "2018'de", "2017 ile 2018 arasında", "2018 Q1"), bunu
  "YYYY-MM-DD : YYYY-MM-DD" formatında bir aralığa çevir (üst sınır dahil değil,
  örn. "2018'de" -> "2018-01-01 : 2019-01-01"). Kullanıcı hiçbir zaman ifadesi
  belirtmemişse (örn. sadece "kategoriye göre satış") time_range = "No filter".
  Bu alan time_filter_column'a uygulanacak filtre aralığıdır, axis_column'dan
  BAĞIMSIZDIR.
  Kullanıcı SADECE alt sınır belirtiyorsa (örn. "2017'den itibaren",
  "2017 sonrasında", "2017'den beri"), üst sınırı boş bırak:
  "2017-01-01 : " (sonunda boşluk olacak şekilde, iki nokta sonrası boş).
  Kullanıcı SADECE üst sınır belirtiyorsa (örn. "2018'den önce",
  "2018'e kadar"), alt sınırı boş bırak: " : 2018-01-01" (başında boşluk,
  iki nokta öncesi boş).
- category_filter_column / category_filter_operator / category_filter_value:
  Kullanıcı belirli bir kategoriye/değere daraltmak veya sayısal bir eşiğe göre
  filtrelemek istiyorsa (örn. "health_beauty kategorisinin aylara göre satışı",
  "2017'den sonraki siparişler", "2018 çeyrek 3'ten önce"), şunları belirle:
  - category_filter_column: rolü "categorical" olan ilgili kolon
  - category_filter_operator: "==", "!=", ">", "<", ">=", "<=" içinden biri
    (eşitlik/dışlama için "==" veya "!=", sayısal eşik için diğerleri)
  - category_filter_value: kullanıcının bahsettiği değer. Metinsel bir kategori
    ise string (örn. "health_beauty"), sayısal bir kolon (örn. year, quarter)
    ise sayı (örn. 2017) olarak ver.
  axis_column ile AYNI kolon olabilir (örn. axis kategoriye göre kırıyor, filtre
  de tek bir kategoriye daraltıyor) -- bu geçerlidir. Kullanıcı böyle bir
  daraltma istemiyorsa üçü de null olmalı.
- metric_filter_operator / metric_filter_value: Kullanıcı TOPLAM/ORTALAMA/SAYI
  gibi agregasyon SONRASI bir eşik istiyorsa (örn. "toplam satışı 10000'i geçen
  kategoriler", "ortalama ödemesi 50'nin altında olan müşteriler", "satılan
  ürün sayısı 10000'i geçen kategoriler"), bu category_filter'dan FARKLIDIR --
  burada eşik tek bir satıra değil, GRUPLANMIŞ metriğin kendisine uygulanır.
  metric_type "column" (SUM/AVG/vb.) veya "count" iken kullanılabilir.
  - metric_filter_operator: ">", "<", ">=", "<=", "==", "!=" içinden biri
  - metric_filter_value: sayısal eşik değeri (örn. 10000)
  Kullanıcı böyle bir eşik istemiyorsa ikisi de null olmalı.
{contribution_rule}- Cevabında SADECE JSON olsun, markdown code block kullanma, açıklama ekleme.

Kolonlar:
{columns_desc}

Format:
{{
  "axis_column": "kolon_adi",
  "time_filter_column": "kolon_adi",
  "time_grain": "P1M" veya null,
  "metric_type": "count",
  "metric_column": null,
  "aggregate": null,
  "sort_ascending": false,
  "time_range": "No filter" veya "YYYY-MM-DD : YYYY-MM-DD",
  "category_filter_column": null veya "kolon_adi",
  "category_filter_operator": null veya "==",
  "category_filter_value": null veya "deger",
  "metric_filter_operator": null veya ">",
  "metric_filter_value": null veya 10000{contribution_format}
}}

Kullanıcı isteği: {user_question}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    chart_params = json.loads(raw)
    return chart_params, enriched


def is_chart_params_safe(chart_type, chart_params, enriched_columns):
    """
    Validates every field the LLM produced in determine_chart_params against
    the actual column roles/types and the fixed whitelists above. Nothing
    from chart_params reaches Superset or SQL generation before passing this.
    """
    spec = CHART_SPECS[chart_type]
    by_name = {c["source_column"]: c for c in enriched_columns}

    axis_col = chart_params.get("axis_column")
    if axis_col not in by_name:
        return False, f"axis_column geçersiz kolon: {axis_col}"
    axis_role = by_name[axis_col]["role"]
    if axis_role not in spec["allowed_axis_roles"]:
        return False, (
            f"axis_column rolü {sorted(spec['allowed_axis_roles'])} içinden biri olmalı, "
            f"'{axis_col}' rolü: {axis_role}"
        )

    if axis_role == "temporal":
        if chart_params.get("time_grain") not in VALID_TIME_GRAINS:
            return False, f"time_grain geçersiz: {chart_params.get('time_grain')}"

    time_filter_col = chart_params.get("time_filter_column")
    if time_filter_col not in by_name:
        return False, f"time_filter_column geçersiz kolon: {time_filter_col}"
    if by_name[time_filter_col]["role"] != "temporal":
        return False, f"time_filter_column temporal olmalı: {time_filter_col}"

    metric_type = chart_params.get("metric_type")
    if metric_type == "column":
        metric_col = chart_params.get("metric_column")
        if metric_col not in by_name:
            return False, f"metric_column geçersiz kolon: {metric_col}"
        if by_name[metric_col]["role"] != "numeric":
            return False, f"metric_column numeric olmalı: {metric_col}"
        if chart_params.get("aggregate") not in VALID_AGGREGATES:
            return False, f"aggregate geçersiz: {chart_params.get('aggregate')}"
    elif metric_type != "count":
        return False, f"metric_type 'count' ya da 'column' olmalı, geldi: {metric_type}"

    time_range = chart_params.get("time_range", "No filter")
    if time_range != "No filter" and not is_valid_time_range(time_range):
        return False, f"time_range geçersiz format: {time_range}"

    category_filter_col = chart_params.get("category_filter_column")
    category_filter_op = chart_params.get("category_filter_operator")
    category_filter_val = chart_params.get("category_filter_value")
    if category_filter_col is not None or category_filter_val is not None:
        if category_filter_col not in by_name:
            return False, f"category_filter_column geçersiz kolon: {category_filter_col}"
        if by_name[category_filter_col]["role"] != "categorical":
            return False, f"category_filter_column categorical olmalı: {category_filter_col}"

        if category_filter_op not in VALID_FILTER_OPERATORS:
            return False, f"category_filter_operator geçersiz: {category_filter_op}"

        # Inequality operators only make sense on numeric-looking categorical
        # columns (year, quarter, ...), not on free-text categories.
        if category_filter_op not in ("==", "!=") and category_filter_col not in CATEGORICAL_INT_COLUMNS:
            return False, (
                f"'{category_filter_op}' operatörü sadece sayısal kolonlarda "
                f"({sorted(CATEGORICAL_INT_COLUMNS)}) kullanılabilir, '{category_filter_col}' değil"
            )

        if isinstance(category_filter_val, str):
            if not category_filter_val.strip():
                return False, "category_filter_value boş olamaz"
            # Superset builds the filter parametrically, but reject anything
            # SQL-injection-shaped anyway as defense in depth.
            if re.search(r"[;'\"]|--|\b(DROP|DELETE|INSERT|UPDATE|UNION|SELECT)\b",
                          category_filter_val, re.IGNORECASE):
                return False, f"category_filter_value şüpheli içerik barındırıyor: {category_filter_val}"

            # LLM sometimes returns a numeric categorical value (e.g. year)
            # as a quoted string -- coerce it to int so Superset doesn't get
            # a type mismatch.
            if category_filter_col in CATEGORICAL_INT_COLUMNS:
                try:
                    category_filter_val = int(category_filter_val)
                except ValueError:
                    return False, (
                        f"category_filter_value sayısal olmalı ({category_filter_col} "
                        f"integer bir kolon): {category_filter_val!r}"
                    )
                chart_params["category_filter_value"] = category_filter_val
        elif isinstance(category_filter_val, (int, float)):
            if category_filter_col not in CATEGORICAL_INT_COLUMNS:
                return False, f"category_filter_value sayısal ama kolon sayısal değil: {category_filter_col}"
        else:
            return False, f"category_filter_value geçersiz tip: {type(category_filter_val)}"

    # Post-aggregation (HAVING) threshold, e.g. "categories with total sales
    # over 10000". Only operator + value are taken from the LLM -- the actual
    # SQL expression is built later from already-whitelisted aggregate/column
    # values, never from free text.
    metric_filter_op = chart_params.get("metric_filter_operator")
    metric_filter_val = chart_params.get("metric_filter_value")
    if metric_filter_op is not None or metric_filter_val is not None:
        if metric_filter_op not in VALID_FILTER_OPERATORS:
            return False, f"metric_filter_operator geçersiz: {metric_filter_op}"
        if not isinstance(metric_filter_val, (int, float)) or isinstance(metric_filter_val, bool):
            return False, f"metric_filter_value sayısal olmalı: {metric_filter_val!r}"
        if chart_params.get("metric_type") not in ("column", "count"):
            return False, "metric_filter sadece metric_type='column' veya 'count' iken kullanılabilir"

    if spec.get("supports_contribution"):
        contribution_mode = chart_params.get("contribution_mode", "none")
        if contribution_mode not in ("none", "share_of_total"):
            return False, f"contribution_mode geçersiz: {contribution_mode}"

    return True, "Safe."


# ============================================================
# Dataset resolution
# ============================================================

def resolve_dataset_id(chart_params, enriched_columns, default_dataset_id=DATASET_ID):
    """
    Figures out which Superset dataset_id the chart's columns belong to,
    using the optional dataset_id column in data_catalog. Falls back to
    default_dataset_id if none of the used columns specify one, and raises
    if the used columns span more than one dataset (can't combine them into
    a single chart).
    """
    used_cols = {chart_params.get("axis_column"), chart_params.get("time_filter_column")}
    if chart_params.get("metric_type") == "column":
        used_cols.add(chart_params.get("metric_column"))
    if chart_params.get("category_filter_column"):
        used_cols.add(chart_params["category_filter_column"])
    used_cols.discard(None)

    by_name = {c["source_column"]: c for c in enriched_columns}
    dataset_ids = {
        by_name[col]["dataset_id"]
        for col in used_cols
        if col in by_name and by_name[col].get("dataset_id") is not None
    }

    if len(dataset_ids) == 0:
        return default_dataset_id
    if len(dataset_ids) == 1:
        return dataset_ids.pop()
    raise ValueError(
        f"Bu grafik farklı dataset'lere ait kolonları birlikte kullanıyor "
        f"({sorted(dataset_ids)}), bu desteklenmiyor. Soruyu ayrı ayrı sormayı deneyin."
    )


# ============================================================
# Superset chart payload
# ============================================================

def build_metric(chart_params):
    if chart_params["metric_type"] == "count":
        return "count"
    return {
        "expressionType": "SIMPLE",
        "column": {"column_name": chart_params["metric_column"]},
        "aggregate": chart_params["aggregate"],
        "label": f"{chart_params['aggregate']}({chart_params['metric_column']})",
    }


def get_axis_info(axis_col, enriched_columns):
    for c in enriched_columns:
        if c["source_column"] == axis_col:
            return c
    return None


def build_chart_payload(chart_type, chart_params, enriched_columns, dataset_id, chart_name):
    """
    Builds the Superset /api/v1/chart/ request body for line/bar/pie from
    already-validated chart_params. Cosmetic settings (colors, legend, ...)
    are fixed sane defaults here, not something the LLM decides.
    """
    spec = CHART_SPECS[chart_type]
    metric = build_metric(chart_params)
    axis_col = chart_params["axis_column"]
    axis_info = get_axis_info(axis_col, enriched_columns)
    axis_role = axis_info["role"]
    time_filter_col = chart_params["time_filter_column"]
    time_range = chart_params.get("time_range", "No filter")

    adhoc_filters = [
        {
            "clause": "WHERE",
            "subject": time_filter_col,
            "operator": "TEMPORAL_RANGE",
            "comparator": time_range,
            "expressionType": "SIMPLE",
        }
    ]

    category_filter_col = chart_params.get("category_filter_column")
    category_filter_op = chart_params.get("category_filter_operator")
    category_filter_val = chart_params.get("category_filter_value")
    if category_filter_col and category_filter_val is not None:
        adhoc_filters.append({
            "clause": "WHERE",
            "subject": category_filter_col,
            "operator": category_filter_op,
            "comparator": category_filter_val,
            "expressionType": "SIMPLE",
        })

    # Superset's Simple filter mode has no HAVING option, so a post-aggregation
    # threshold has to be sent as a raw SQL expression. It's built here from
    # already-whitelisted aggregate/metric_column values only.
    metric_filter_op = chart_params.get("metric_filter_operator")
    metric_filter_val = chart_params.get("metric_filter_value")
    if metric_filter_op and metric_filter_val is not None:
        if chart_params["metric_type"] == "count":
            having_expr = f"COUNT(*) {metric_filter_op} {metric_filter_val}"
        else:
            having_expr = (
                f"{chart_params['aggregate']}({chart_params['metric_column']}) "
                f"{metric_filter_op} {metric_filter_val}"
            )
        adhoc_filters.append({
            "expressionType": "SQL",
            "sqlExpression": having_expr,
            "clause": "HAVING",
            "subject": None,
            "operator": None,
            "comparator": None,
        })

    params = {
        "datasource": f"{dataset_id}__table",
        "viz_type": spec["viz_type"],
        "adhoc_filters": adhoc_filters,
        "row_limit": 10000 if chart_type in ("line", "bar") else 100,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "legendType": "scroll",
        "legendOrientation": "top",
    }

    if chart_type in ("line", "bar"):
        params.update({
            "x_axis": axis_col,
            "metrics": [metric],
            "groupby": [],
            "truncate_metric": True,
            "show_empty_columns": True,
            "comparison_type": "values",
            "x_axis_time_format": "smart_date",
            "x_axis_number_format": "~g",
            "y_axis_format": "SMART_NUMBER",
            "rich_tooltip": True,
            "tooltipTimeFormat": "smart_date",
            "x_axis_sort_asc": chart_params.get("sort_ascending", False),
            "order_desc": not chart_params.get("sort_ascending", False),
        })
        if axis_role == "temporal":
            params["time_grain_sqla"] = chart_params["time_grain"]

        # If axis_column is numeric in the DB but categorical in our
        # classification (e.g. dim_date.quarter), force Superset to draw it
        # as discrete categories instead of a continuous numeric axis.
        if axis_role == "categorical" and axis_info["data_type"] in NUMERIC_SQL_TYPES:
            params["xAxisForceCategorical"] = True

        if chart_type == "bar":
            params["orientation"] = "vertical"
            # Sorting by metric size only makes sense for a categorical axis;
            # a temporal axis must stay chronological.
            if axis_role == "categorical":
                sort_metric_label = "count" if metric == "count" else metric["label"]
                params["x_axis_sort"] = sort_metric_label

        # contributionMode="column" is correct as long as groupby stays
        # empty (single-column pivot) -- if multi-dimension grouping is
        # added later this needs to be revisited.
        if spec.get("supports_contribution") and chart_params.get("contribution_mode") == "share_of_total":
            if params["groupby"]:
                raise ValueError(
                    "contribution_mode='share_of_total' için 'column' varsayımı "
                    "sadece groupby boşken geçerlidir. groupby dolu -- bu mantık "
                    "gözden geçirilmeden devam edilemez."
                )
            params["contributionMode"] = "column"

    else:  # pie
        params.update({
            "groupby": [axis_col],
            "metric": metric,
            "sort_by_metric": True,
            "label_type": "key",
            "number_format": "SMART_NUMBER",
            "date_format": "smart_date",
            "show_labels": True,
            "labels_outside": True,
            "label_line": False,
            "show_total": False,
            "outerRadius": 70,
            "donut": False,
            "innerRadius": 30,
        })

    return {
        "slice_name": chart_name,
        "viz_type": spec["viz_type"],
        "datasource_id": dataset_id,
        "datasource_type": "table",
        "params": json.dumps(params)
    }


def superset_login():
    session = requests.Session()
    login_payload = {"username": SUPERSET_USERNAME, "password": SUPERSET_PASSWORD, "provider": "db"}
    response = session.post(f"{SUPERSET_URL}/api/v1/security/login", json=login_payload)
    session.headers.update({"Authorization": f"Bearer {response.json()['access_token']}"})

    csrf_response = session.get(f"{SUPERSET_URL}/api/v1/security/csrf_token/")
    session.headers.update({"X-CSRFToken": csrf_response.json()["result"]})
    return session


# ============================================================
# Local (Superset-independent) chart rendering
# ============================================================
#
# Superset resolves joins internally via its virtual dataset, which Python
# has no direct access to. To render a chart even when Superset is
# unreachable, the same validated chart_params are turned into a plain SQL
# query (via the LLM, but with no room left to reinterpret anything -- it
# only translates already-fixed parameters into syntax) and run directly
# against PostgreSQL. The result is drawn locally with matplotlib.

def build_chart_query_sql(chart_params, columns_summary, joins_summary):
    prompt = f"""
Aşağıda ZATEN belirlenmiş (kullanıcıdan değil, önceden doğrulanmış sabit
parametrelerden) bir grafik sorgusu tanımı var. Bunu SADECE bir PostgreSQL
SELECT sorgusuna çevir -- yorum ekleme, farklı bir yorum getirme, sadece
verilen parametreleri sözdizimine dök.

axis_column: {chart_params['axis_column']}
time_grain: {chart_params.get('time_grain')}
metric_type: {chart_params['metric_type']}
metric_column: {chart_params.get('metric_column')}
aggregate: {chart_params.get('aggregate')}
time_filter_column: {chart_params['time_filter_column']}
time_range: {chart_params.get('time_range')}
category_filter_column: {chart_params.get('category_filter_column')}
category_filter_operator: {chart_params.get('category_filter_operator')}
category_filter_value: {chart_params.get('category_filter_value')}
metric_filter_operator: {chart_params.get('metric_filter_operator')}
metric_filter_value: {chart_params.get('metric_filter_value')}
sort_ascending: {chart_params.get('sort_ascending')}

Kurallar:
- SELECT listesi: axis_column (temporal ve time_grain doluysa
  DATE_TRUNC('gün'/'hafta'/'ay'/'çeyrek'/'yıl', axis_column) kullan;
  P1D->gün, P1W->hafta, P1M->ay, P3M->çeyrek, P1Y->yıl), ve metriği
  (metric_type "count" ise COUNT(*), "column" ise aggregate(metric_column)).
- GROUP BY axis_column (ya da DATE_TRUNC ifadesi neyse).
- time_range "No filter" değilse, time_filter_column'a WHERE ile aralık
  filtresi uygula. Format "START : END" -- START veya END boş olabilir,
  boş olan taraf sınırsızdır (örn. "2017-01-01 : " => sadece
  time_filter_column >= '2017-01-01').
- category_filter_column doluysa WHERE'e category_filter_operator ve
  category_filter_value ile ekle.
- metric_filter_operator doluysa HAVING'e metrik ifadesiyle ekle.
- sort_ascending'e göre ORDER BY (axis temporal ise axis'e göre kronolojik,
  değilse metriğe göre) ASC/DESC belirle.
- SADECE SELECT sorgusu yaz, sadece aşağıdaki tabloları/join'leri kullan.
- Cevabında SADECE SQL olsun, açıklama/markdown code block ekleme.

Kolonlar:
{columns_summary}

Join info:
{joins_summary}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    sql = response.choices[0].message.content.strip()
    return sql.replace("```sql", "").replace("```", "").strip()


def run_chart_query(chart_params, relevant_columns):
    """
    Expects chart_params to have already passed is_chart_params_safe.
    The SQL generated from it is checked again with is_sql_safe as a second
    line of defense.
    """
    columns_summary, joins_summary = build_schema_context(relevant_columns)
    sql = build_chart_query_sql(chart_params, columns_summary, joins_summary)

    is_safe, message = is_sql_safe(sql)
    if not is_safe:
        raise ValueError(f"Üretilen chart SQL'i reddedildi: {message}\nSQL: {sql}")

    rows = run_sql(sql)
    return rows, sql


def render_chart_png(chart_type, rows, axis_label, metric_label):
    """Renders SQL result rows [(axis_value, metric_value), ...] to a PNG buffer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    labels = [str(r[0]) for r in rows]
    values = [float(r[1]) if r[1] is not None else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%")
    else:
        if chart_type == "line":
            ax.plot(labels, values, marker="o")
        else:  # bar
            ax.bar(labels, values)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(metric_label)
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def answer_with_chart(user_question, relevant_columns):
    """CLI-facing chart flow: returns a plain text answer with the Superset link."""
    chart_type = determine_chart_type(user_question)
    chart_params, enriched = determine_chart_params(user_question, chart_type, relevant_columns)

    is_safe, message = is_chart_params_safe(chart_type, chart_params, enriched)
    if not is_safe:
        return f"⚠️ Chart rejected: {message}"

    try:
        dataset_id = resolve_dataset_id(chart_params, enriched)
    except ValueError as e:
        return f"⚠️ {e}"

    session = superset_login()
    payload = build_chart_payload(chart_type, chart_params, enriched, dataset_id, user_question[:100])
    response = session.post(f"{SUPERSET_URL}/api/v1/chart/", json=payload)
    result = response.json()

    if "id" in result:
        return f"Chart oluşturuldu: {SUPERSET_PUBLIC_URL}/explore/?slice_id={result['id']}"
    return f"⚠️ Chart oluşturulamadı: {result}"


def answer_with_chart_full(user_question, relevant_columns):
    """
    Streamlit-facing version of answer_with_chart: instead of one text
    string, returns a dict with every piece the UI needs separately
    (error / Superset link / local PNG buffer / SQL / warning).
    Saving to Superset is optional -- if it's unreachable, the local PNG is
    still attempted so the user can see and download the chart regardless.
    """
    chart_type = determine_chart_type(user_question)
    chart_params, enriched = determine_chart_params(user_question, chart_type, relevant_columns)

    is_safe, message = is_chart_params_safe(chart_type, chart_params, enriched)
    if not is_safe:
        return {"error": message}

    result = {
        "chart_type": chart_type,
        "chart_params": chart_params,
        "superset_url": None,
        "png_buf": None,
        "sql": None,
        "warning": None,
    }

    try:
        dataset_id = resolve_dataset_id(chart_params, enriched)
        session = superset_login()
        payload = build_chart_payload(chart_type, chart_params, enriched, dataset_id, user_question[:100])
        response = session.post(f"{SUPERSET_URL}/api/v1/chart/", json=payload)
        superset_result = response.json()
        if "id" in superset_result:
            result["superset_url"] = f"{SUPERSET_PUBLIC_URL}/explore/?slice_id={superset_result['id']}"
        else:
            result["warning"] = f"Superset chart'ı oluşturulamadı: {superset_result}"
    except Exception as e:
        result["warning"] = f"Superset'e ulaşılamadı ({e}) -- sadece yerel grafik gösteriliyor."

    try:
        rows, sql = run_chart_query(chart_params, relevant_columns)
        result["sql"] = sql
        if not rows:
            result["warning"] = (result["warning"] or "") + " Sorgu sonucu boş döndü."
        else:
            axis_label = chart_params["axis_column"]
            metric_label = (
                "count" if chart_params["metric_type"] == "count"
                else f"{chart_params['aggregate']}({chart_params['metric_column']})"
            )
            result["png_buf"] = render_chart_png(chart_type, rows, axis_label, metric_label)
    except Exception as e:
        result["error_local_render"] = str(e)

    return result


# ============================================================
# Main entry point (CLI)
# ============================================================

def answer_question(user_question, want_chart):
    catalog = fetch_data_catalog()
    relevant_columns = get_all_columns(catalog)
    print(f"Kataloğun tamamı kullanılıyor ({len(relevant_columns)} kolon):")
    for c in relevant_columns:
        print(f"{c['source_column']} - {c['friendly_name']}")

    if want_chart:
        return answer_with_chart(user_question, relevant_columns)
    return answer_with_sql(user_question, relevant_columns)


if __name__ == "__main__":
    user_question = input("Sorunuzu yazın: ")
    chart_input = input("Chart (y/n): ").strip().lower()
    want_chart = chart_input == "y"

    print()
    print(answer_question(user_question, want_chart))
