import os
import re
import json
import requests
import psycopg2
from decimal import Decimal
from contextlib import contextmanager
from openai import OpenAI
# sentence_transformers artık default akışta kullanılmıyor (bkz. find_relevant_columns notu).
# Tekrar açmak istersen: from sentence_transformers import SentenceTransformer, util


# ---- Configuration ----
# Ortam değişkeni verilmezse eski (yerel/localhost) davranış aynen devam
# eder -- bu sayede Docker içinde çalışırken (localhost artık container'ın
# kendisini işaret ettiği için) sadece env var vermek yeterli, kod
# değişmiyor.

DB_CONFIG = {
    "host": os.environ.get("DWH_DB_HOST"),
    "port": int(os.environ.get("DWH_DB_PORT")),
    "dbname": os.environ.get("DWH_DB_NAME"),
    "user": os.environ.get("DWH_DB_USER"),
    "password": os.environ.get("DWH_DB_PASSWORD"),
}

SUPERSET_URL = os.environ.get("SUPERSET_URL", "http://localhost:8088")
SUPERSET_USERNAME = os.environ.get("SUPERSET_USERNAME")
SUPERSET_PASSWORD = os.environ.get("SUPERSET_PASSWORD")
DATASET_ID = int(os.environ.get("SUPERSET_DATASET_ID"))

ALLOWED_TABLES = {
    "dim_customer", "dim_product", "dim_seller", "dim_date",
    "fact_order_items", "fact_order_payments"
}

# ---- Chart type spec'leri: her tipin gerçek Superset şemasına göre neye
# ihtiyacı olduğunu tanımlar (bkz. line/bar/pie export JSON'ları) ----

CHART_SPECS = {
    "line": {
        "viz_type": "echarts_timeseries_line",
        "allowed_axis_roles": {"temporal"},   # line'da eksen sadece zaman olabilir
        "metric_as_list": True,    # payload'da metrics: [metric]
        "supports_contribution": True,
    },
    "bar": {
        "viz_type": "echarts_timeseries_bar",
        # bar hem kategori karşılaştırması ("kategoriye göre satış") hem de
        # zaman bazlı gruplama ("aylık sipariş sayısı") için kullanılabilir
        "allowed_axis_roles": {"categorical", "temporal"},
        "metric_as_list": True,
        "supports_contribution": True,
    },
    "pie": {
        "viz_type": "pie",
        "allowed_axis_roles": {"categorical"},
        "metric_as_list": False,   # payload'da tekil metric
        "supports_contribution": False,  # pie zaten doğası gereği oran gösterir
    },
}

VALID_AGGREGATES = {"SUM", "AVG", "COUNT", "MIN", "MAX"}
# category_filter_operator için whitelist. Superset'in adhoc_filters SIMPLE
# clause'unda gerçekten bu düz string'lerle çalıştığı F12 Network export'u ile
# doğrulandı (operatorId alanı frontend state'i için, backend zorunlu saymıyor).
VALID_FILTER_OPERATORS = {"==", "!=", ">", "<", ">=", "<="}
VALID_TIME_GRAINS = {"P1D", "P1W", "P1M", "P3M", "P1Y"}

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
# embedding_model kaldırıldı: find_relevant_columns şu an kullanılmıyor.


# ---- Database helpers ----

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
    with get_db_cursor() as cursor:
        cursor.execute(
            "SELECT friendly_name, source_table, source_column, description, join_info, dataset_id FROM data_catalog;"
        )
        return cursor.fetchall()


def run_sql(sql):
    with get_db_cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


# ---- Schema retrieval ----

def get_all_columns(catalog):
    """
    Embedding/cosine-similarity filtresi olmadan, data_catalog'daki TÜM satırları
    relevant_columns ile aynı sözlük şekline çevirir. Şu an default akış bu.

    Neden: threshold=0 + top_n=50 zaten kataloğun neredeyse tamamını LLM'e
    gönderiyordu, yani eski find_relevant_columns fiilen filtre değil sadece
    (kalitesiz) bir sıralama yapıyordu. Küçük bir katalogda (şu an ~43 satır)
    doğrudan tamamını vermek daha basit ve en az o kadar isabetli.
    """
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


def find_relevant_columns(user_question, catalog, threshold=0, top_n=50):
    """
    DEVRE DIŞI (default akışta çağrılmıyor). Katalog büyüyüp tamamını LLM'e
    vermek verimsizleştiğinde tekrar açmak için duruyor.

    Açmak için: dosyanın başındaki sentence_transformers importunu ve
    embedding_model satırını geri aç, answer_question içinde
    get_all_columns(catalog) çağrısını bununla değiştir.
    """
    from sentence_transformers import SentenceTransformer, util
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embed_texts = [f"{row[0]}: {row[3]}" for row in catalog]  # row[3] = description
    catalog_embeddings = embedding_model.encode(embed_texts, convert_to_tensor=True)
    question_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, catalog_embeddings)[0]

    results = []
    for idx, score in enumerate(similarities):
        if score.item() >= threshold:
            row = catalog[idx]
            results.append({
                "friendly_name": row[0],
                "source_table": row[1],
                "source_column": row[2],
                "description": row[3],
                "join_info": row[4],
                "dataset_id": row[5],
                "score": round(score.item(), 3)
            })
    results.sort(key=lambda x: -x["score"])
    return results[:top_n]


def build_schema_context(relevant_columns):
    column_lines = [
        f"- Table: {c['source_table']} | Column: {c['source_column']} | Meaning: {c['friendly_name']} | {c['description']}"
        for c in relevant_columns
    ]
    join_lines = {c["join_info"] for c in relevant_columns if c["join_info"]}
    return "\n".join(column_lines), "\n".join(join_lines) if join_lines else "None"


# ---- Chart için kolon tipi zenginleştirme (information_schema tabanlı) ----

def get_column_types(relevant_columns):
    """
    (table, column) -> gerçek PostgreSQL data_type.
    LLM'e isimden tip tahmin ettirmek yerine gerçek şemayı veriyoruz.
    """
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

# Bazı kolonlar veritabanında integer/smallint olarak saklanır ama anlamsal
# olarak toplanacak/ortalanacak bir DEĞER değil, bir GRUPLAMA/KATEGORİdir
# (örn. dim_date.quarter: 1/2/3/4 -- SUM(quarter) anlamsız, ama "çeyreğe göre
# satış" gibi bir axis olarak kullanılabilir). Bunları classify_column'da
# "numeric" yerine "categorical" sayıyoruz.
CATEGORICAL_INT_COLUMNS = {"year", "quarter", "month", "day", "day_of_week"}


def classify_column(data_type, column_name=None):
    if data_type in ("date", "timestamp", "timestamp without time zone", "timestamptz"):
        return "temporal"
    if column_name in CATEGORICAL_INT_COLUMNS:
        return "categorical"
    if data_type in NUMERIC_SQL_TYPES:
        return "numeric"
    return "categorical"


def enrich_columns(relevant_columns):
    type_map = get_column_types(relevant_columns)
    enriched = []
    for c in relevant_columns:
        data_type = type_map.get((c["source_table"], c["source_column"]), "unknown")
        role = classify_column(data_type, c["source_column"])
        enriched.append({**c, "data_type": data_type, "role": role})
    return enriched


# ---- Time range validation ----

def is_valid_time_range(time_range):
    """
    Kabul edilen format: "START : END", START/END her biri boş ya da
    YYYY-MM-DD olabilir (ikisi birden boş olamaz). Superset'in
    TEMPORAL_RANGE comparator'ının gerçek gösterim şekli bu (F12 ile
    doğrulandı: kapalı "2017-01-01 : 2018-01-01", sadece-alt
    "2017-01-01 : ", sadece-üst " : 2018-01-01").
    """
    if " : " not in time_range:
        return False
    start, end = time_range.split(" : ", 1)
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    start_ok = start == "" or date_pattern.match(start)
    end_ok = end == "" or date_pattern.match(end)
    return bool(start_ok and end_ok and (start != "" or end != ""))


# ---- SQL safety ----

def is_select_only(sql):
    return sql.strip().upper().startswith("SELECT")


def extract_table_names(sql):
    pattern = r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    return {m.lower() for m in re.findall(pattern, sql, re.IGNORECASE)}


def is_sql_safe(sql, allowed_tables=ALLOWED_TABLES):
    if not is_select_only(sql):
        return False, "Only SELECT queries are allowed."
    disallowed = extract_table_names(sql) - allowed_tables
    if disallowed:
        return False, f"Disallowed table(s) used: {', '.join(disallowed)}"
    return True, "Safe."


# ---- Result formatting ----

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


# ---- SQL answer path ----

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
        temperature=0.1,  # yapısal/deterministik çıktı için -- rastgelelik azaltılıyor
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


# ---- Chart answer path ----

def determine_chart_type(user_question):
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
    Tek, chart_type'a göre dallanan parametre üretici (line/bar/pie hepsi burada).
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

    # time_grain, chart tipine değil axis_column'un GERÇEK rolüne bağlı:
    # bar'da axis temporal seçilmişse (örn. "aylık sipariş sayısı") zorunlu,
    # axis categorical seçilmişse (örn. "kategoriye göre satış") kullanılmaz.
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

    # Kategorik/sayısal filtre (örn. "health_beauty kategorisi",
    # "year > 2017") -- opsiyonel.
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

        # inequality operatörleri sadece sayısal-ama-categorical kolonlarda
        # anlamlı (örn. year, quarter). Metinsel kategori kolonunda ">"
        # anlamsız/hatalı olur.
        if category_filter_op not in ("==", "!=") and category_filter_col not in CATEGORICAL_INT_COLUMNS:
            return False, (
                f"'{category_filter_op}' operatörü sadece sayısal kolonlarda "
                f"({sorted(CATEGORICAL_INT_COLUMNS)}) kullanılabilir, '{category_filter_col}' değil"
            )

        if isinstance(category_filter_val, str):
            if not category_filter_val.strip():
                return False, "category_filter_value boş olamaz"
            # Superset filtreyi parametreli olarak kendi kuruyor (raw SQL
            # string birleştirme değil), yine de savunma amaçlı basit bir
            # sağlamlık kontrolü: tırnak/noktalı virgül/SQL anahtar kelimesi
            # içermesin.
            if re.search(r"[;'\"]|--|\b(DROP|DELETE|INSERT|UPDATE|UNION|SELECT)\b",
                          category_filter_val, re.IGNORECASE):
                return False, f"category_filter_value şüpheli içerik barındırıyor: {category_filter_val}"

            # DB'de gerçekten integer olan (CATEGORICAL_INT_COLUMNS) bir
            # kolon için LLM'in değeri "2017" gibi tırnaklı STRING dönmesi
            # mümkün (JSON'da hangi tipte üreteceği garanti değil). Bu
            # durumda değeri buradan itibaren gerçek int'e çeviriyoruz --
            # aksi halde Superset'e string "2017" gider, year kolonu
            # integer olduğu için sessiz tip uyuşmazlığı/yanlış filtre
            # riski oluşur. Sayıya çevrilemiyorsa (örn. "twenty-seventeen")
            # açıkça reddediyoruz.
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

    # Agregasyon SONRASI (HAVING) filtre -- örn. "toplam satışı 10000'i geçen
    # kategoriler". category_filter'dan farklı olarak burada eşik gruplanmış
    # metriğe uygulanır. Superset'in HAVING alanı ham SQL (sqlExpression)
    # bekliyor (F12 ile doğrulandı: Simple modda HAVING seçeneği bile yok,
    # Superset kendisi "Custom SQL" kullanmayı dayatıyor) -- bu yüzden LLM'e
    # asla sqlExpression'ı kendisi yazdırmıyoruz, sadece operator + value
    # alıyoruz ve ifadeyi build_chart_payload'da zaten whitelist'ten geçmiş
    # aggregate/metric_column değerlerinden biz kendimiz inşa ediyoruz.
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


# ---- Dataset resolution ----
# data_catalog'daki her satır opsiyonel olarak bir Superset dataset_id'sine
# bağlanabilir (örn. item-level fact_order_items -> dataset 22, order-level
# aggregate fact_order_payments -> dataset 23). Bu ALAN OPSİYONEL: tek
# dataset kullanan biri hiç doldurmaz, hepsi NULL kalır ve sistem eskisi
# gibi tek DATASET_ID (.env) üzerinden çalışmaya devam eder.

def resolve_dataset_id(chart_params, enriched_columns, default_dataset_id=DATASET_ID):
    """
    chart_params içinde kullanılan kolonların (axis, metric, filtreler)
    hangi Superset dataset_id'sine ait olduğunu data_catalog'dan bulur.

    - Hiçbir kolon dataset_id taşımıyorsa (hepsi NULL) -> default_dataset_id.
    - Kullanılan kolonlar tek/aynı bir dataset_id'ye aitse -> onu döner.
    - Farklı dataset_id'ler karışmışsa -> ValueError fırlatır (LLM tek
      chart'ta birleştirilemeyecek granülaritedeki kolonları karıştırmış
      demektir, bunu sessizce yanlış sonuç üretmek yerine reddediyoruz).
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
    Tek, chart_type'a göre dallanan payload üretici (line/bar/pie hepsi burada).
    Kozmetik/stil alanları (renk, legend, donut ayarları vb.) LLM'e sordurulmuyor,
    burada sabit makul defaultlarla kuruluyor.
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

    # HAVING (agregasyon sonrası metrik eşiği), örn. "SUM(price) > 10000".
    # Superset'in HAVING alanı Simple modda bile yok, sadece ham SQL
    # (sqlExpression) kabul ediyor (F12 ile doğrulandı) -- bu yüzden
    # sqlExpression'ı LLM'den gelen serbest metinle DEĞİL, zaten
    # is_chart_params_safe'te whitelist'ten geçmiş aggregate/metric_column
    # değerlerinden burada kendimiz inşa ediyoruz. metric_filter_operator
    # VALID_FILTER_OPERATORS'tan, metric_filter_value ise sayısal olduğu
    # doğrulanmış -- yani bu string'e LLM'in ürettiği hiçbir serbest metin
    # karışmıyor.
    metric_filter_op = chart_params.get("metric_filter_operator")
    metric_filter_val = chart_params.get("metric_filter_value")
    if metric_filter_op and metric_filter_val is not None:
        if chart_params["metric_type"] == "count":
            # count metriğinde aggregate/metric_column None'dır (spec gereği),
            # bu yüzden standart SQL COUNT(*) ifadesini kullanıyoruz.
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
            # order_desc, sorgunun VERİTABANI SEVİYESİNDE (row_limit ile
            # birlikte "top N" satır SEÇİMİ için) hangi yönde sıralanacağını
            # belirler -- x_axis_sort_asc ise SADECE post_processing'teki
            # GÖRÜNTÜLEME sırasını kontrol eder. İkisi prensipte bağımsızdır.
            # "not sort_ascending" eşitlemesi şu an güvenli çünkü row_limit
            # (10000) gerçek kategori sayısından kat kat büyük -- yani
            # order_desc'in satır SEÇME işlevi hiç devreye girmiyor, sadece
            # doldurulması gereken bir alan. limit/timeseries_limit_metric
            # (series limit / "top N") eklendiğinde row_limit gerçekten
            # satır elemeye başlayacak; o noktada bu coupling YANLIŞ olur --
            # örn. "en çok satan 10 kategoriyi küçükten büyüğe sırala" gibi
            # bir istekte seçim (descending) ile görüntüleme (ascending) zıt
            # yönde olmalı. O zaman bu satır kaldırılıp order_desc ayrı bir
            # kararla (ya da sabit "her zaman descending seç" kuralıyla)
            # ele alınmalı.
            "order_desc": not chart_params.get("sort_ascending", False),
        })
        # time_grain, chart_type'a değil axis'in GERÇEK rolüne bağlı: bar'da
        # axis temporal seçildiyse ("aylık sipariş sayısı") de gerekli.
        if axis_role == "temporal":
            params["time_grain_sqla"] = chart_params["time_grain"]

        # xAxisForceCategorical: axis_column DB'de sayısal (integer/smallint)
        # ama bizim sınıflandırmamızda categorical ise (örn. dim_date.quarter:
        # 1/2/3/4 -- toplanacak bir değer değil, bir gruplama), Superset'e
        # bunu sürekli/numeric eksen değil ayrık kategori olarak çizdirtiyoruz.
        # LLM'e sorulmuyor -- axis_column zaten belli, tipi de DB'den biliniyor.
        if axis_role == "categorical" and axis_info["data_type"] in NUMERIC_SQL_TYPES:
            params["xAxisForceCategorical"] = True

        if chart_type == "bar":
            params["orientation"] = "vertical"
            # x_axis_sort (metrik büyüklüğüne göre sıralama) SADECE axis
            # categorical iken anlamlı ("en çok satan kategori önce" gibi).
            # axis temporal ise ("aylık sipariş sayısı") bunu eklemiyoruz,
            # yoksa Superset ayları kronolojik sırada değil metrik büyüklüğüne
            # göre dizer -- sessizce yanlış/yanıltıcı bir grafik üretir.
            if axis_role == "categorical":
                sort_metric_label = "count" if metric == "count" else metric["label"]
                params["x_axis_sort"] = sort_metric_label

        # contribution_mode: kullanıcı "pay/yüzde/oranı" istediyse Superset'e
        # her axis değerinin toplam içindeki payını hesaplattırıyoruz.
        # NOT: "column" seçimi veriye/soruya göre değil, pivot YAPIMIZA göre
        # sabit doğru: groupby her zaman [] olduğu için pivot tek kolonlu,
        # bu şekilde "row" hep %100 (anlamsız), "column" hep gerçek pay verir.
        # Bu varsayım groupby boş kaldığı sürece geçerli -- ileride çoklu-boyut
        # (groupby dolu) desteği eklenirse bu mantık YENİDEN değerlendirilmeli,
        # o yüzden burada sessizce yanlış sonuç üretmek yerine hata veriyoruz.
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


# ---- Yerel (Superset'siz) chart üretimi ----
#
# Superset chart'ı, JOIN mantığını kendi içindeki "virtual dataset"te
# (DATASET_ID) tutuyor -- Python kodu bu JOIN'leri bilmiyor. Superset açık/
# kurulu olmayan bir kullanıcı için de grafiği gösterip indirtebilmek adına,
# aynı chart_params'tan (zaten is_chart_params_safe'ten geçmiş, whitelist'li)
# bağımsız bir SQL sorgusu üretip PostgreSQL'e doğrudan soruyoruz.
#
# JOIN'leri Python'da elle sabit kodlamak yerine (gerçek foreign-key kolon
# adları elimizde net olmadığı için) answer_with_sql'de zaten kanıtlanmış
# çalışan yöntemi tekrar kullanıyoruz: LLM'e join_info'yu vererek SQL
# yazdırıyoruz -- ama LLM'e YORUMLAMA alanı bırakmıyoruz, sadece ZATEN
# belirlenmiş (axis/metric/filtre) parametreleri SQL sözdizimine çevirmesini
# istiyoruz. Üretilen SQL yine is_sql_safe'ten geçmek zorunda -- aynı
# whitelist savunması burada da geçerli.

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
    chart_params'ın ZATEN is_chart_params_safe'ten geçmiş olması bekleniyor
    -- burada ayrıca is_sql_safe ile üretilen SQL'in kendisi de doğrulanıyor
    (defense-in-depth: hem parametreler hem üretilen SQL whitelist'ten geçiyor).
    """
    columns_summary, joins_summary = build_schema_context(relevant_columns)
    sql = build_chart_query_sql(chart_params, columns_summary, joins_summary)

    is_safe, message = is_sql_safe(sql)
    if not is_safe:
        raise ValueError(f"Üretilen chart SQL'i reddedildi: {message}\nSQL: {sql}")

    rows = run_sql(sql)
    return rows, sql


def render_chart_png(chart_type, rows, axis_label, metric_label):
    """
    SQL sonucundan (rows: [(axis_value, metric_value), ...]) bir PNG üretir.
    Superset'e hiç bağımlı değil -- matplotlib gerekir (pip install matplotlib).
    """
    import matplotlib
    matplotlib.use("Agg")  # ekran/GUI gerektirmeyen backend
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
        return f"Chart oluşturuldu: {SUPERSET_URL}/explore/?slice_id={result['id']}"
    return f"⚠️ Chart oluşturulamadı: {result}"


def answer_with_chart_full(user_question, relevant_columns):
    """
    answer_with_chart'ın Streamlit için genişletilmiş hali -- düz metin
    yerine, arayüzün ihtiyaç duyduğu her parçayı (hata mesajı / Superset
    linki / yerel PNG buffer) ayrı ayrı bir sözlükte döner. Superset chart'ı
    oluşturma adımı burada da opsiyonel: erişilemiyorsa (bağlantı hatası
    vb.) yerel PNG yine de üretilmeye çalışılır -- kullanıcı Superset'e
    hiç erişemese bile grafiği görüp indirebilsin diye.
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

    # Superset'e kaydetmeyi dene (opsiyonel -- başarısız olursa yerel PNG'yi
    # engellemesin). dataset_id çözümlemesi de bu try içinde: farklı
    # dataset'lere ait kolonlar karışmışsa bu da bir "Superset'e kaydedemedim"
    # durumu olarak ele alınıp yerel PNG akışına düşülür.
    try:
        dataset_id = resolve_dataset_id(chart_params, enriched)
        session = superset_login()
        payload = build_chart_payload(chart_type, chart_params, enriched, dataset_id, user_question[:100])
        response = session.post(f"{SUPERSET_URL}/api/v1/chart/", json=payload)
        superset_result = response.json()
        if "id" in superset_result:
            result["superset_url"] = f"{SUPERSET_URL}/explore/?slice_id={superset_result['id']}"
        else:
            result["warning"] = f"Superset chart'ı oluşturulamadı: {superset_result}"
    except Exception as e:
        result["warning"] = f"Superset'e ulaşılamadı ({e}) -- sadece yerel grafik gösteriliyor."

    # Yerel SQL + PNG (Superset'ten bağımsız)
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


# ---- Main entry point ----

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