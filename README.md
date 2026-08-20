# Olist Chatbot

Olist e-ticaret veri ambarı üzerinde doğal dilde soru sorup SQL cevabı veya
grafik (chart) alabileceğiniz bir Streamlit uygulaması. Sorular DeepSeek LLM
ile SQL'e / grafik parametrelerine çevrilir, üretilen SQL ve parametreler
çalıştırılmadan önce whitelist tabanlı güvenlik kontrollerinden geçer.

## Mimari

```
Kullanıcı (Streamlit arayüzü)
        │
        ▼
  soru + data_catalog (PostgreSQL)
        │
        ▼
   DeepSeek LLM  ──► SQL ya da chart parametreleri üretir
        │
        ▼
  güvenlik kontrolleri (is_sql_safe / is_chart_params_safe)
        │
        ├─► PostgreSQL (fact/dim tabloları) ──► sonuç metni
        │
        └─► Superset API (opsiyonel, kalıcı chart linki)
                 +
            matplotlib (yerel, Superset'ten bağımsız PNG)
```

- **SQL modu:** Soru, `data_catalog` tablosundaki kolon/join bilgisiyle
  birlikte LLM'e verilir, dönen SQL sadece `SELECT` ve izinli tablolar
  (`ALLOWED_TABLES`) üzerinde çalışıyorsa çalıştırılır.
- **Chart modu:** LLM önce chart tipini (line/bar/pie), sonra eksen/metrik/
  filtre parametrelerini üretir. Parametreler `is_chart_params_safe` ile
  doğrulandıktan sonra hem Superset'te kalıcı bir chart oluşturulmaya
  çalışılır hem de aynı parametrelerden bağımsız bir SQL sorgusuyla yerel bir
  PNG (matplotlib) üretilir — Superset'e erişimi olmayan bir kullanıcı bile
  grafiği görüp indirebilir.

## Kurulum

```bash
pip install -r requirements.txt
```

Ortam değişkenlerini ayarlayın (bkz. `.env.example`), ardından:

```bash
streamlit run streamlit_app.py
```

### Docker ile çalıştırma

```bash
docker build -t data-chatbot .
docker run -p 8501:8501 --env-file .env data-chatbot
```

## Gerekli ortam değişkenleri

| Değişken | Açıklama |
|---|---|
| `DWH_DB_HOST` | PostgreSQL veri ambarı host adresi |
| `DWH_DB_PORT` | PostgreSQL port |
| `DWH_DB_NAME` | Veritabanı adı |
| `DWH_DB_USER` | Veritabanı kullanıcı adı |
| `DWH_DB_PASSWORD` | Veritabanı şifresi |
| `SUPERSET_URL` | Superset instance URL'i (varsayılan: `http://localhost:8088`) |
| `SUPERSET_USERNAME` | Superset kullanıcı adı |
| `SUPERSET_PASSWORD` | Superset şifresi |
| `SUPERSET_DATASET_ID` | Varsayılan Superset dataset ID'si |
| `DEEPSEEK_API_KEY` | DeepSeek API anahtarı |

## Güvenlik notları

- Üretilen SQL yalnızca `SELECT` ise ve yalnızca `ALLOWED_TABLES` içindeki
  tabloları kullanıyorsa çalıştırılır.
- Chart parametreleri (eksen/metrik/filtre kolonları, operatörler, zaman
  aralığı formatı) çalıştırılmadan önce tip/rol bazlı olarak doğrulanır;
  kategori filtre değerleri SQL anahtar kelimesi/özel karakter içeriyorsa
  reddedilir.
- HAVING ifadesi LLM'in ürettiği serbest metinden değil, zaten
  whitelist'ten geçmiş `aggregate`/`metric_column` değerlerinden uygulama
  tarafında inşa edilir.

## Proje durumu

Geliştirme aşamasında bir portföy/öğrenme projesidir; test kapsamı henüz
yoktur.
