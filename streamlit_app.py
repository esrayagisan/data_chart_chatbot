"""
Olist Chatbot -- Streamlit arayüzü.

Çalıştırmak için:
    pip install streamlit matplotlib
    streamlit run streamlit_app.py

Chart isteklerinde iki şey birden üretilir:
  1) Superset'te kalıcı bir chart (varsa erişim, link olarak gösterilir)
  2) Superset'ten TAMAMEN bağımsız, yerel bir PNG (matplotlib) -- Superset
     kurulu/açık olmayan bir kullanıcı bile grafiği görüp indirebilsin diye.
"""

import streamlit as st
from data_chatbot import (
    fetch_data_catalog,
    get_all_columns,
    answer_with_sql,
    answer_with_chart_full,
)

st.set_page_config(page_title="Olist Chatbot", layout="centered")
st.title("Olist Veri Ambarı Chatbot")

question = st.text_input("Sorunuzu yazın:")
want_chart = st.checkbox("Grafik olarak göster")
submit = st.button("Gönder")

if submit and question.strip():
    with st.spinner("Katalog yükleniyor..."):
        catalog = fetch_data_catalog()
        relevant_columns = get_all_columns(catalog)

    if not want_chart:
        with st.spinner("SQL üretiliyor ve çalıştırılıyor..."):
            answer = answer_with_sql(question, relevant_columns)
        st.markdown(answer)

    else:
        with st.spinner("Grafik parametreleri belirleniyor..."):
            result = answer_with_chart_full(question, relevant_columns)

        if "error" in result:
            st.error(f"⚠️ Chart reddedildi: {result['error']}")
        else:
            if result.get("warning"):
                st.warning(result["warning"])

            if result.get("superset_url"):
                st.markdown(f"🔗 [Grafiği Superset'te görüntüle]({result['superset_url']})")

            if result.get("png_buf"):
                st.image(result["png_buf"], caption=question, use_column_width=True)
                st.download_button(
                    "Grafiği indir (PNG)",
                    data=result["png_buf"],
                    file_name="chart.png",
                    mime="image/png",
                )
            elif "error_local_render" in result:
                st.error(f"Yerel grafik üretilemedi: {result['error_local_render']}")

            with st.expander("Üretilen SQL'i gör"):
                st.code(result.get("sql") or "(SQL üretilemedi)", language="sql")