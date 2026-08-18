FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#copy chatbot2'den önce: her kod değişikliğinde tekrar çalışmıyor, yalnızca req değişince

COPY chatbot2.py .
COPY streamlit_app.py .

EXPOSE 8501

# --server.address=0.0.0.0: container dışından erişilebilir olması için
# --server.headless=true: tarayıcı otomatik açma denemesini kapatır (sunucuda gereksiz)
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
