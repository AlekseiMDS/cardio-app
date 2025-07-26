FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . /app
WORKDIR /app

# Команда запуска Streamlit
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false"]
