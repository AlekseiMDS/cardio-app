# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системную библиотеку libgomp1 (нужна для LightGBM)
RUN apt-get update && apt-get install -y libgomp1

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Команда по умолчанию — запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
