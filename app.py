import streamlit as st
import pandas as pd
import joblib

# Загружаем модель и пайплайн
model = joblib.load("models/model.joblib")
pipeline = joblib.load("pipeline/preprocessing_pipeline.joblib")

# Интерфейс
st.title("💓 Предсказание сердечно-сосудистых заболеваний")

# Добавляем переключатель для ввода данных вручную
input_method = st.radio(
    "Выберите метод ввода данных:",
    ("Загрузить CSV-файл", "Ввести данные вручную")
)

# 📁 Ввод через файл
if input_method == "Загрузить CSV-файл":
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:")
        st.dataframe(data)

        # Кнопка предсказания
        if st.button("📊 Предсказать"):
            # Приведение типов
            numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
            cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            for col in cat_cols:
                if col in data.columns:
                    data[col] = data[col].astype(str)

            try:
                X = pipeline.transform(data)
                preds = model.predict(X)
                st.write("Результат предсказания:")
                st.write(preds)
            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")

# ✍ Ручной ввод
else:
    st.write("Введите данные вручную:")

    # Ввод признаков
    age = st.number_input("Возраст", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Пол", options=[1, 2], format_func=lambda x: "Женщина" if x == 1 else "Мужчина")
    height = st.number_input("Рост (см)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Вес (кг)", min_value=20, max_value=300, value=70)
    ap_hi = st.number_input("Систолическое давление", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Диастолическое давление", min_value=30, max_value=150, value=80)

    cholesterol = st.selectbox(
        "Уровень холестерина", 
        options=[1, 2, 3],
        format_func=lambda x: f"Уровень {x} (1 - нормальный, 2 - выше нормы, 3 - высокий)"
    )

    gluc = st.selectbox(
        "Уровень глюкозы", 
        options=[1, 2, 3],
        format_func=lambda x: f"Уровень {x} (1 - нормальный, 2 - выше нормы, 3 - высокий)"
    )

    smoke = st.selectbox("Курите ли вы?", options=[0, 1], format_func=lambda x: "Да" if x else "Нет")
    alco = st.selectbox("Употребляете ли алкоголь?", options=[0, 1], format_func=lambda x: "Да" if x else "Нет")
    active = st.selectbox("Активный образ жизни?", options=[0, 1], format_func=lambda x: "Да" if x else "Нет")

    # Вычисляем BMI
    bmi = weight / (height / 100) ** 2
    st.write(f"💡 Индекс массы тела (BMI): **{bmi:.2f}**")

    if st.button("📊 Предсказать"):
        # Подготовка данных
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active],
            'bmi': [bmi]
        })

        # Приведение типов
        cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        for col in cat_cols:
            input_data[col] = input_data[col].astype(str)

        # Предсказание
        try:
            X = pipeline.transform(input_data)
            pred = model.predict(X)
            st.success(f"🩺 Результат: {'Есть риск' if pred[0] == 1 else 'Риск не обнаружен'}")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
