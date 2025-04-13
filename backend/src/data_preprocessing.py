import pandas as pd
import joblib

# Загружаем обученные пайплайн и модель
pipeline = joblib.load("pipeline/preprocessing_pipeline.joblib")
model = joblib.load("models/model.joblib")


def load_data(path: str):
    """Загружает датасет из CSV-файла"""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """Обрабатывает признаки и возвращае X"""
    object_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    for col in object_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return pipeline.transform(df)


def predict(path: str):
    """Загружает данные, обрабатывает их и делает предсказание"""
    df = load_data(path)
    X_user = preprocess_data(df)
    preds = model.predict(X_user)
    return preds
