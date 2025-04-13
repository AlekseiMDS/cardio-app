def get_metrics_classifier(y_test, y_pred, y_score, name):
    """
    Вычисляет метрики классификации для моделей машинного обучения и возвращает их в виде DataFrame.

    Функция принимает истинные метки классов, предсказанные классы и вероятности предсказаний,
    вычисляет ключевые метрики классификации и возвращает их в виде таблицы.

    Метрики:
        - ROC_AUC (площадь под ROC-кривой)
        - Recall/Sensitivity (чувствительность)
        - Precision/PPV (положительная предсказательная ценность)
        - NPV (отрицательная предсказательная ценность)
        - F1-score (гармоническое среднее precision и recall)
        - Accuracy (доля правильных предсказаний)
        - Average Precision (средняя точность, AP)

    Аргументы:
        y_test (array-like): Истинные метки классов (0 или 1).
        y_pred (array-like): Предсказанные классы (0 или 1).
        y_score (array-like): Предсказанные вероятности классов (формат: [n_samples, 2]).
        name (str): Название модели (будет добавлено в DataFrame).

    Возвращает:
        pd.DataFrame: Таблица с рассчитанными метриками.

    Пример использования:
        >>> df_metrics = get_metrics_classifier(y_test, y_pred, y_score, "RandomForest")
        >>> print(df_metrics)
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, average_precision_score
    import pandas as pd
    import numpy as np
    
    df_metrics = pd.DataFrame()
    
    df_metrics['model'] = [name]

    # confusion_matrix для вычисления метрики NPV
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Основные метрики для задачи классификации
    df_metrics['ROC_AUC'] = roc_auc_score(y_test, y_score[:, 1])
    df_metrics['Recall/Sensitivity'] = recall_score(y_test, y_pred)
    df_metrics['Precision/PPV'] = precision_score(y_test, y_pred)
    df_metrics['NPV'] = tn / (tn + fn)  # Negative Predictive Value
    df_metrics['f1'] = f1_score(y_test, y_pred)
    df_metrics['Accuracy'] = accuracy_score(y_test, y_pred)
    df_metrics['Average_Precision'] = average_precision_score(y_test, y_score[:, 1])
            
    return df_metrics


def highlight_top_values(x):
    """
    Функция используется после применения функции из группы get_metrics.
    Функция ничего не возвращает. Находит в DataFrame ячейку с максимальным значением по колонкам
    и выделяет ее цветом.
    background-color: #90EE90 - Зеленый: 1 место (максимальное значение в колонке)
    background-color: #FFD580 - Оранжевый: 2 место
    background-color: #FFFF99 - Желтый : 3 место
    color: black -  меняет цвет текста в выделенных колонках на черный

    Пример использования:
    >>> df_metrics.style.apply(highlight_top_values, axis=0)
    >>> print(df_metrics)
    """
    import pandas as pd
    import numpy as np
    
    if x.dtype == 'O':  # Проверяем, является ли колонка строковой (object)
        return [''] * len(x)
    
    max_val = x.max()
    second_max = x.nlargest(2).iloc[-1] if len(x) > 1 else max_val
    third_max = x.nlargest(3).iloc[-1] if len(x) > 2 else second_max
    
    return [
        'background-color: #90EE90; color: black' if v == max_val else  # Светло-зеленый (лучший)
        'background-color: #FFD580; color: black' if v == second_max else  # Светло-оранжевый (2 место)
        'background-color: #FFFF99; color: black' if v == third_max else  # Светло-желтый (3 место)
        ''  # Без цвета
        for v in x
    ]


def check_overfitting(model, X_train, y_train, X_valid, y_valid, metric_fun):
    """
    Проверяет наличие overfitting (переобучения) у модели машинного обучения.

    Функция вычисляет заданную метрику на обучающей и валидационной выборках 
    и сравнивает их, чтобы оценить степень переобучения.

    Аргументы:
        model (sklearn-like estimator): Обученная модель, поддерживающая метод `predict`.
        X_train (array-like): Матрица признаков обучающей выборки.
        y_train (array-like): Целевые метки обучающей выборки.
        X_valid (array-like): Матрица признаков валидационной выборки.
        y_valid (array-like): Целевые метки валидационной выборки.
        metric_fun (callable): Функция метрики (например, `roc_auc_score`, `accuracy_score`).

    Выводит:
        - Значение метрики на обучающей выборке.
        - Значение метрики на валидационной выборке.
        - Разницу между ними в процентах.

    Пример использования:
        >>> check_overfitting(model, X_train, y_train, X_valid, y_valid, roc_auc_score)
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, average_precision_score
    import pandas as pd
    import numpy as np
   
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    value_train = metric_fun(y_train, y_pred_train)
    value_valid = metric_fun(y_valid, y_pred_valid)

    print(f'{metric_fun.__name__} train: %.3f' % value_train)
    print(f'{metric_fun.__name__} valid: %.3f' % value_valid)
    print(f'delta = {(abs(value_train - value_valid)/value_valid*100):.1f} %')


def create_logreg_param_distributions():
    """
    Создает список распределений гиперпараметров для моделей логистической регрессии 
    с различными солверами и штрафами.

    Возвращает:
        list: Список словарей, где каждый словарь представляет собой набор 
              гиперпараметров для использования в RandomizedSearchCV.

    Логика выбора параметров:
    - 'liblinear': поддерживает штрафы 'l1' и 'l2'.
    - 'saga': поддерживает штрафы 'l1', 'l2', 'elasticnet' и None.
    - 'lbfgs', 'sag', 'newton-cg': поддерживают штрафы 'l2' и None.

    Каждый набор гиперпараметров включает:
    - 'solver': алгоритм решения (солвер).
    - 'penalty': тип регуляризации.
    - 'C': коэффициенты силы регуляризации.
    - 'max_iter': максимальное количество итераций.
    - 'class_weight': учет дисбаланса классов (сбалансированные веса или None).
    - 'l1_ratio': (только для 'saga' с 'elasticnet') 
      значения от 0 до 1 для регулирования соотношения L1 и L2.

    Пример использования:
        >>> param_distributions = create_param_distributions()
        >>> print(len(param_distributions))  # Количество сгенерированных наборов параметров
    """
    import pandas as pd
    import numpy as np
    
    param_distributions = []
    for solver in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']:
        if solver == 'liblinear':
            penalties = ['l1', 'l2']
        elif solver == 'saga':
            penalties = ['l1', 'l2', 'elasticnet', None]
        elif solver in ['lbfgs', 'sag', 'newton-cg']:
            penalties = ['l2', None]
        else:
            penalties = []

        for penalty in penalties:
            params = {
                'solver': [solver],
                'penalty': [penalty],
                'max_iter': [100, 200, 500],
                'class_weight': ['balanced', None],
            }
            # Убираем 'C', если penalty=None (чтобы избежать предупреждения)
            if penalty is not None:
                params['C'] = [0.001, 0.01, 0.1, 1, 10, 100]

            # Добавляем l1_ratio только для 'saga' с 'elasticnet'
            if solver == 'saga' and penalty == 'elasticnet':
                params['l1_ratio'] = np.linspace(0, 1, 5)
            
            param_distributions.append(params)
    
    return param_distributions