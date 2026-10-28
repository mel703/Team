
"""
Общие функции проекта Sensorless Drive Diagnosis (common.py).

Мелине: load_data(), split().
Сергей: prepare(), evaluate(), get_model(), outliers().
День 1:
- prepare()   — разбиение и нормализация без утечки теста;
- evaluate()  — accuracy, F1 и матрица ошибок;
- get_model() — базовый RandomForestClassifier;
- outliers()  — поиск строк с выбросами по правилу |z| > 3.

Все скрипты команды начинаются со строки:
    from common import load_data, split, prepare, evaluate, get_model, outliers

Формат исходного файла:
48 признаков + класс (1..11), разделитель — пробел.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
правка: убраны from __future__, typing.Union и ArrayLike —
функции принимают только таблицы pandas, другие типы данных не нужны
"""

"""Мелине"""
DATA_PATH = Path(__file__).parent.parent / 'data' / 'Sensorless_drive_diagnosis.txt'

COLUMNS = []
for stat in ('mean', 'std', 'skew', 'kurt'):
    for i in range(1, 13):
        COLUMNS.append(f'{stat}_{i}')

"""
правка:Сергей, функция _load_data() убрана — файл читает только load_data().
Её проверки (49 столбцов, NaN) перенесены в load_data().
Проверка ndim не нужна: read_csv всегда возвращает таблицу.
"""


def load_data(path=DATA_PATH):
    """
    Мелине: читает файл, делит на признаки X и класс y, даёт столбцам имена.
    Сергей: проверки формата файла.
    """
    df = pd.read_csv(path, sep=' ', header=None)

    """Реально, проверки нужны, я что-не задумалась"""
    """Сергей"""

    if df.shape[1] != 49:
        raise ValueError(
            f'Ожидается 49 столбцов: 48 признаков + класс. Получено: {df.shape[1]}.'
        )
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('В данных обнаружены NaN или бесконечные значения.')

    """Мелине"""
    X = df.iloc[:, :48]
    X.columns = COLUMNS
    y = df[48].astype(int)
    if not y.between(1, 11).all():
        raise ValueError('Номер класса должен быть от 1 до 11.')
    return X, y


def split(X, y):
    """Мелине: 70/30 со стратификацией, одно и то же разбиение у всей команды."""
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


"""
правка: prepare() больше не читает файл и не делит данные —
это уже делают load_data() и split(), иначе разбиение было бы в двух местах.
Возвращает таблицы с именами столбцов: они нужны Юлии, Параскуну и Валерии.
scaler не возвращаем: для новых замеров в main.py хватит prepare(X_train, X_new).
"""


def prepare(X_train, X_test):
    """
    Сергей: z-нормализация без утечки теста.
    1. StandardScaler.fit() — только на X_train.
    2. К X_test применяется уже обученный scaler.
    Возвращает: X_train, X_test — таблицы с теми же именами столбцов.
    День 2: сюда добавятся решения Валерии (выбросы) и Юлии (лишние столбцы).
    """
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train),
                             columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test),
                            columns=X_test.columns, index=X_test.index)
    return X_train_s, X_test_s


"""
правка: F1 теперь и по каждому из 11 классов ('f1') — это нужно Жанне,
f1_macro оставлен. Матрица ошибок сохраняется в figures/ вместо plt.show():
plt.show() останавливает скрипт, пока не закроешь окно.
"""
FIGURES_PATH = Path(__file__).parent.parent / 'figures'
CLASSES = np.arange(1, 12)


def evaluate(model, X_test, y_test, name='confusion_matrix', show=False):
    """
    Сергей: оценить уже обученную модель на тестовой части.
    Модель и нормализация должны быть обучены до вызова evaluate().
    Возвращает: {'accuracy': число, 'f1': 11 чисел, 'f1_macro': число};
    картинку матрицы ошибок кладёт в figures/<name>.png.
    """
    y_pred = model.predict(X_test)
    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, labels=CLASSES, average=None, zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
    }

    FIGURES_PATH.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=CLASSES, xticks_rotation='vertical'
    )
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{name}.png', dpi=150)
    if show:
        plt.show()
    plt.close()
    return result


"""правка: n_estimators=200 указан явно — на день 3 Жанна меняет одну цифру"""


def get_model(random_state=42):
    """
    Сергей: базовая модель проекта.
    Настройки Жанны вносятся на день 3, после этого — заморозка.
    """
    return RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )


"""
правка: среднее и СКО считаются по X_train, а выбросы ищем в X —
как в правиле «тест не трогать». Возвращается таблица True/False по каждой
клетке (Валерии нужно считать по столбцам). Строки с выбросом:
outliers(X_train, X).any(axis=1).
"""


def outliers(X_train, X, threshold=3.0):
    """
    Сергей: выбросы по правилу трёх сигм |z| > threshold.
    Возвращает: таблицу True/False того же размера, что X.
    """
    mean = X_train.mean()
    std = X_train.std()  # ddof=1, как было у Сергея
    safe_std = std.replace(0, 1.0)  # Сергей: константные признаки не считаем источником выбросов
    z = (X - mean) / safe_std
    return z.abs() > threshold


if __name__ == '__main__':
    """Мелине: быстрая проверка, что common.py работает"""
    X, y = load_data()
    print(X.shape)  # (58509, 48)
    print(y.value_counts())  # 11 классов по 5319
    X_train, X_test, y_train, y_test = split(X, y)
    print(len(X_train), len(X_test))  # 40956 17553

    """Сергей"""
    X_train_s, X_test_s = prepare(X_train, X_test)
    print(X_train_s.columns[:3].tolist())  # имена столбцов сохранились
    print(outliers(X_train, X).any(axis=1).mean())  # доля строк с выбросами

