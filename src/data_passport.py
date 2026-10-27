"""
Мелине: паспорт данных Sensorless Drive Diagnosis.

Что проверяет:
1. Размер данных.
2. Пропуски.
3. Дубли (полные и «одинаковые признаки — разные классы»).
4. Баланс классов.
5. Диапазоны столбцов (по блокам и по каждому столбцу).
6. Расшифровку столбцов: 1–12 — среднее, 13–24 — СКО, 25–36 — асимметрия, 37–48 — эксцесс.

Запуск: python data_passport.py
Результат: вывод в консоль + таблица reports/data_passport.csv
"""

from pathlib import Path

import pandas as pd

from common import load_data

REPORTS_PATH = Path(__file__).parent.parent / 'reports'
BLOCKS = ('mean', 'std', 'skew', 'kurt')


def passport():
    """Мелине: собрать паспорт данных и сохранить таблицу по столбцам."""
    X, y = load_data()

    print('1. Размер')
    print(f'   строк: {len(X)}, признаков: {X.shape[1]}, классов: {y.nunique()}')

    print('2. Пропуски')
    print(f'   всего пропусков: {int(X.isna().sum().sum() + y.isna().sum())}')

    print('3. Дубли')
    full = pd.concat([X, y.rename('cls')], axis=1)
    print(f'   полных дублей строк: {int(full.duplicated().sum())}')
    conflicts = full.groupby(list(X.columns))['cls'].nunique()
    print(f'   одинаковые признаки с разными классами: {int((conflicts > 1).sum())}')

    print('4. Баланс классов')
    counts = y.value_counts().sort_index()
    for cls, n in counts.items():
        print(f'   класс {cls:>2}: {n} ({n / len(y):.1%})')
    print(f'   самый малый / самый большой класс: {counts.min() / counts.max():.2f}')

    print('5. Диапазоны по блокам')
    for block in BLOCKS:
        part = X.filter(like=f'{block}_')
        print(f'   {block:>4}: от {part.min().min():.4g} до {part.max().max():.4g}')
    spread = X.std()
    print(f'   СКО столбцов: от {spread.min():.2g} до {spread.max():.2g} '
          f'— масштабы разные, нужна нормализация')

    print('6. Проверка расшифровки столбцов')
    std_min = X.filter(like='std_').min().min()
    kurt_min = X.filter(like='kurt_').min().min()
    print(f'   std_*: минимум {std_min:.4g} — СКО не бывает отрицательным: '
          f'{"сходится" if std_min >= 0 else "НЕ сходится"}')
    print(f'   kurt_*: минимум {kurt_min:.3g} — эксцесс не бывает меньше -2: '
          f'{"сходится" if kurt_min >= -2 else "НЕ сходится"}')
    tail = X[[f'kurt_{i}' for i in range(7, 13)]].mean()
    print(f'   kurt_7…kurt_12: среднее {tail.mean():.3f} '
          f'(у чистой синусоиды эксцесс равен -1,5)')

    REPORTS_PATH.mkdir(exist_ok=True)
    table = pd.DataFrame({
        'min': X.min(), 'max': X.max(), 'mean': X.mean(), 'std': X.std(),
        'missing': X.isna().sum(),
    })
    table.to_csv(REPORTS_PATH / 'data_passport.csv', encoding='utf-8-sig')
    print(f'Таблица по столбцам сохранена: "data_passport.csv')


if __name__ == '__main__':
    passport()
