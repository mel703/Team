import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import load_data, split, prepare, evaluate, get_model

# Пути — относительно расположения скрипта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / 'figures'
REPORTS = PROJECT_ROOT / 'reports'
FIGURES.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# ---------- 1. Загрузка и подготовка ----------
print('ЗАГРУЗКА ДАННЫХ')
X, y = load_data()
X_train, X_test, y_train, y_test = split(X, y)
print(f'Обучающая часть: {X_train.shape}')
print(f'Тестовая часть:  {X_test.shape}')

X_train_raw, X_test_raw = X_train.copy(), X_test.copy()
X_train_norm, X_test_norm = prepare(X_train.copy(), X_test.copy())


# ---------- 2. Разведочный анализ ----------
def measure(model, X_train_, y_train_, X_test_, y_test_, name):
    t0 = time.perf_counter()
    model.fit(X_train_, y_train_)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test_)
    acc = accuracy_score(y_test_, y_pred)
    f1 = f1_score(y_test_, y_pred, average='macro')

    print(f'{name:<40} accuracy = {acc:.4f}   f1_macro = {f1:.4f}   '
          f'время = {train_time:.2f} с')
    return {
        'модель': name,
        'accuracy': round(acc, 4),
        'f1_macro': round(f1, 4),
        'время_обучения_с': round(train_time, 2),
    }


print('\nРАЗВЕДОЧНЫЙ АНАЛИЗ')
results = []
results.append(measure(KNeighborsClassifier(n_neighbors=5),
                       X_train_raw, y_train, X_test_raw, y_test,
                       'kNN (сырые данные)'))
results.append(measure(KNeighborsClassifier(n_neighbors=5),
                       X_train_norm, y_train, X_test_norm, y_test,
                       'kNN (после нормализации)'))
results.append(measure(LogisticRegression(max_iter=1000),
                       X_train_norm, y_train, X_test_norm, y_test,
                       'Логистическая регрессия'))
results.append(measure(get_model(),
                       X_train_norm, y_train, X_test_norm, y_test,
                       'Случайный лес'))

table = pd.DataFrame(results)
print('\nТАЛБИЦА РАЗВЕДОЧНОГО АНАЛИЗА')
print(table.to_string(index=False))
table.to_csv(REPORTS / 'razvedka_table.csv', index=False)
print(f'Таблица сохранена: {REPORTS / "razvedka_table.csv"}')


# ---------- 3. Гипотеза Г5 ----------
print('\n=== Гипотеза Г5 ===')
acc_lr = table.loc[table['модель'] == 'Логистическая регрессия', 'accuracy'].values[0]
acc_rf = table.loc[table['модель'] == 'Случайный лес', 'accuracy'].values[0]
diff_g5 = (acc_rf - acc_lr) * 100
print(f'Accuracy логистической регрессии: {acc_lr:.4f}')
print(f'Accuracy случайного леса:         {acc_rf:.4f}')
print(f'Разница: {diff_g5:.2f} п.п.')
if diff_g5 < 2.0:
    print('Г5 ПОДТВЕРЖДЕНА: сложная модель не нужна')
else:
    print('Г5 ОПРОВЕРГНУТА: случайный лес заметно лучше')


# ---------- 4. Гипотеза Г6 ----------
print('\n=== Гипотеза Г6 ===')
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_norm, y_train)
importances = pd.Series(rf.feature_importances_, index=X_train_norm.columns)
importances = importances.sort_values(ascending=False)
print('\nТоп-10 самых важных признаков:')
print(importances.head(10).to_string())

top10 = importances.head(10).index.tolist()
X_train_top = X_train_norm[top10]
X_test_top = X_test_norm[top10]

rf_top = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_top.fit(X_train_top, y_train)
acc_top = accuracy_score(y_test, rf_top.predict(X_test_top))
diff_g6 = (acc_rf - acc_top) * 100
print(f'\nAccuracy на всех 48 признаках:    {acc_rf:.4f}')
print(f'Accuracy на 10 важных признаках:  {acc_top:.4f}')
print(f'Падение: {diff_g6:.2f} п.п.')
if diff_g6 < 1.0:
    print('Г6 ПОДТВЕРЖДЕНА: 38 признаков можно убрать')
else:
    print('Г6 ОПРОВЕРГНУТА: все 48 признаков важны')


# ---------- 5. Графики ----------
plt.figure(figsize=(10, 6))
importances.head(20).plot(kind='barh', color='steelblue')
plt.gca().invert_yaxis()
plt.xlabel('Важность', fontsize=12)
plt.title('Топ-20 самых важных признаков', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES / 'feature_importance_top20.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 5))
colors = ['#e74c3c', '#e74c3c', '#3498db', '#2ecc71']
plt.bar(table['модель'], table['accuracy'], color=colors)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Разведочный анализ: сравнение моделей', fontsize=14)
plt.xticks(rotation=15, ha='right')
for i, v in enumerate(table['accuracy']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(FIGURES / 'models_comparison.png', dpi=150)
plt.show()

print('\nГотово. Графики в figures/, таблица в reports/')