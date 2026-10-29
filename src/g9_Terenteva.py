"""
Гипотеза Г9: Анализ структуры ошибок модели.
Вопрос: Путает ли модель всё подряд или только похожие неисправности?
Формулировка: Больше половины всех ошибок простой модели (LogisticRegression) 
приходится на 3 пары классов.
Автор: Терентьева Д.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Импортируем общие функции из common.py
from common import load_data, split, prepare

# 1. Загружаем и готовим данные
print("Загрузка данных...")
X, y = load_data()
X_train, X_test, y_train, y_test = split(X, y)
X_train, X_test = prepare(X_train, X_test)

# 2. Обучаем простую модель (LogisticRegression)
print("Обучение LogisticRegression...")
# max_iter=1000 нужен, чтобы алгоритм точно сошелся на наших данных
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 3. Делаем предсказания
y_pred = model.predict(X_test)

# 4. Строим матрицу ошибок
cm = confusion_matrix(y_test, y_pred)

# 5. Считаем долю ошибок в топ-3 парах классов
# Делаем копию матрицы, чтобы не испортить оригинал для картинки
cm_calc = cm.copy()
np.fill_diagonal(cm_calc, 0)  # Обнуляем диагональ (правильные ответы)

total_errors = cm_calc.sum()
print(f"\nВсего ошибок на тесте: {total_errors}")

# Находим индексы 3 самых больших значений (топ-3 пары)
# np.argsort сортирует по возрастанию, поэтому берем с конца [::-1]
flat_indices = np.argsort(cm_calc, axis=None)[::-1][:3] 
top_3_errors = cm_calc.flatten()[flat_indices].sum()

# Считаем долю
share_top_3 = (top_3_errors / total_errors) * 100
print(f"Ошибок в топ-3 парах: {top_3_errors}")
print(f"Доля топ-3 пар от всех ошибок: {share_top_3:.2f}%")

print("\nСамые частые ошибки (Истинный класс -> Предсказанный класс):")
for idx in flat_indices:
    # Превращаем плоский индекс обратно в координаты строки и столбца
    row, col = np.unravel_index(idx, cm_calc.shape)
    # Прибавляем 1, так как классы у нас от 1 до 11, а индексы от 0 до 10
    print(f"  Класс {row+1} -> Класс {col+1}: {cm_calc[row, col]} ошибок")

# 6. Строим и сохраняем красивую матрицу ошибок
print("\nСохранение графика...")
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1, 12))
disp.plot(cmap='Blues', values_format='d')
plt.title('Матрица ошибок (LogisticRegression)')
plt.tight_layout()

# Сохраняем картинку в папку figures/ (как требует README)
plt.savefig('../figures/g9_confusion_matrix.png', dpi=300)
print("Готово! Картинка сохранена в figures/g9_confusion_matrix.png")

# Показываем график на экране (можно закрыть после просмотра)
plt.show()

# 7. Вывод по гипотезе (для отчета)
print("\n--- ВЫВОД ---")
if share_top_3 > 50:
    print(f"Гипотеза ПОДТВЕРЖДЕНА (доля {share_top_3:.1f}% > 50%).")
    print("Модель путает конкретные похожие классы, а не всё подряд.")
else:
    print(f"Гипотеза ОПРОВЕРГНУТА (доля {share_top_3:.1f}% < 50%).")
    print("Ошибки распределены хаотично, простая модель не улавливает сложные зависимости.")