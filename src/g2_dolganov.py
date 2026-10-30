
"""
Г2. Всплески тока — мусор или почерк неисправности 11?

Сравниваем две модели:
A — обучение на всех строках.
B — обучение без строк с выбросами.

Обе модели проверяются на одном тестовом наборе.
"""

from sklearn.metrics import recall_score

from common import load_data, split, prepare, get_model, outliers


def main():
    # 1. Загружаем данные и делим на train/test
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    # 2. Определяем выбросы только в обучающей выборке
    outlier_flags = outliers(X_train, X_train)
    outlier_rows = outlier_flags.any(axis=1)

    # Оставляем только строки без выбросов
    X_train_clean = X_train.loc[~outlier_rows]
    y_train_clean = y_train.loc[~outlier_rows]

    print("ГИПОТЕЗА Г2")
    print("-" * 50)
    print(f"Всего обучающих строк: {len(X_train)}")
    print(f"Строк с выбросами: {outlier_rows.sum()}")
    print(f"Строк без выбросов: {len(X_train_clean)}")

    # 3. Модель A: все обучающие строки
    X_train_scaled, X_test_scaled = prepare(
        X_train, X_test
    )

    model_all = get_model()
    model_all.fit(X_train_scaled, y_train)

    # 4. Модель B: обучение без выбросов
    # Нормализация заново обучается только на очищенном train
    X_train_clean_scaled, X_test_clean_scaled = prepare(
        X_train_clean, X_test
    )

    model_clean = get_model()
    model_clean.fit(X_train_clean_scaled, y_train_clean)

    # 5. Проверяем обе модели на одном тестовом наборе
    pred_all = model_all.predict(X_test_scaled)
    pred_clean = model_clean.predict(X_test_clean_scaled)

    # 6. Считаем Recall для класса 11
    recall_all = recall_score(
        y_test,
        pred_all,
        labels=[11],
        average=None,
        zero_division=0
    )[0]

    recall_clean = recall_score(
        y_test,
        pred_clean,
        labels=[11],
        average=None,
        zero_division=0
    )[0]

    # 7. Сравниваем результаты
    difference = recall_all - recall_clean
    difference_pp = difference * 100

    print("\nРЕЗУЛЬТАТЫ")
    print("-" * 50)
    print(f"Recall класса 11 (все строки): {recall_all:.4f}")
    print(f"Recall класса 11 (без выбросов): {recall_clean:.4f}")
    print(f"Разница: {difference_pp:.2f} процентных пункта")

    # 8. Проверяем гипотезу
    print("\nВЫВОД")
    print("-" * 50)

    if difference > 0.02:
        print(
            "Гипотеза Г2 подтверждается: "
            "без выбросов Recall класса 11 упал больше чем на 2 п.п."
        )
    else:
        print(
            "Гипотеза Г2 не подтверждается по заданному критерию: "
            "падение Recall класса 11 не превышает 2 п.п."
        )


if __name__ == "__main__":
    main()
