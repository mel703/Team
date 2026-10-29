
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import load_data, split, prepare


def main():
    X, y = load_data()

    # Отделяем тестовую выборку и не используем её для настройки
    X_train, X_test, y_train, y_test = split(X, y)

    # Создаём внутренние обучающую и проверочную выборки
    X_subtrain, X_valid, y_subtrain, y_valid = split(X_train, y_train)

    # Подготавливаем данные
    X_subtrain, X_valid = prepare(X_subtrain, X_valid)

    results = []

    # Проверяем три значения количества деревьев
    for n_trees in [100, 200, 300]:
        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=-1
        )

        print(f"\nОбучение Random Forest: n_estimators={n_trees}")

        start = time.perf_counter()
        model.fit(X_subtrain, y_subtrain)
        train_time = time.perf_counter() - start

        y_pred = model.predict(X_valid)

        accuracy = accuracy_score(y_valid, y_pred)
        f1_macro = f1_score(y_valid, y_pred, average="macro")

        results.append({
            "Деревья": n_trees,
            "Accuracy": accuracy,
            "F1 macro": f1_macro,
            "Время, сек": train_time
        })

    print("\nСРАВНЕНИЕ ПАРАМЕТРОВ RANDOM FOREST")
    print("-" * 65)
    print(
        f"{'Деревья':<15}"
        f"{'Accuracy':>12}"
        f"{'F1 macro':>12}"
        f"{'Время, сек':>15}"
    )

    for row in results:
        print(
            f"{row['Деревья']:<15}"
            f"{row['Accuracy']:>12.4f}"
            f"{row['F1 macro']:>12.4f}"
            f"{row['Время, сек']:>15.2f}"
        )

    best = max(results, key=lambda row: row["F1 macro"])

    print("\nЛУЧШИЙ РЕЗУЛЬТАТ")
    print(f"Количество деревьев: {best['Деревья']}")
    print(f"Accuracy: {best['Accuracy']:.4f}")
    print(f"F1 macro: {best['F1 macro']:.4f}")

    print("\nТестовая выборка не использовалась для настройки.")


if __name__ == "__main__":
    main()
