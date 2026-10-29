
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import load_data, split, prepare


def main():
    # Загружаем данные
    X, y = load_data()

    # Отделяем тестовую часть.
    # Не используем её для сравнения и выбора модели.
    X_train, X_test, y_train, y_test = split(X, y)

    # Создаём внутренние обучающую и проверочную части
    X_subtrain, X_valid, y_subtrain, y_valid = split(X_train, y_train)

    # Нормализация: параметры рассчитываются только на обучающей части
    X_subtrain, X_valid = prepare(X_subtrain, X_valid)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nОбучение модели: {name}")

        start = time.perf_counter()
        model.fit(X_subtrain, y_subtrain)
        train_time = time.perf_counter() - start

        y_pred = model.predict(X_valid)

        accuracy = accuracy_score(y_valid, y_pred)
        f1_macro = f1_score(y_valid, y_pred, average="macro")

        results.append({
            "Модель": name,
            "Accuracy": accuracy,
            "F1 macro": f1_macro,
            "Время, сек": train_time
        })

    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    print("-" * 75)
    print(
        f"{'Модель':<25}"
        f"{'Accuracy':>12}"
        f"{'F1 macro':>12}"
        f"{'Время, сек':>15}"
    )

    for row in results:
        print(
            f"{row['Модель']:<25}"
            f"{row['Accuracy']:>12.4f}"
            f"{row['F1 macro']:>12.4f}"
            f"{row['Время, сек']:>15.2f}"
        )

    best = max(results, key=lambda row: row["F1 macro"])

    print("\nЛУЧШАЯ МОДЕЛЬ ПО F1 MACRO")
    print(f"Модель: {best['Модель']}")
    print(f"Accuracy: {best['Accuracy']:.4f}")
    print(f"F1 macro: {best['F1 macro']:.4f}")

    print("\nТестовая часть не использовалась для выбора модели.")


if __name__ == "__main__":
    main()
