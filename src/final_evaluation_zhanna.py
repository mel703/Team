
# Итоговая оценка Random Forest

from sklearn.ensemble import RandomForestClassifier

from common import load_data, split, prepare, evaluate


def main():
    # 1. Загружаем данные
    X, y = load_data()

    # 2. Используем общее командное разбиение 70/30
    X_train, X_test, y_train, y_test = split(X, y)

    # 3. Обучаем нормализацию только на обучающей части
    X_train, X_test = prepare(X_train, X_test)

    # 4. Создаём Random Forest с выбранной настройкой
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # 5. Обучаем модель на обучающих данных
    print("Обучение итоговой модели Random Forest...")
    model.fit(X_train, y_train)

    # 6. Оцениваем модель на тестовой выборке
    metrics = evaluate(
        model,
        X_test,
        y_test,
        name="zhanna_random_forest_final"
    )

    # 7. Выводим итоговые метрики
    print("\nИТОГОВАЯ ОЦЕНКА RANDOM FOREST")
    print("-" * 45)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")

    print("\nF1 по каждому классу:")
    for class_number, f1_value in enumerate(metrics["f1"], start=1):
        print(f"Класс {class_number}: {f1_value:.4f}")

    print(
        "\nМатрица ошибок сохранена в "
        "figures/zhanna_random_forest_final.png"
    )


if __name__ == "__main__":
    main()
