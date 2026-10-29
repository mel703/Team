"""
Г7. Можно ли доверять диагнозу программы?
Проверяем, что accuracy итоговой модели на тестовой выборке
не ниже 95%.
"""

from common import load_data, split, prepare, evaluate, get_model


def main():
    # 1. Загружаем данные
    X, y = load_data()

    # 2. Используем общее разбиение команды
    X_train, X_test, y_train, y_test = split(X, y)

    # 3. Используем общую предобработку
    X_train, X_test = prepare(X_train, X_test)

    # 4. Создаём и обучаем модель
    model = get_model()
    model.fit(X_train, y_train)

    # 5. Оцениваем модель на тестовой выборке
    metrics = evaluate(
        model,
        X_test,
        y_test,
        name="g7_zhanna"
    )

    # 6. Выводим результаты
    accuracy = metrics["accuracy"]

    print("\nГипотеза Г7")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print("F1 по классам:", metrics["f1"].round(3))

    # 7. Проверяем критерий гипотезы
    if accuracy >= 0.95:
        print("Вывод: критерий Г7 выполнен — accuracy не ниже 95%.")
    else:
        print("Вывод: критерий Г7 не выполнен — accuracy ниже 95%.")


if __name__ == "__main__":
    main()
