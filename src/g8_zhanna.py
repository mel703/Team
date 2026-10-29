"""
Г8. Понимает ли модель, когда сомневается?

Сравниваем медианную уверенность модели
для правильных и неправильных ответов.
"""

import numpy as np

from common import load_data, split, prepare, get_model


def main():
    # 1. Загружаем данные и используем общее разбиение
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    # 2. Применяем общую предобработку
    X_train, X_test = prepare(X_train, X_test)

    # 3. Обучаем итоговую модель
    model = get_model()
    model.fit(X_train, y_train)

    # 4. Получаем предсказания и вероятности классов
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # 5. Уверенность — максимальная вероятность среди классов
    confidence = probabilities.max(axis=1)

    # 6. Разделяем правильные и неправильные ответы
    correct = predictions == np.asarray(y_test)

    correct_confidence = confidence[correct]
    wrong_confidence = confidence[~correct]

    print("\nГипотеза Г8")

    print(f"Правильных ответов: {correct.sum()}")
    print(f"Неправильных ответов: {(~correct).sum()}")

    if len(correct_confidence) == 0 or len(wrong_confidence) == 0:
        print(
            "Недостаточно данных для сравнения: "
            "в одной из групп нет объектов."
        )
        return

    # 7. Считаем медианную уверенность в каждой группе
    median_correct = np.median(correct_confidence)
    median_wrong = np.median(wrong_confidence)

    # Разница в процентных пунктах
    difference_pp = (median_correct - median_wrong) * 100

    print(
        "Медианная уверенность правильных ответов: "
        f"{median_correct * 100:.2f}%"
    )
    print(
        "Медианная уверенность неправильных ответов: "
        f"{median_wrong * 100:.2f}%"
    )
    print(
        "Разница: "
        f"{difference_pp:.2f} процентного пункта"
    )

    # 8. Проверяем критерий Г8
    if difference_pp >= 30:
        print(
            "Вывод: критерий Г8 выполнен — "
            "разница не меньше 30 процентных пунктов."
        )
    else:
        print(
            "Вывод: критерий Г8 не выполнен — "
            "разница меньше 30 процентных пунктов."
        )


if __name__ == "__main__":
    main()
