"""
Сергей: минимальный пример — load_data → split → prepare → get_model → fit → evaluate.
Образец для скриптов команды.
"""

"""
правка: все функции берутся из одного common.py, разбиение — через split().
Путь к данным знает load_data(), поэтому DATA_PATH здесь больше не нужен.
"""

from common import load_data, split, prepare, evaluate, get_model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, X_test = prepare(X_train, X_test)

    model = get_model()
    model.fit(X_train, y_train)

    metrics = evaluate(
        model,
        X_test,
        y_test,
        name='example'
    )

    print(f"accuracy = {metrics['accuracy']:.4f}")
    print(f"f1_macro = {metrics['f1_macro']:.4f}")
    print('F1 по классам:', metrics['f1'].round(3))


if __name__ == '__main__':
    main()
