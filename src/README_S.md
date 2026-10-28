# День 1 — Sensorless Drive Diagnosis

## Файлы

- `common.py` — общие функции проекта.
- `example.py` — минимальный сценарий: prepare → get_model → fit → evaluate.
- `Sensorless_drive_diagnosis.txt` — исходные данные.

## Что реализовано

### `prepare()`

- принимает исходный TXT или таблицу;
- использует первые 48 столбцов как признаки;
- последний столбец — класс;
- выполняет stratified train/test split 70/30;
- `StandardScaler.fit()` вызывается только на обучающей части;
- тестовая часть преобразуется тем же scaler.

Это предотвращает утечку информации из теста.

### `evaluate()`

Считает:

- `accuracy_score`;
- `f1_score(..., average="macro")`;
- по запросу показывает `ConfusionMatrixDisplay`.

### `get_model()`

Возвращает `RandomForestClassifier`.

Параметры модели пока базовые — это заготовка Дня 1. Финальные настройки из команды должны быть внесены позже.

### `outliers()`

Ищет строки, в которых хотя бы один из 48 признаков имеет:

`|z| > 3`

Функция только определяет строки с выбросами и ничего сама не удаляет.

## Запуск

Положите `Sensorless_drive_diagnosis.txt` рядом с `example.py` и `common.py`, затем:

```bash
python example.py
```
