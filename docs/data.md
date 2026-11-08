# Формат данных

## Задача на входе

Агент запускается в папке задачи: `python solution.py task.md`. Раскладка как у организаторов:

```
task.md                          описание: таргет, метрика, формат ответа (свободный текст)
<любые подпапки>/train.csv       обучающая выборка с таргетом
<любые подпапки>/test.csv        тест без таргета
<любые подпапки>/sample_submission.csv   пример ответа: id + таргет
<любые подпапки>/data_dictionary.csv     необязательно: столбец, тип, описание
```

`automl/task.py` находит файлы по имени в любой подпапке, а таргет, тип задачи, метрику и формат
ответа (`label` или `proba`) определяет по `task.md`, `sample_submission.csv` и данным.
Поддерживаются бинарная и многоклассовая классификация и регрессия; метрики — ROC AUC, PR AUC, F1
(macro, micro, weighted), accuracy, balanced accuracy, logloss, RMSE, MSE, RMSLE, MAE, MAPE, R² (`automl/metrics.py`).

## Ответ

`submission.csv` рядом с `task.md`: те же столбцы и тот же порядок `id`, что в
`sample_submission.csv`, без пропусков; для `label` — только классы из train.
`automl/submission.py` проверяет это перед атомарной записью.

## Полигон

Открытые датасеты, упакованные в тот же формат (`polygon/build_tasks.py`). Ответы лежат в
`private/` и агенту не видны; `polygon/score.py` считает балл по формуле кейса.

| Задача | Источник | Тип / метрика | Train / test |
|---|---|---|---|
| `amazon_access` | OpenML 4135 (Amazon Employee Access) | бинарная / ROC AUC | 22 938 / 9 831 |
| `bike_sharing` | OpenML 42712 (Bike Sharing Demand) | регрессия / RMSLE | 12 165 / 5 214 |
| `home_credit` | OpenML 45567 (Home Credit, основная таблица) | бинарная / ROC AUC | 77 000 / 33 000 |

```bash
python polygon/download.py        # parquet с OpenML в polygon/raw/ (с повторами: OpenML часто отвечает 503)
python -I polygon/build_tasks.py  # задачи в polygon/tasks/<имя>/{public,private}
```

`polygon/raw/` и `polygon/tasks/` в git не хранятся. Условия использования датасетов —
в [THIRD_PARTY.md](../THIRD_PARTY.md).

### Имитации публичных задач

`polygon/build_derived.py` строит из публичных задач организаторов задачи с известными ответами:
`wids_seen` / `wids_future` (train — годы 1–4, тест — год 5) и `unsw_shift` / `unsw_covshift`
(тест с другими долями классов / другим трафиком). Сами данные организаторов в репозиторий не
входят и не публикуются, поэтому эти задачи можно собрать только при наличии исходных файлов:

```bash
python -I polygon/build_derived.py <папка регрессии> <папка многоклассовой задачи>
```

## Записи результатов

Каждый прогон `agent/scripts/bench.py --record experiments/results` сохраняет JSON: подход, коммит,
лимит времени, режим LLM, переключатели и по каждой задаче метрику, балл и время.
`experiments/report.py` собирает из них таблицу в [results.md](results.md).
