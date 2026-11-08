# Сторонние компоненты

Код репозитория распространяется по лицензии MIT. Он использует библиотеки ниже как зависимости
(устанавливаются через `pip`, в репозиторий не входят); все их лицензии совместимы с MIT.

| Библиотека | Версия | Лицензия |
|---|---|---|
| numpy | 1.26.4 | BSD-3-Clause |
| scipy | 1.12.0 | BSD-3-Clause |
| pandas | 2.2.3 | BSD-3-Clause |
| pyarrow | 19.0.1 | Apache-2.0 |
| scikit-learn | 1.5.2 | BSD-3-Clause |
| LightGBM | 4.6.0 | MIT |
| CatBoost | 1.2.8 | Apache-2.0 |
| XGBoost | 2.1.4 | Apache-2.0 |
| openai (необязательно) | 1.68.2 | Apache-2.0 |
| matplotlib (только график) | 3.9.4 | Matplotlib License (BSD-совместимая) |

`agent/requirements.txt` — список пакетов платформы хакатона; он кладётся в архив для сдачи
как есть и шире, чем нужно агенту.

## Идеи из других проектов

`agent/automl/research/rd.py` переносит идеи (не код) из RD-Agent, Microsoft
([arXiv 2505.14738](https://arxiv.org/abs/2505.14738), [github.com/microsoft/RD-Agent](https://github.com/microsoft/RD-Agent)):
отбор идей на выборке, LLM-разработчик блока признаков, выбор компонента по UCB.

## Данные

Репозиторий не содержит данных. `polygon/download.py` скачивает открытые датасеты с OpenML;
используя их, соблюдайте условия источников:

| Задача полигона | Датасет | Исходный источник |
|---|---|---|
| `amazon_access` | [OpenML 4135](https://www.openml.org/d/4135) | Kaggle, Amazon.com — Employee Access Challenge |
| `bike_sharing` | [OpenML 42712](https://www.openml.org/d/42712) | UCI Bike Sharing Dataset / Kaggle Bike Sharing Demand |
| `home_credit` | [OpenML 45567](https://www.openml.org/d/45567) | Kaggle, Home Credit Default Risk |

Публичные задачи хакатона (энергопотребление зданий на основе WiDS Datathon 2022 и сетевые атаки
UNSW-NB15) предоставлены организаторами и не публикуются.
