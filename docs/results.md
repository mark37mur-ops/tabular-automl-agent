# Результаты

## Как считается балл

Балл платформы — относительно простого бейзлайна организаторов: для метрик «больше — лучше»
`100 · (S − B) / (1 − B)`, для ошибок `100 · (1 − S / B)`. 0 — уровень бейзлайна, 100 — идеальный
ответ. На публичном этапе две задачи: регрессия (энергопотребление зданий, RMSE, тест — следующий
год) и многоклассовая классификация (типы сетевых атак UNSW-NB15, Macro F1). Разбивку по задачам
платформа не показывает.

Полигон (`polygon/`) считает балл той же формулой, но с собственным линейным бейзлайном, поэтому
его баллы ориентировочные и с платформенными напрямую не сравниваются.

## Платформа

Платформа показывает балл ×100 (2468 = 24.68). Время — работа агента на платформе на обе задачи.

| Сдача | Что изменено | Балл | Время |
|---|---|---|---|
| v2 | исследование с ролями LLM, три семейства моделей, смесь | 18.81 | 3 ч 21 мин |
| v4e | + проверка на будущем периоде, история объекта, исправления LLM | 23.09 | 1 ч 33 мин |
| v6_shift | v4e + чистый holdout + веса строк под сдвиг признаков | 21.63 | 2 ч 5 мин |
| v5_holdout | v4e + чистый holdout | 21.65 | 1 ч 25 мин |
| v4e_bagging | v4e + усреднение дообучения по 10 сидам | 23.89 | 1 ч 53 мин |
| v7_resid_shift | v7 + страховка смеси от сдвига | 22.16 | 1 ч 43 мин |
| **v7_resid_only** | **v4e_bagging + модели на разнице с прошлым значением объекта** | **24.68** | 1 ч 35 мин |
| diag_v7_reg_only | v7, но классификация отвечает самым частым классом (диагностика) | 17.78 | 38 мин |
| v9_holdout_pick | v7 + выбор шага исследования по holdout | 19.37 | 1 ч 24 мин |
| v8_resid_models | v7 + перераспределение времени между моделями, варианты CatBoost | 22.48 | 2 ч 9 мин |

Код в этом репозитории — v7_resid_only: `python agent/scripts/make_zip.py` собирает тот же архив.
Промежуточный рейтинг при 23.09 — 29 место из 150.

### Разбивка по задачам из диагностики

Самый частый класс даёт Macro F1 около 0.06 — ниже бейзлайна, то есть 0 баллов. Если итог — среднее
двух задач с обрезкой снизу нулём, то у v7: **регрессия ≈ 35.6, классификация ≈ 13.8**
(`2 · 17.78` и `2 · 24.68 − 35.6`). Это оценка при двух допущениях, но она согласуется с другими
сдачами: у v4e_bagging регрессия ≈ 34.0, у v7_resid_shift классификация ≈ 8.8.

## Что помогло

| Изменение | Эффект | Где видно |
|---|---|---|
| Проверка на будущем периоде вместо случайных фолдов | v2 на `wids_future` — 0, v4 — 12.1 | полигон, 20 мин |
| История объекта | `wids_seen` 22.2 → 29.8 после исправления ключа | полигон, 20 мин |
| Всё вместе (v4e) | 18.81 → 23.09 | платформа |
| Усреднение дообучения по 10 сидам | 23.09 → 23.89 | платформа |
| Модели на разнице с прошлым значением | 23.89 → 24.68 | платформа |
| Исправление LLM (лимит 8000 токенов) | пустых ответов 7 из 11 → 1 из 7–9 на задачу | журналы прогонов |

Почему понадобилось усреднение: на `wids_future` одна и та же модель при разных сидах давала от 1.9
до 19.1 балла; среднее 10 сидов — 12.17 против 10.88 у одного сида в среднем, на `wids_seen` — 32.28
против 31.12 ([скрипт опыта](../scripts/experiments/refit_seed_bagging_probe.py)).
Больше деревьев при дообучении, наоборот, хуже: ×4 деревьев стоило 2.2 балла на `wids_future`.

## Что не сработало

| Гипотеза | Результат |
|---|---|
| Чистый holdout: отложенная выборка не участвует ни в одном выборе | −1.45 на платформе (21.65 против 23.09) |
| Веса строк под сдвиг признаков (adversarial validation) | не влияют: 21.63 против 21.65 |
| Удаление признаков, по которым тест отличается от train | на настоящем мультиклассе ничего не удаляет: по отдельности признаки почти не сдвинуты (AUC 0.50–0.55), сдвиг — в их сочетаниях (вместе AUC 0.88) |
| Страховка смеси от сдвига (половина веса — модели со сбалансированными классами) | −2.5 на платформе (≈ −5 на классификации) |
| Выбор шага исследования по holdout | −5.3 на платформе |
| Больше времени на модели, варианты CatBoost | +0.33 на полигоне, −2.2 на платформе |
| EM-поправка долей классов в тесте | +0.03 на полигоне; на публичном тесте оценила долю DoS в 0% при ≈5% по описанию датасета |
| Идеи LLM на часовом прогоне | 32.06 с Qwen против 32.11 без: все идеи на `wids_seen` отклонены проверкой |

## Разбор ошибок

- **Ключ объекта ломался на настоящих данных.** Поиск ключа взял колонку, постоянную для здания в
  годах train, но другую в тестовом году. Покрытие теста упало до 27% при пороге 30%, и история
  выключилась. На полигоне такой колонки не было. Нашлось на репетиции с настоящими файлами,
  исправлено: колонка в ключе не должна снижать покрытие теста.
- **Проигравшая смесь подменялась старой моделью.** Если смесь проигрывала базовой модели на
  holdout, агент отдавал ответ v1 — модель без истории и без проверки на будущем. Теперь в этом
  случае дообучается базовая модель v2.
- **LLM почти не работал.** Qwen всегда «размышляет» в скрытом поле; при лимите 2000 токенов ответ
  приходил пустым. Лимит поднят до 8000, обрезанный ответ один раз повторяется с удвоенным лимитом.
- **Полигон оптимистичнее платформы.** Имитации публичных задач обещали около 32, платформа дала 23.
  На регрессии полигон предсказывает направление изменений верно, на классификации — нет: тест
  UNSW-NB15 — другой трафик, и улучшения на `unsw_shift` на платформе не повторялись.

## Все прогоны на полигоне

Таблица ниже собирается из `experiments/results/*.json` командой `python experiments/report.py`.
Сравнивать можно только прогоны с одинаковым лимитом времени и режимом LLM. Коммиты в записях
относятся к исходному репозиторию команды.

<!-- RESULTS:START -->
_Сгенерировано `experiments/report.py` из 50 прогонов. Не править вручную._

### Все прогоны

| Дата | Подход | Ветка @ коммит | LLM | Лимит | Переключатели | amazon_access | bike_sharing | forest_cover | home_credit | news_shares | unsw_covshift | unsw_shift | wids_future | wids_seen | Среднее |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | v1 | `feature/e2e-baseline` @ `cf83ec1` | нет | 30 мин | — | 67.74 | 78.38 | — | 5.12 | — | — | — | — | — | — |
| 2026-10-20 | v2 | `feature/agent-v2` @ `9a09d15` | нет | 30 мин | — | 76.43 | 78.57 | — | 6.19 | — | — | — | — | — | — |
| 2026-10-20 | v2 | `feature/agent-v2` @ `79d1218` | нет | 30 мин | — | 76.43 | 78.55 | — | 6.34 | — | — | — | — | — | — |
| 2026-10-20 | v2 | `feature/agent-v2` @ `85269fd` | qwen/qwen3.8-27b (Groq) | 30 мин | — | 76.49 | — | — | — | — | — | — | — | — | — |
| 2026-10-20 | v2 | `feature/agent-v2` @ `fd0fc28` | qwen/qwen3.8-27b (Groq) | 30 мин | — | 76.79 | — | — | — | — | — | — | — | — | — |
| 2026-10-21 | v2 | `feature/agent-v2` @ `a7d521d` | нет | 20 мин | — | 76.43 | 78.55 | — | 6.40 | — | — | — | — | — | — |
| 2026-10-21 | v3-rd | `feature/agent-v3-rd` @ `614778f` | нет | 20 мин | screen=1, developer=1, components=1 | 72.04 | 78.55 | — | 6.01 | — | — | — | — | — | — |
| 2026-10-21 | v2 | `feature/agent-v2` @ `a7d521d` | qwen/qwen3.8-27b (Groq) | 20 мин | — | 76.86 | — | — | 6.39 | — | — | — | — | — | — |
| 2026-10-21 | v3-rd | `feature/agent-v3-rd` @ `614778f` | qwen/qwen3.8-27b (Groq) | 20 мин | screen=1, developer=1, components=1 | 76.43 | — | — | 8.25 | — | — | — | — | — | — |
| 2026-10-21 | v3-rd | `feature/agent-v3-rd` @ `7536cb3` | нет | 20 мин | screen=1, developer=1, components=1 | 76.43 | — | — | — | — | — | — | — | — | — |
| 2026-10-21 | v3-rd | `feature/agent-v3-rd` @ `7536cb3` | qwen/qwen3.8-27b (Groq) | 20 мин | screen=1, developer=1, components=1 | 75.64 | — | — | — | — | — | — | — | — | — |
| 2026-10-21 | v2 | `feature/agent-v2` @ `a7d521d` | нет | 20 мин | screen=1, developer=1, components=1 | — | — | — | — | — | — | 35.33 | 0.00 | — | — |
| 2026-10-21 | v2-ivan | `codex/agent-model-improvements` @ `7b26bba` | нет | 20 мин | screen=1, developer=1, components=1 | 75.57 | 78.58 | — | 6.16 | — | — | 35.50 | 11.49 | — | — |
| 2026-10-21 | v4 | `feature/agent-v4` @ `31ecfcb` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=1 | — | 78.58 | — | 6.06 | — | — | 35.53 | 12.12 | — | — |
| 2026-10-21 | v4 | `feature/agent-v4` @ `31ecfcb` | нет | 20 мин | screen=1, developer=1, components=1, history=0, label_shift=1 | — | — | — | — | — | — | — | 11.49 | — | — |
| 2026-10-21 | v4 | `feature/agent-v4` @ `31ecfcb` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | 35.50 | — | — | — |
| 2026-10-21 | v4 | `feature/agent-v4` @ `31ecfcb` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=1 | 75.57 | — | — | — | — | — | — | — | — | — |
| 2026-10-21 | v4b | `feature/agent-v4` @ `c5b5397` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=1 | — | — | — | — | — | — | — | 14.51 | 22.21 | — |
| 2026-10-21 | v4b | `feature/agent-v4` @ `c5b5397` | qwen/qwen3.8-27b (прокси организаторов) | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=1 | — | — | — | — | — | — | 35.53 | — | 21.95 | — |
| 2026-10-21 | v4c | `feature/agent-v4` @ `9b25f95` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=1 | — | — | — | — | — | — | — | 17.12 | 29.83 | — |
| 2026-10-21 | v2 | `feature/agent-v2` @ `a7d521d` | нет | 20 мин | — | — | 78.55 | 75.55 | — | 23.76 | — | — | — | — | — |
| 2026-10-21 | v2-text | `feature/pavel-v2-robustness` @ `0631e6a` | нет | 20 мин | — | — | — | — | — | 23.73 | — | — | — | — | — |
| 2026-10-22 | v4-debbeaa-news_shares | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | 23.71 | — | — | — | — | — |
| 2026-10-22 | v4-pavel-features-news_shares | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | 23.72 | — | — | — | — | — |
| 2026-10-22 | v4-debbeaa-forest_cover | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | 79.12 | — | — | — | — | — | — | — |
| 2026-10-22 | v4-debbeaa-wids_seen | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | — | — | 29.83 | — |
| 2026-10-22 | v4-debbeaa-wids_future | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | — | 17.12 | — | — |
| 2026-10-22 | v4-debbeaa-unsw_shift | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | 35.43 | — | — | — |
| 2026-10-22 | v4-pavel-features-forest_cover | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | 79.12 | — | — | — | — | — | — | — |
| 2026-10-22 | v4-pavel-features-wids_seen | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | — | — | 29.83 | — |
| 2026-10-22 | v4-pavel-features-wids_future | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | — | 17.12 | — | — |
| 2026-10-22 | v4-pavel-features-unsw_shift | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | 35.43 | — | — | — |
| 2026-10-22 | v4-debbeaa-amazon_access | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | 75.57 | — | — | — | — | — | — | — | — | — |
| 2026-10-22 | v4-pavel-features-amazon_access | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | 75.57 | — | — | — | — | — | — | — | — | — |
| 2026-10-22 | v4-debbeaa-bike_sharing | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | 78.58 | — | — | — | — | — | — | — | — |
| 2026-10-22 | v4-pavel-features-bike_sharing | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | 78.58 | — | — | — | — | — | — | — | — |
| 2026-10-22 | v4-debbeaa-home_credit | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | 5.99 | — | — | — | — | — | — |
| 2026-10-22 | v4-pavel-features-home_credit | `feature/pavel-feature-types` @ `898facf` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | 5.99 | — | — | — | — | — | — |
| 2026-10-22 | v4-debbeaa-ivan-baseline | `HEAD` @ `debbeaa` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | 75.54 | 78.58 | — | 6.07 | — | — | 35.44 | 16.60 | 31.48 | — |
| 2026-10-22 | v4 | `feature/agent-v4` @ `3957cfb` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | 75.57 | 78.58 | — | 5.96 | — | — | 35.50 | 17.12 | 29.83 | — |
| 2026-10-22 | v4-ivan-holdout-clean | `feature/ivan-holdout-clean` @ `5f900ca` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | 75.95 | 78.67 | — | 6.04 | — | — | 35.44 | 16.60 | 31.48 | — |
| 2026-10-22 | ivan-holdout-smoke | `feature/ivan-holdout-clean` @ `5f900ca` | нет | 5 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | — | — | 31.48 | — |
| 2026-10-22 | v4-misha-port | `feature/agent-v4` @ `ab4afcc` | нет | 20 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | 75.57 | 78.58 | — | 5.96 | — | — | 33.73 | 17.12 | 29.83 | — |
| 2026-10-22 | v4 | `feature/agent-v4` @ `3957cfb` | нет | 60 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | 34.40 | — | 29.83 | — |
| 2026-10-22 | v4 | `feature/agent-v4` @ `3957cfb` | qwen/qwen3.8-27b (прокси организаторов) | 60 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | 34.28 | — | 29.83 | — |
| 2026-10-22 | v6-shift-weights | `feature/agent-v4` @ `de20d62` | нет | 10 мин | screen=1, developer=1, components=1, history=1, label_shift=0, adv_weights=0 | — | — | — | — | — | 40.35 | 34.84 | — | — | — |
| 2026-10-22 | v6-shift-weights | `feature/agent-v4` @ `de20d62` | нет | 10 мин | screen=1, developer=1, components=1, history=1, label_shift=0, adv_weights=1 | — | — | — | — | — | 39.79 | 35.50 | — | — | — |
| 2026-10-23 | v5-holdout | `feature/agent-v4` @ `5243b10` | нет | 10 мин | screen=1, developer=1, components=1, history=1, label_shift=0, adv_weights=0 | — | — | — | — | — | — | 35.50 | 16.90 | 29.64 | — |
| 2026-10-23 | v7-bagging-holdout | `feature/agent-v4` @ `6631f78` | нет | 10 мин | screen=1, developer=1, components=1, history=1, label_shift=0, adv_weights=0 | — | — | — | — | — | — | 36.06 | 16.46 | 29.81 | — |
| 2026-10-25 | v7-resid | `feature/agent-v7-resid` @ `7f400a8` | нет | 15 мин | screen=1, developer=1, components=1, history=1, label_shift=0 | — | — | — | — | — | — | — | 18.05 | 31.23 | — |

Среднее считается только для прогонов по всем задачам. «—» — задача не запускалась. Логи и журналы агента лежат в папке с тем же именем, что и файл прогона.

### Лучший результат по каждой задаче

| Задача | Балл | Метрика | Подход | Коммит |
|---|---|---|---|---|
| amazon_access | 76.86 | roc_auc = 0.8891 | v2 | `a7d521d` |
| bike_sharing | 78.67 | rmsle = 0.27757 | v4-ivan-holdout-clean | `5f900ca` |
| forest_cover | 79.12 | macro_f1 = 0.901 | v4-debbeaa-forest_cover | `debbeaa` |
| home_credit | 8.25 | roc_auc = 0.75209 | v3-rd | `614778f` |
| news_shares | 23.76 | mae = 2352.13186 | v2 | `a7d521d` |
| unsw_covshift | 40.35 | macro_f1 = 0.65761 | v6-shift-weights | `de20d62` |
| unsw_shift | 36.06 | macro_f1 = 0.61641 | v7-bagging-holdout | `6631f78` |
| wids_future | 18.05 | rmse = 40.4372 | v7-resid | `7f400a8` |
| wids_seen | 31.48 | rmse = 30.86939 | v4-debbeaa-ivan-baseline | `debbeaa` |

### Заметки к прогонам

- `2026-10-20_v1_cf83ec1_rules`: Один LightGBM, 5 фолдов, правила разбора условия.
- `2026-10-20_v2_9a09d15_rules`: Первая v2: исследование на правилах, LGBM+CatBoost+XGBoost, бленд, контроль 15%.
- `2026-10-20_v2_79d1218_rules`: v2 после ревью: охранник, случайный поиск параметров, доп. сиды.
- `2026-10-20_v2_85269fd_groq`: Первый прогон с Qwen: упёрлись в лимит Groq 1000 выходных токенов/мин, LLM отключился.
- `2026-10-20_v2_fd0fc28_groq`: Qwen после правок промпта; MAX_TOKENS=900, IDEAS=3, MAX_CALLS=16. Из 12 идей LLM принято 4.
- `2026-10-21_v2_a7d521d_rules_20m_0335`: v2 без LLM, эталон для сравнения с v3-rd.
- `2026-10-21_v3-rd_614778f_rules_20m_scr1_dev1_cmp1_0426`: v3-rd без LLM: работает только отбор на выборке.
- `2026-10-21_v2_a7d521d_llm_20m_0454`: v2 с Qwen через Groq, эталон для v3-rd.
- `2026-10-21_v3-rd_614778f_llm_20m_scr1_dev1_cmp1_0525`: v3-rd с Qwen через Groq: отбор на выборке + разработчик + выбор компонента.
- `2026-10-21_v3-rd_7536cb3_rules_20m_scr1_dev1_cmp1_0543`: v3-rd после правки отбора (только дорогие эксперименты), без LLM.
- `2026-10-21_v3-rd_7536cb3_llm_20m_scr1_dev1_cmp1_0559`: v3-rd после правки отбора, с Qwen через Groq.
- `2026-10-21_v2_a7d521d_rules_20m_scr1_dev1_cmp1_1418`: v2 (сдано, 19 баллов) на задачах-ловушках.
- `2026-10-21_v2-ivan_7b26bba_rules_20m_scr1_dev1_cmp1_1508`: Ветка Ивана: временная валидация, полное переобучение, контракты метрик.
- `2026-10-21_v4_31ecfcb_rules_20m_scr1_dev1_cmp1_his1_lsh1_1728`: v4: Иван + гибрид + история объекта + поправка на сдвиг классов, без LLM. amazon_access убран: прогон попал на сон ноутбука (87 мин вместо 20), повтор — отдельная запись.
- `2026-10-21_v4_31ecfcb_rules_20m_scr1_dev1_cmp1_his0_lsh1_1731`: v4 без истории объекта (абляция).
- `2026-10-21_v4_31ecfcb_rules_20m_scr1_dev1_cmp1_his1_lsh0_1745`: v4 без поправки на сдвиг классов (абляция).
- `2026-10-21_v4_31ecfcb_rules_20m_scr1_dev1_cmp1_his1_lsh1_1756`: v4 без LLM: повтор amazon (первый прогон испорчен сном ноутбука).
- `2026-10-21_v4b_c5b5397_rules_20m_scr1_dev1_cmp1_his1_lsh1_1827`: v4b: дообучение с числом деревьев последнего временного фолда, без LLM.
- `2026-10-21_v4b_c5b5397_llm_20m_scr1_dev1_cmp1_his1_lsh1_1854`: v4b с Qwen организаторов, лимит 8000 токенов.
- `2026-10-21_v4c_9b25f95_rules_20m_scr1_dev1_cmp1_his1_lsh1_1925`: v4c: ключ истории сохраняет покрытие теста, без LLM.
- `2026-10-21_v2_a7d521d_rules_20m_2052`: v2 без LLM на новых задачах полигона (forest_cover, news_shares) + контроль bike_sharing. Машина 4 CPU (у платформы 8): на forest_cover не хватило времени на исследование, CatBoost успел 1 фолд, XGBoost и переобучение пропущены.
- `2026-10-21_v2-text_0631e6a_rules_20m_2139`: v2 + URL, свободный текст и даты не становятся категориями (url -> дата публикации и части адреса). Без LLM, 4 CPU; сравнивать с v2 a7d521d от 21.10 20:52 на той же машине.
- `2026-10-22_v4-debbeaa-news_shares_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_0128`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-news_shares_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_0140`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-forest_cover_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_0201`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-wids_seen_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_0202`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-wids_future_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_0203`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-unsw_shift_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_0217`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-forest_cover_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1215`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-wids_seen_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1216`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-wids_future_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1217`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-unsw_shift_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1231`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-amazon_access_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_1237`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-amazon_access_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1242`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-bike_sharing_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_1245`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-bike_sharing_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1249`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-home_credit_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_1259`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-pavel-features-home_credit_898facf_rules_20m_scr1_dev1_cmp1_his1_lsh0_1309`: Paired night-pavel benchmark; 1200s; LLM off; fixed 8 CPU affinity; same task data.
- `2026-10-22_v4-debbeaa-ivan-baseline_debbeaa_rules_20m_scr1_dev1_cmp1_his1_lsh0_1356`: Ivan holdout audit; no LLM; exact same six inputs and 1200 sec; sequential 8-thread comparison
- `2026-10-22_v4_3957cfb_rules_20m_scr1_dev1_cmp1_his1_lsh0_1437`: v4 с исправлением текстовых колонок (3957cfb), без LLM: эталон для переноса частей Миши.
- `2026-10-22_v4-ivan-holdout-clean_5f900ca_rules_20m_scr1_dev1_cmp1_his1_lsh0_1446`: Ivan holdout audit; no LLM; exact same six inputs and 1200 sec; sequential 8-thread comparison
- `2026-10-22_v4-misha-port_ab4afcc_rules_20m_scr1_dev1_cmp1_his1_lsh0_1525`: v4 + удаление признаков среды и выбор смеси после весов классов из ветки Миши. Откачено (ee3bbd5): unsw −1.8, остальное без изменений.
- `2026-10-22_v4_3957cfb_rules_60m_scr1_dev1_cmp1_his1_lsh0_1607`: v4 (3957cfb) без LLM, 60 мин.
- `2026-10-22_v4_3957cfb_llm_60m_scr1_dev1_cmp1_his1_lsh0_1710`: v4 (3957cfb) с Qwen организаторов, 60 мин: Qwen не помогает и не вредит (32.06 против 32.11 без LLM).
- `2026-10-22_v6-shift-weights_de20d62_rules_10m_scr1_dev1_cmp1_his1_lsh0_adv0_1753`: v4 + чистый holdout + веса сдвига, AGENT_ADV_WEIGHTS=0, 10 мин.
- `2026-10-22_v6-shift-weights_de20d62_rules_10m_scr1_dev1_cmp1_his1_lsh0_adv1_1808`: v4 + чистый holdout + веса сдвига, AGENT_ADV_WEIGHTS=1, 10 мин.
- `2026-10-23_v5-holdout_5243b10_rules_10m_scr1_dev1_cmp1_his1_lsh0_adv0_0226`: без бэггинга сидов, веса сдвига выключены
- `2026-10-23_v7-bagging-holdout_6631f78_rules_10m_scr1_dev1_cmp1_his1_lsh0_adv0_0238`: бэггинг сидов поверх чистого holdout, веса сдвига выключены
- `2026-10-25_v7-resid_7f400a8_rules_15m_scr1_dev1_cmp1_his1_lsh0_2311`: v7: модели на разнице с прошлым значением объекта (24.68 на платформе)
<!-- RESULTS:END -->
