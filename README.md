# VK-LSVD: гибридные рекомендательные модели

## Цель проекта

Оценить, как контентные сигналы влияют на качество рекомендаций по сравнению с чистой коллаборативной фильтрацией.

## Что реализовано

- базовые модели коллаборативной фильтрации:
  - `popularity`
  - `item_knn_cf`
  - `item_jaccard_cf`
- контентная и гибридная модели:
  - `content_knn`
  - `hybrid_knn`
- многократный запуск экспериментов на нескольких `seed`
- метрики ранжирования:
  - `precision@k`
  - `recall@k`
  - `map@k`
  - `ndcg@k`
  - `hit_rate@k`
  - `coverage@k`
- сохранение результатов отдельных запусков и итоговой сводки

## Структура проекта

```text
data/
  raw/                   локальные файлы VK-LSVD
reports/
  vklsvd_final/          итоговый отчёт и результаты экспериментов
src/
  vk_recsys_hybrid/      код моделей и бенчмарка
run_vklsvd_benchmark.py  основной скрипт запуска
```

## Какие данные нужны

Для запуска ожидается, что датасет VK-LSVD лежит в папке:

```text
data/raw/VK-LSVD/
```

В текущем проекте используются:

- недели взаимодействий из `subsamples/ur0.01_ir0.01`
- `metadata/items_metadata.parquet`
- `metadata/users_metadata.parquet`
- `metadata/item_embeddings.npz`

## Быстрый запуск

Запуск основного эксперимента:

```powershell
.venv\Scripts\python.exe run_vklsvd_benchmark.py `
  --data-root data/raw/VK-LSVD `
  --subset ur0.01_ir0.01 `
  --train-start-week 20 `
  --train-end-week 24 `
  --top-k 10 `
  --min-user-interactions 5 `
  --min-item-interactions 5 `
  --min-watch-ratio 0.8 `
  --max-eval-users 100 `
  --seeds 42 43 `
  --embedding-dims 16 64 `
  --reports-dir reports/vklsvd_final
```

## Что получается на выходе

Итоговые артефакты лежат в `reports/vklsvd_final/`:

- `benchmark_runs.csv`
- `benchmark_summary.csv`
- `benchmark_summary.json`
- `dataset_stats.json`
- `VK_LSVD_report.docx`

## Комментарий по VK-LSVD

VK-LSVD включает:

- пользовательские метаданные
- метаданные объектов
- взаимодействия пользователей с видео
- готовые `item_embeddings`
- публичные подвыборки для экспериментов

Источник: [VK-LSVD на Hugging Face](https://huggingface.co/datasets/deepvk/VK-LSVD).

