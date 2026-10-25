# MetaACO для RCPSP

Исследовательский проект построения расписаний с ограниченными возобновляемыми ресурсами: baseline ACO и MetaACO с портфелем колоний. Цель — минимизация makespan в single-mode RCPSP.

## Запуск на Linux-сервере

Требуется Python 3.9 или новее. Алгоритмы, тесты, чтение XLSX и анализ результатов используют стандартную библиотеку Python; установка сторонних пакетов для этих команд не требуется. Для воспроизведения последнего локального сравнения рекомендуется Python 3.13.

Клонируйте репозиторий и перейдите в каталог проекта:

```bash
git clone https://github.com/irakli8888/metaACO.git
cd metaACO
```

Далее выполните:

```bash
python3 -m venv .venv
.venv/bin/python -B -m unittest discover -s tests -v
```

Проверочный запуск двух алгоритмов на одной небольшой задаче:

```bash
.venv/bin/python -B rcpsp/run_experiments.py \
  --instance psplib/j301_1.sm --evaluations 100 --seeds 2 \
  --output-dir experiment_results_v2/server_smoke
.venv/bin/python -B rcpsp/analyze_results.py experiment_results_v2/server_smoke
```

Общее сравнение: 15 локальных задач, 10 случайных инициализаций, два алгоритма, по 10 000 оценок расписаний (включая локальный поиск):

```bash
mkdir -p experiment_results_v2
nohup .venv/bin/python -u -B rcpsp/run_experiments.py \
  --evaluations 10000 --seeds 10 \
  --output-dir experiment_results_v2/server_10000x10 \
  > experiment_results_v2/server_10000x10.log 2>&1 &
```

Следить за журналом:

```bash
tail -f experiment_results_v2/server_10000x10.log
```

После завершения:

```bash
.venv/bin/python -B rcpsp/analyze_results.py experiment_results_v2/server_10000x10
```

Каталог каждого прогона должен быть новым. В нем сохраняются `manifest.json`, `summary.csv`, снимки исходного кода и входных данных, конфигурации, журналы и сертификаты расписаний. Анализатор создает `analysis.json`. Не читайте manifest до создания каталога запуска. Для отдельной задачи замените параметры выборки на `--instance psplib/j1201_5.sm`.

## Научный протокол

- [Протокол экспериментов](EXPERIMENT_PROTOCOL_RU.md)
- [Аудит кода](SCIENTIFIC_CODE_AUDIT_RU.md)
- [Отчет об исправлениях](SCIENTIFIC_CODE_FIXES_REPORT_RU.md)

Локальные результаты, виртуальные окружения и настройки IDE исключены из Git. Включены 15 исходных задач PSPLIB и существующая таблица эталонов `rcpsp/RCPLIB (Parameters and BKS).xlsx`. Это локальная выборка, а не полный PSPLIB. Сравнение алгоритмов целиком не определяет отдельный вклад UCB, репликации или локального поиска и не доказывает общую эффективность метода.
