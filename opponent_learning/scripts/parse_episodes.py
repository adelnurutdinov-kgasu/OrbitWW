"""
DEPRECATED — не используйте.

Этот файл создавался при первичной разведке. После обнаружения существующих
скриптов 02b_extract_steps.py / 02c_extract_launches.py / 02d_extract_choice_sets.py
и готовых датасетов data/processed/steps.csv (360K rows, 59 cols),
launches.csv (270K rows, 87 cols), choice_sets.csv (1.7 GB) — данный парсер
становится дубликатом. Используйте существующие скрипты.

Парсер сохранён только как референс корректной семантики action в Kaggle replays:
  action в записи (state=obs[t], action=action[t+1]) — то есть action, выбранный
  игроком в ответ на obs[t]. Это эмпирически проверено smoke-тестом
  (94.7% совпадения с симулятором; 100% на 1v1).
"""
raise ImportError(
    "parse_episodes.py deprecated — используйте 02b_extract_steps.py и др."
)
