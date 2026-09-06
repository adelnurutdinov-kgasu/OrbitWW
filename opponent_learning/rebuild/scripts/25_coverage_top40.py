#!/usr/bin/env python3
"""
25_coverage_top40.py — coverage анализ топ-40 в существующих эпизодах.

ИДЕЯ
────
У нас 627+ скачанных JSON. У каждого есть info.TeamNames — кто играл.
Топ-40 актуальных игроков известны (вписаны ниже из лидерборда).
Сопоставляем — сколько эпизодов с каждым из них уже есть, и какие
эпизоды содержат ОБЕ команды из топ-40 (самые ценные данные).

ВЫХОД
─────
  rebuild/data/top40_coverage.csv
  rebuild/data/top40_episode_ids.json — episode IDs с участием топ-40
  rebuild/reports/top40_coverage_report.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE       = Path(__file__).resolve().parent
REBUILD    = HERE.parent
OL_DIR     = REBUILD.parent
RAW_DIR    = OL_DIR / "data" / "raw"
DATA_DIR   = REBUILD / "data"
REPORT_DIR = REBUILD / "reports"
for d in (DATA_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Топ-40 от лидерборда (последняя проверка)
TOP_40 = [
    'Isaiah @ Tufa Labs', 'Jake Will', 'TonyK', 'Hober Malloc',
    'Felix M Neumann', 'flg', 'Audun Ljone Henriksen', 'Boey', 'Ender',
    'Xiangyu Liu', 'Luca', 'One Man Wrecking Machine', 'M & J & M.ver2',
    'Azat Akhtyamov', 'moriiiiiiiiim', 'Gregor Lied', 'Vadasz & Ascalon',
    'dragon warrior', '213tubo', 'Luke Li', 'Slawek Biel', 'Orbit Goblins',
    'Scool', 'LuckyXC&me', 'Yuki Okumura', 'Sheeesh---', 'Roche Overflow',
    'Artem', 'jonathan breitgand', 'kosmostars @ pseudolabeling',
    'Alan C52', 'Tufaben', 'This waiting will kill me!', 'bowwowforeach',
    'vkhydras', 'skalermo', 'typeIIIfairy', 'XtraLearning', 'Ebi', 'ma name',
]


def main():
    top_set = set(TOP_40)
    files = sorted(p for p in RAW_DIR.glob('*.json')
                   if p.name != 'episodes_index.json' and p.stem.isdigit())
    print(f'Файлов: {len(files)}')

    all_teams = Counter()
    eps_with_any_top40 = []     # хотя бы один из топ-40 играл
    eps_with_all_top40 = []     # ВСЕ участники из топ-40 (целевой dataset)
    team_eps = {}                # name → list of episode_ids

    for i, fp in enumerate(files):
        try:
            with open(fp) as f:
                ep = json.load(f)
        except Exception:
            continue
        teams = ep.get('info', {}).get('TeamNames') or []
        n_players = len(teams)
        matches = [t for t in teams if t in top_set]
        for t in teams:
            all_teams[t] += 1
            team_eps.setdefault(t, []).append(fp.stem)
        if matches:
            eps_with_any_top40.append(fp.stem)
        if n_players and len(matches) == n_players:
            eps_with_all_top40.append(fp.stem)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)}')

    print(f'\nКоманд найдено всего: {len(all_teams)}')
    print(f'\n══ Coverage топ-40 ══')
    found = [(name, all_teams[name]) for name in TOP_40 if all_teams.get(name, 0) > 0]
    missing = [name for name in TOP_40 if all_teams.get(name, 0) == 0]

    print(f'Из топ-40 найдено в наших данных: {len(found)}')
    print(f'  не найдено: {len(missing)}')

    print(f'\nЭпизодов с участием хоть кого из топ-40:        {len(eps_with_any_top40)}')
    print(f'Эпизодов где ВСЕ участники из топ-40 (целевые): {len(eps_with_all_top40)}')

    if found:
        print(f'\nТоп-40 найденные (по числу эпизодов):')
        for name, n in sorted(found, key=lambda x: -x[1])[:20]:
            print(f'  {name:<40}  {n:>4} эпизодов')

    if missing:
        print(f'\nТоп-40 ОТСУТСТВУЮТ:')
        for name in missing:
            print(f'  • {name}')

    print(f'\n══ Топ-15 наиболее частых команд (★ = в топ-40) ══')
    for name, n in all_teams.most_common(15):
        mark = '★' if name in top_set else ' '
        print(f'  {mark} {name:<45}  {n:>4}')

    # Сохранение
    import csv
    with open(DATA_DIR / 'top40_coverage.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['team_name', 'n_episodes', 'in_top40'])
        for name, n in all_teams.most_common():
            w.writerow([name, n, int(name in top_set)])

    with open(DATA_DIR / 'top40_episode_ids.json', 'w') as f:
        json.dump({
            'any_top40':  eps_with_any_top40,
            'all_top40':  eps_with_all_top40,
            'team_eps':   {k: v for k, v in team_eps.items() if k in top_set},
        }, f)

    # Markdown
    md = []
    md.append('# Coverage топ-40 актуальных игроков в наших данных\n\n')
    md.append(f'**Файлов проанализировано**: {len(files)}\n')
    md.append(f'**Команд всего**: {len(all_teams)}\n')
    md.append(f'**Топ-40 найдено**: {len(found)} / 40\n\n')
    md.append(f'## Coverage metrics\n\n')
    md.append('| Metric | Value |\n|---|---|\n')
    md.append(f'| Эпизодов с хоть кем из топ-40 | {len(eps_with_any_top40)} |\n')
    md.append(f'| Эпизодов где ВСЕ участники топ-40 | {len(eps_with_all_top40)} |\n')

    md.append('\n## Топ-40 в наших данных\n\n')
    md.append('| name | эпизодов |\n|---|---|\n')
    for name, n in sorted(found, key=lambda x: -x[1]):
        md.append(f'| {name} | {n} |\n')

    if missing:
        md.append(f'\n## Топ-40 ОТСУТСТВУЮТ ({len(missing)})\n\n')
        for name in missing:
            md.append(f'- {name}\n')
        md.append('\nЭти игроки в наших скачанных реплеях не появлялись. Чтобы получить\n')
        md.append('их эпизоды, нужны их submission IDs (найти вручную на странице команды).\n')

    md.append('\n## Использование\n\n')
    md.append('Эпизоды с участием топ-40 в `top40_episode_ids.json`:\n')
    md.append('- `any_top40` — для расширенного датасета (хотя бы один топ играл)\n')
    md.append('- `all_top40` — целевой dataset (топ vs топ)\n\n')
    md.append('Для обучения модели можно использовать эти списки как фильтр на choice_sets.csv.\n')

    (REPORT_DIR / 'top40_coverage_report.md').write_text(''.join(md))
    print(f'\n✓ CSV: {DATA_DIR / "top40_coverage.csv"}')
    print(f'✓ JSON: {DATA_DIR / "top40_episode_ids.json"}')
    print(f'✓ Отчёт: {REPORT_DIR / "top40_coverage_report.md"}')


if __name__ == '__main__':
    main()
