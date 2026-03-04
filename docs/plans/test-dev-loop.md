# Test-Dev Loop: Automated QA Cycle

## Overview

Automated testing-development loop for the companion bot.
Three agents coordinate iterative quality improvement:

1. **Orchestrator** — manages the loop, reads reports, decides next action
2. **Tester** — runs black-box tests against the live bot (all 6 personas)
3. **Developer** — fixes bugs found in test reports, deploys changes

Max 5 rounds. Stops early if 0 P0 + 0 P1 bugs.

---

## Orchestrator Flow

```
for round in 3..7:
  1. Launch Tester Agent (subagent, foreground)
     → waits for completion
     → reads roundN_summary.md

  2. Parse summary:
     - Extract P0/P1/P2 bug counts
     - Extract average score
     - If 0 P0 AND 0 P1 → STOP (success)
     - If round == max → STOP (max rounds reached)

  3. Launch Developer Agent (subagent, foreground)
     → passes bug list from summary
     → waits for completion (including CI + deploy)

  4. Verify deploy succeeded (gh run list)

  5. Loop → next round
```

---

## Tester Agent Prompt

**Working directory**: `/Users/mingazhev/Repos/SideProjects/test-companion-bot`

**Task**:
```
Запусти тестирование бота — все 6 персон.

Шаги:
1. cd /Users/mingazhev/Repos/SideProjects/test-companion-bot
2. Запусти: ./orchestrate.sh 2>&1 | tee logs/roundN.log
   (замени N на номер текущего раунда)
3. Дождись завершения (может занять 20-30 минут)
4. Прочитай итоговый отчёт: docs/reports/round2_summary.md
5. Прочитай каждый персональный отчёт: docs/reports/round2_*.md
6. Верни структурированный результат:

РЕЗУЛЬТАТ:
- Средний балл: X.XX
- P0 багов: N (список)
- P1 багов: N (список)
- P2 багов: N (список)
- Ключевые проблемы: (краткое описание каждой)
```

**Permissions**: Bash (python3, ./orchestrate.sh, sleep), Read, Glob

**Note**: orchestrate.sh использует `claude --print` для каждой персоны
и для генерации summary. Файлы отчётов генерируются в
`docs/reports/round2_*.md`. Номер раунда в именах файлов задаётся
внутри `run_one_persona.sh` / `test_agent.md`.

---

## Developer Agent Prompt

**Working directory**: `/Users/mingazhev/Repos/SideProjects/companion-bot-core`

**Task**:
```
Ты — разработчик companion-bot-core. Тебе передан отчёт QA-тестирования.
Исправь найденные баги.

## Баги из отчёта
{вставить список P0 и P1 багов из summary}

## Процесс
1. Прочитай CLAUDE.md для понимания архитектуры
2. Для каждого бага:
   a. Найди root cause в коде
   b. Исправь
   c. Напиши/обнови unit-тесты если нужно
3. Запусти полную проверку:
   - pytest tests/unit/
   - ruff check .
   - mypy .
4. Если всё зелёное — сделай один коммит:
   git add <files>
   git commit -m "Fix <краткое описание>

   Round N fixes:
   - <bug 1>
   - <bug 2>

   Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
5. Push в main: git push origin main
6. Дождись CI: gh run list --limit 1 --json status,conclusion
   (поллить каждые 30 секунд до завершения)
7. Дождись deploy: gh run list --workflow deploy --limit 1 --json status,conclusion
   (поллить каждые 30 секунд до завершения)
8. Верни результат:

РЕЗУЛЬТАТ:
- Исправлено багов: N
- Коммит: <hash>
- CI: passed/failed
- Deploy: passed/failed
- Что не удалось исправить: (если есть)
```

**Permissions**: Read, Write, Edit, Glob, Grep, Bash (pytest, ruff, mypy, git, gh)

---

## Stop Criteria

| Condition | Action |
|-----------|--------|
| 0 P0 + 0 P1 bugs in summary | **STOP** — success |
| Round == 5 (round 7) | **STOP** — max rounds |
| Developer cannot fix remaining P0/P1 | **STOP** — manual intervention needed |
| CI or Deploy fails after 3 retries | **STOP** — infra issue |

---

## Round Tracking

After each round, orchestrator updates `docs/backlog.md` with:
- Round number
- Bugs found
- Bugs fixed
- Average score trend

---

## SSH Access (for debugging)

```bash
ssh -i ~/.ssh/gh_actions_vps_nopass root@45.150.65.119
```

Use to check bot logs if tester reports silence/timeout bugs.
