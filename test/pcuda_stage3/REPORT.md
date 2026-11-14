# Этап 3: PCUDA runtime и владение ресурсами

## 1. Статус и границы

**Этап 3 выполнен, финальная регрессия завершена.** Дата работы —
2026-11-05. Этот отчёт описывает именно runtime, а не завершение всей миграции.
Этапы 4–8 не выполнялись, SLA полного SQL не устанавливается.

Теперь устройство, явный stream, GPU/pinned-память, загрузка колонок и validity,
обнуление, копии и sparse-fetch принадлежат PCUDA runtime. SQL-ядра ещё SYCL:
пользователь согласовал **временный bridge на том же executor**, без второго
независимого stream или постоянной поддержки двух backend-реализаций.
ABI v1, разделение frontend/literal binding, одна legacy-библиотека и старый
SQL-зависимый кеш сохранены. Значения пока могут вызывать новую компиляцию.

### Как устроена реализация

| Файлы | Обязанность |
| --- | --- |
| `include/core/pcuda_runtime.h`, `src/core/pcuda_runtime.cpp` | Найти GPU, создать stream, проверить API/возможности и вести фактический учёт памяти |
| `include/core/memory_guard.h`, `src/core/memory_guard.cpp` | Единая политика резерва, проверка пиковых байтов и роста ёмкости |
| `include/core/device_buffer.h` | Move-only RAII, High-Water Mark, pinned-снимки для upload/download |
| `src/core/pcuda_compat.{h,cpp}` | Изолированная совместимость с установленным нижним runtime, без SYCL frontend |
| `include/core/legacy_queue.h`, `src/core/legacy_plan_runtime.cpp` | Удаляемое legacy launch-view над PCUDA executor; ресурсы больше не выделяются через SYCL |
| `include/core/pcuda_services.h`, `src/core/pcuda_services.cpp` | Заранее собранная PCUDA-компактизация sparse-результата |
| `include/core/execution.h`, `src/db/database_instance.cpp` | Владение колонками/validity/result/scratch и получение типизированного CPU-результата |
| `src/core/execution_plan.cpp`, `src/core/execution.cpp` | Время жизни host-копий/аргументов, безопасное завершение и unload библиотек |

**Словарь:** resource domain — одна сессия с одним устройством/stream и ledger;
ledger — фактически выделенные байты, не оценка SQL-размера; High-Water Mark —
сохранённая ёмкость буфера; staging — временный pinned-снимок CPU-данных;
bridge — только вид прежних launch wrappers на тот же упорядоченный executor.

Малые буферы растут в 1,5 раза, большие получают точную ёмкость. Рост не сохраняет
данные result/scratch; при отказе старый буфер остаётся жив. Guard учитывает оба
буфера в момент роста, retained typed storage, validity, временный scratch и
pinned staging. Отключение budget не отключает overflow checks. Резерв не меньше
всей памяти даёт нулевой budget, а не прежний неочевидный fallback к половине.
Другие процессы/driver allocations не измеряются: guard не обещает отсутствие OOM.

Обычный host-код собирается Release без GPU-компилятора. Только prebuilt service
и removable bridge используют acpp/PCUDA. Новые runtime/storage заголовки не
подключают `sycl/sycl.hpp`; старые device-примитивы и SQL wrappers остаются SYCL.
Идея и контракты описаны Doxygen-комментариями в коде. **Doxygen не запускался,
HTML и другие сгенерированные страницы не создавались.**

## 2. Окружение и конфигурация

Артефакты: [`build/pcuda-stage3/run-t5DU9URo`](../../build/pcuda-stage3/run-t5DU9URo/).
Это локальное хранилище `build/`, не опубликованный переносимый архив.
Снимок исходной незакоммиченной работы — `initial/`, `initial.diff` и
`initial-sha256.json`; исходный Git HEAD — `ac5e7b9c086b2dfde8f793e4a151dd3541f8fde8`.

Версии и существенные переменные сохранены в
[`environment/`](../../build/pcuda-stage3/run-t5DU9URo/environment/).
Стенд: AMD Radeon 780M, AdaptiveCpp
`25.10.0+git.5989e1cf.20260821.branch.HEAD`, Clang/LLVM/LLD 17, C++17.
NVIDIA/Intel аппаратно не проверялись.

Итоговые hashes исходников/бинарников, PCH, SQL-модулей и список traces —
[`final-evidence.json`](../../build/pcuda-stage3/run-t5DU9URo/final-evidence.json).
Проверка отсутствия full-SYCL dependency у шести перенесённых компонентов —
[`no-full-sycl.json`](../../build/pcuda-stage3/run-t5DU9URo/no-full-sycl.json).

Полная конфигурация сборки — `configure-v2.command.json`, `engine/CMakeCache.txt`,
`engine/compile_commands.json`, `engine/CMakeFiles/*/link.txt` и launcher rule.
Host Release — `-O3 -DNDEBUG`; тесты сохраняют assertions через `-UNDEBUG`.
Служебный PCUDA-код собирается при сборке приложения с
`--acpp-targets=generic --acpp-pcuda`; это не динамическое SQL-ядро.
Legacy JIT по-прежнему использует штатный acpp, `-O0 -fPIC -shared -fuse-ld=lld`
и старый PCH. Модульный compiler/build-time SQL PCH относятся к этапу 5;
существующие hardcoded compiler paths и глобальное имя PCH здесь не расширялись.

`device_name` в конфигурации или `CRYSTAL_DEVICE_NAME` выбирает GPU по
case-insensitive подстроке. Пустое имя выбирает первый перечисленный GPU;
непустое должно давать ровно одно совпадение. CPU fallback отсутствует.
Backend/platform/device перечисляются, а не предполагаются равными константам.
Пустая первая платформа не скрывает последующие устройства. Для определения
GPU используется тот же compatibility seam: публичный `pcudaDeviceProp`
установленной версии не содержит признака CPU/GPU.

## 3. Методика

Это функциональная и ресурсная приёмка. CPU double откладывает копии/обнуление
до wait, проверяет allocator pairing и позволяет воспроизвести отказ allocation,
copy, completion и небезопасный teardown без повреждения GPU. Он связан с
**настоящими** runtime/буферами, а не с копией их реализации.

Отдельный аппаратный fixture проверяет PCUDA sparse-компактизацию на пустых,
полных/неполных блоках и повторных запусках, точные значения выше 2^53 и
реальную очередь async-ошибок. Tiny SQL fixture использует runtime-owned
входные колонки/NULL bitmaps и независимые ожидаемые результаты MHT/OR/агрегатов.
Основной CLI прогоняет 13 SSB ×3, действующий manifest, последовательности
параметров и отдельные большие Top-K результаты с CPU-oracles.

GPU-процессы выполняются последовательно. Runner создаёт отдельный device cache
на процесс и сохраняет его внутри сессии; OS/HIP/driver caches не очищаются.
Legacy PCH резервируется runner-ом через exclusive-create, затем архивируется.
Завершённые прогоны не перезаписываются после изменения тестов или методики.

Таймеры процесса/компилятора — native Bash `time`, `LC_ALL=C`; SQL-интервалы —
`steady_clock`. Python только организует процессы и разбирает готовые артефакты.
Clang `-ftime-trace` и LLD `--time-trace` сохраняются отдельно от plain-контролей.
Timing model остаётся `explicit-plan-v1`: physical lowering может повторяться
на warm; это не новый запуск acpp. Отсутствие SQL-компиляции не означает отсутствие
device JIT. Полный process wall не называется `summary`.

## 4. Результаты

| Проверка | Результат | `summary`, мс | Артефакт |
| --- | --- | --- | --- |
| Release CTest | 9/9 | Не измерялся | [ctest-final.stdout](../../build/pcuda-stage3/run-t5DU9URo/ctest-final.stdout) |
| ASan/UBSan, runtime и plan | Успех; включая lifetime при ошибке D2H | Не измерялся | `runtime-sanitized.*`, `plan-sanitized.*` |
| Локальные ub-tools | 24/24 self-tests | Неприменимо | [ub-selftest-final.stderr](../../build/pcuda-stage3/run-t5DU9URo/ub-selftest-final.stderr) |
| Runtime mutants | 12/12 killed, без unresolved | Неприменимо | `ub/runtime-mutants-final/report.json` |
| Plan lifetime mutants | 3/3 killed, без unresolved | Неприменимо | [report.json](../../build/pcuda-stage3/run-t5DU9URo/ub/plan-mutants-final/report.json) |
| CRAP / duplication | Выбранные host-компоненты проходят | Неприменимо | `ub/crapper-final/report.json`, `ub/plan-crapper-final/report.json`, `ub/dryer-final/report.json` |
| 13 SSB ×3, финальный прогон | 36/39 с файлами; только известный Q1.1 ×3; compiler-cache gate успешен | Не оценивался | [report.json](../../build/pcuda-stage3/run-t5DU9URo/sql-baseline-oan3f8s6/report.json) |
| Действующий manifest + targeted | 85/85 checks: 84 SQL-результата и 1 ожидаемый отказ | Не оценивался | [report.json](../../build/pcuda-stage3/run-t5DU9URo/cli-plain-qr08q215/report.json) |
| Targeted plain/traced, по 3 процесса | В каждом 22/22: 21 oracle + 1 ожидаемый отказ; все 6 процессов успешны | Ниже отдельный диагностический контроль | [validation-summary.json](../../build/pcuda-stage3/run-t5DU9URo/validation-summary.json) |
| Tiny SQL fixture, финальные plain/traced | По 10 SQL-результатов и 4 отрицательных сценария | Не оценивался | [plain](../../build/pcuda-stage3/run-t5DU9URo/gpu-fixture-kr2wrkrg/report.json), [traced](../../build/pcuda-stage3/run-t5DU9URo/gpu-fixture-qrsnewwg/report.json) |
| Native PCUDA service, 3 новых процесса | 63 sparse-fixtures и по одному async-error check на процесс | SQL не выполняется | [gpu-runtime-final.json](../../build/pcuda-stage3/run-t5DU9URo/gpu-runtime-final.json) |
| Nullable CSV/TSV | 5/5 по независимому SQLite oracle; это CPU-проверка fixture | Неприменимо | [nullable-oracle.json](../../build/pcuda-stage3/run-t5DU9URo/nullable-oracle.json) |

Полные команды финальных GPU-процессов —
[`validation-series.json`](../../build/pcuda-stage3/run-t5DU9URo/validation-series.json).
Manifest включает 64 `current`/`current_heavy` файла и 66 SQL-операторов; `future`
не включался. Для произвольных manifest-результатов проверяются исполнение/форма,
а не независимый oracle всех 66 операторов. Отдельные targeted результаты,
включая 2049/4096 строк Top-K, проверены полностью с порядком и multiplicity.
В SSB зарегистрировано ровно **1 PCH + 13 compile/link**; все warm `compile_ms=0`.
Это reuse одинакового SQL, не обещание отсутствия компиляции при смене значений.

### Диагностическая задержка, не performance-приёмка

Для прозрачности показан первый `SELECT COUNT(*) FROM supplier WHERE s_region>1`
в шести targeted-сессиях: **3 plain и 3 traced**, порядок P/T → T/P → P/T.
Это реальный текущий SQL-путь с PCUDA ресурсами и **legacy SYCL-ядром**, не
четырёхмодульный PCUDA Q3.3. `summary` — прямой `total_query_ms` после data-ready
до CPU QueryResult. PCH/startup/data loading/oracle/печать исключены.

Все значения — **мс**, медиана `[min; max]`; load/first/fetch — отдельные медианы,
которые не складываются для получения `summary`.

| Режим | acpp compile+link native wall | Load | First execute с возможным device JIT | Fetch | **summary** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain, n=3 | 1867,188 [1845,697; 1883,495] | 0,322 | 137,667 | 0,208 | **2010,038 [1994,339; 2025,244]** |
| Traced, n=3 | 1924,234 [1921,549; 2012,090] | 0,318 | 139,300 | 0,221 | **2068,193 [2062,684; 2160,818]** |

| Режим | Compile+link native user / system | PCH native wall [min; max], вне summary | PCH user / system | Warm total [min; max] | Warm execute [min; max] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 1770,377 / 105,739 | 2813,327 [2789,049; 2843,337] | 2640,197 / 135,185 | 0,385 [0,383; 0,409] | 0,120 [0,095; 0,123] |
| Traced | 1819,881 / 99,013 | 2885,012 [2855,472; 2892,388] | 2719,025 / 148,360 | 0,290 [0,261; 0,378] | 0,117 [0,095; 0,127] |

Сырые значения, разброс **всех** колонок, ссылки на исходные reports:
[`diagnostic-latency.json`](../../build/pcuda-stage3/run-t5DU9URo/diagnostic-latency.json).
PCH и SQL compilation имеют Clang traces; SQL linking — также LLD traces.
Наблюдаемая разность медиан summary trace−plain **+58,155 мс** не является чистым
overhead диагностики; n=3 недостаточно для статистического вывода. Warm-разница
также не выдаётся за ускорение traces. Ничего из исторических чисел не вычитается.
Чистые native compile и link раздельно старый путь не измеряет; trace intervals
их не заменяют. Startup/data loading и process→первый CPU-ready здесь отдельно
не измерены; сохранённый native process wall включает всю сессию и teardown.

Согласованные цели 600/200/10/9 мс ограниченного PCUDA-прототипа и бюджет
SQL→PCUDA ≤10 мс здесь не принимаются и не изменяются.

## 5. Корректность и ограничения

### Ошибки и время жизни

Пользователь согласовал изолированный compatibility-адаптер: в установленном
AdaptiveCpp публичный `pcudaStreamSynchronize` теряет `inorder_queue::wait()`
result, а `pcudaGetLastError` не видит все типы ошибок. `pcuda_compat.cpp` проверяет
native result и очередь runtime-ошибок, не подключая SYCL frontend. Исходники
этого поведения и их hashes сохранены в `environment/`. Эти нижние API требуют
перепроверки при обновлении toolchain; его установка не менялась. Очередь ошибок
в этой версии process-wide: ошибки сообщаются консервативно, без обещания
независимой атрибуции конкурентным сессиям.

При неподтверждённом завершении teardown прекращает процесс, а не освобождает
используемую память/код. Это намеренная fail-closed политика. Обычная ошибка
запроса с успешным последующим drain остаётся исключением, не обязательным abort.
Host download destinations защищены до завершения даже при первом неуспешном
wait; pending upload snapshots живут внутри исполнения. Scalar/argument values
копируются wrapper-ом в kernel closure до возврата; GPU не получает указатель
на обычный host-массив аргументов. FP64/atomic64, shared memory и размеры launch
проверяются по устройству, без предположений о warp size.

### Сохранённая проблема storage

Первый CLI-контроль остановился до запросов: строгая проверка `rows×4` выявила
в SF10 текстовые поля и byte-флаги при Int32-каталоге. Пользователь согласовал
**сохранить legacy raw-prefix/zero-tail**, не добавляя SQL-типы на этапе runtime,
и перенести исправление в подходящий функциональный этап. Задача явно добавлена
в этап 7 `todo.md`. Это ограничение не выдаётся за корректную поддержку текста.
Отдельные CPU-тесты фиксируют прежнее представление и проверяют overflow.

### Границы доказательств

Host LLVM coverage не является device coverage. CRAP/dryer относятся к указанным
runtime/буферам и новому completion-guard, не ко всему `visitor.cpp`, GPU-примитивам
или полному SQL-движку. Непокрытые error/abort ветви и неполное region coverage
публикуются в raw reports. Mutation score относится только к 15 перечисленным
семантическим мутациям, не является доказательством отсутствия всех ошибок.
Реальный разрушительный GPU fault/driver OOM не провоцировался; ошибки API/native
completion моделируются, а process-wide async queue проверяется на настоящем runtime.

В runtime scope — **102 AST definitions/instantiations**, максимум CRAP-region **7**;
у трёх новых completion-guard функций максимум **3,042**. Region coverage неполное:
например, fail-fast ветвь дочернего abort-процесса не даёт parent LLVM profile.
Отсутствующих coverage identities в выбранном scope нет; LLVM mismatch warnings
при этой коллекции не было. Межфайловый dryer по реальным runtime/plan телам не
нашёл пар ≥0,8 при минимуме 50 токенов:
[report.json](../../build/pcuda-stage3/run-t5DU9URo/ub/dryer-cross-file/report.json).

Q1.1 снова проверен отдельным CPU C++ oracle: по текущему SQL
**4486422947100**, файл `11.txt` содержит **4474878955766**; разница
**11543991334** относится к включённому `1994-01-01`. Прочитано 59 986 217 fact rows:
[`q11-oracle.json`](../../build/pcuda-stage3/run-t5DU9URo/q11-oracle.json).
SQL и эталоны не менялись, SSB report намеренно сохраняет `success:false`/exit 1.

### История отрицательных проверок

- `runtime-red.stderr`: тест требовал ещё отсутствовавший PCUDA API.
- `build-v1.stderr`: неверное shell-экранирование `BASE_PATH` в конфигурации стенда;
  исправлены параметры запуска, не ослаблен compiler.
- `gpu-fixture-szgsf6rq`: private handler API; заменён публичной nonblocking
  проверкой статуса event, без ожидания после каждого ядра.
- `sql-baseline-8kpq2ksl`: 0/39, отказ загрузки смешанного storage до запросов;
  последующее согласование не превращает его в успешный прогон.
- `ub/crapper-v1`, `ub/dryer-v1`: Clang синтезировал тело defaulted move по адресу
  токена `default`. Локальный AST-анализ исправлен с самостоятельным failing test;
  production move-операции не переписывались ради метрики.
- `ub-selftest-v2.stderr`: race при чтении исчезнувшего `/proc/PID/stat` в timeout
  self-test; исчезнувший процесс теперь корректно считается завершённым.
- `ub/plan-mutants/report.json`: 1 killed / 2 survived. Усилена проверка порядка
  копия→wait; последующий 3/3 не переписывает первоначальный результат.
- `cli-plain-1ice5oz8`: внешнее ограничение harness 120 секунд прервало manifest.
  Незавершённые команды/PCH сохранены, этот каталог не считается успешной приёмкой.

## 6. Выводы и согласованные решения

- Ресурсами и порядком операций управляет PCUDA runtime, а не исходник запроса.
- Временный SYCL bridge и нижний checked-wait seam согласованы пользователем;
  ABI v1 не меняется. Перенос device bodies/codegen остаётся следующим этапам.
- Storage raw-view сохранён по отдельному решению пользователя, проблема поставлена
  в roadmap, а не скрыта изменением данных или эталонов.
- Полный SQL cold/warm SLA, build-time SQL PCH, отдельные `.so`, typed/fallback
  cache и отсутствие codegen при смене значений не реализовывались.
- Критерий этапа подтверждён в согласованных границах: runtime управляет памятью
  и порядком независимо от query source; ранние выходы завершают использование
  ресурсов до их освобождения, неподтверждённый teardown не продолжается.

## 7. Воспроизведение и оставшаяся работа

Сборка стенда воспроизводится командой из `configure-v2.command.json`, затем:

```bash
cmake --build build/pcuda-stage3/run-t5DU9URo/engine -j4
ctest --test-dir build/pcuda-stage3/run-t5DU9URo/engine --output-on-failure
python3 -m unittest discover -s tools/ub/tests -v
python3 test/pcuda_stage0/sql_baseline.py \
  --db-cli build/pcuda-stage3/run-t5DU9URo/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --runs 3 --timing-model explicit-plan-v1 --output-root build/pcuda-stage3
python3 test/pcuda_stage1/validate_cli.py \
  --db-cli build/pcuda-stage3/run-t5DU9URo/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --include-heavy --output-root build/pcuda-stage3
python3 test/pcuda_stage2/validate_cli.py \
  --db-cli build/pcuda-stage3/run-t5DU9URo/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --skip-manifest --output-limit 5000 --output-root build/pcuda-stage3
```

SSB-команда возвращает nonzero для известного Q1.1 mismatch. GPU runner-ы нельзя
запускать одновременно из-за ещё не перенесённого SQL PCH. Для диагностической
пары повторить targeted/fixture с `--traces`; для серий чередовать порядок.
Не ограничивать полный manifest внешним timeout 120 секунд.

Локальные ub recipes: `runtime-recipe.json`, `plan-recipe.json`; используются
общие `tools/ub/coverage.py`, `crapper.py`, `dryer.py`, `mutator.py`. Полные argv,
scope/hashes, native compiler/test stdout/stderr и `actions.jsonl` находятся
в каждом каталоге `ub/`. GCC/SDK include paths передаются через `--flag`, не
зашиты в эти recipes. `runtime_analysis.cpp` только включает реальные `.cpp` и
тестовые instantiations для межфайлового AST, не копирует алгоритмы.

Общий журнал действий и причин —
[`actions.jsonl`](../../build/pcuda-stage3/run-t5DU9URo/actions.jsonl).
Изменения относительно начальной рабочей копии и аудит её сохранности —
`stage3-only.diff` и `preservation-final.json` в том же каталоге артефактов.
Commit/push, очистка чужих артефактов и генерация документации не выполнялись.
Следующие этапы автоматически не начинаются.
