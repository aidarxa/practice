# Этап 6: размеры, алгоритмы и числовые параметры во время выполнения

## 1. Статус и границы

**Этап 6 выполнен 2026-11-08 на AMD Radeon 780M.** Изменение совместимых числовых
значений меняет аргументы и runtime-размеры, но не требует нового GPU-исходника,
компиляции, линковки или загрузки модуля. Проверены также смена типов, известный
typed NULL и ограниченный tagged fallback.

Это не окончательная приёмка миграции: **этапы 7–8 остаются открытыми**. Не менялись
SF10 text/byte decoding и независимые ручные SSB; новые SQL-типы и общий унарный
минус не добавлялись. Группа CASE по-прежнему принимает только integer/NULL-листья.
Численные SLA полного SQL и бюджет SQL→PCUDA ≤10 мс не объявляются достигнутыми.

Все артефакты этой работы находятся в
[`build/pcuda-stage6/run-p3mq4nfy`](../../build/pcuda-stage6/run-p3mq4nfy/).
Начальные файлы и незакоммиченные изменения сохранены в `initial/`,
`initial-sha256.json`, `initial-status.txt`, `initial.diff`. Журнал — `actions.jsonl`.

### Идея реализации простыми словами

- Размер домена CASE — число возможных листьев выражения, а не расстояние между
  текущими числами. Поэтому замена `1/1000000000` на `7/7` или NULL не увеличивает
  массивы и не создаёт другой алгоритм. Равные значения объединяются в SQL-группу.
- Локальное агрегирование применяется при структурном домене до 8 групп;
  однопоточная fused-материализация — до 256. Эти границы не зависят от значений
  параметров. Для больших результатов работает count/scan/write с runtime-ёмкостью.
- LIMIT и размеры сортировки вычисляются при исполнении. Готовый план содержит
  необходимые варианты; проверены переходы через 2048/2049 и 4096/4097, LIMIT 0,
  пустая выборка и возврат к большой выборке в той же сессии.
- Числовое состояние и факт существования группы — разные вещи. Нулевой COUNT,
  нулевой SUM и NULL-ключ больше не означают, что группы не было. Дополнительный
  presence-буфер нужен только там, где ключ или COUNT(*) не дают другого свидетельства.
- UInt64 у COUNT не делает отрицательный SQL-операнд unsigned. Для смешанных
  целых арифметика и сравнения используют внутренние знак/модуль без double;
  INT64_MIN обрабатывается без опасного отрицания. Нулевой модуль имеет один знак.

| Файлы | Ответственность |
| --- | --- |
| `include/core/result_layout.h`, `src/core/result_layout.cpp` | Единые schema-only типы результатов для execution, HAVING и сортировки |
| `include/core/aggregate_codegen.h`, `src/core/aggregate_codegen.cpp` | Общая политика typed register/block/atomic-агрегирования; COUNT не требует лишнего reduction validity |
| `include/core/aggregate_numeric.h`, `include/core/device_atomics.h` | Точные 64-битные состояния, signed/Float64 MIN/MAX и CAS-обновления без обязательного native FP64 atomicAdd |
| `src/core/visitor.cpp`, `src/core/physical_planner.cpp` | Подключение типов, shared scratch, presence и корректная материализация AVG |
| `include/core/expression_codegen.h`, `src/core/expression_codegen.cpp` | Общий тип CASE сохраняется и в tagged fallback, независимо от выбранной ветви |
| `include/core/translator.h`, `src/core/translator.cpp` | Минус перед числовым литералом нормализуется в значение до извлечения параметров |
| `include/core/operators.h` | Checked сложения/умножение оценки projection; структурный bound агрегации |
| `include/core/inline_math.h`, `include/core/tagged_numeric.h` | Единая политика mixed signed/unsigned promotion для typed и tagged путей; операции знак/модуль используют существующий dispatcher |

Слитые scan/filter/probe/aggregate и отдельные модули сохранены. Память остаётся у
runtime, данные — на GPU; новый host fallback для вычисления SQL не вводился.

## 2. Окружение и конфигурация

- AMD Radeon 780M Graphics, FP64/atomic64 подтверждены runtime; NVIDIA/Intel не проверялись.
- AdaptiveCpp `25.10.0+git.5989e1cf.20260821.branch.HEAD`; Clang/LLVM и LLD 17.0.0,
  ROCm LLVM revision `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`.
- C++17; host Release `-O3 -DNDEBUG`, тесты дополнительно `-UNDEBUG`.
- Модули/PCH: `--acpp-targets=generic --acpp-pcuda -std=c++17 -O0 -g0 -fPIC
  -Wno-narrowing -Wno-deprecated-declarations`, project include directory и
  `--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1` для этого стенда.
  Линковка отдельно: `acpp ... -fuse-ld=lld -Wl,-O0 -shared`.
- PCH подготовлен сборкой, не SQL-запросом; ABI v1 layout и entries не менялись.
  Toolchain, SDK, replay и постоянный SQL-module cache не изменялись.
- В диагностических CLI-сериях: `LC_ALL=C`, `TMPDIR=/tmp/opencode`,
  `ACPP_ADAPTIVITY_LEVEL=1`, `ACPP_ALLOCATION_TRACKING=1`.

Итоговые версии, env и **149 hashes** —
[`post-count-v2/final-inventory.json`](../../build/pcuda-stage6/run-p3mq4nfy/post-count-v2/final-inventory.json).
Ранний [`environment/inventory.json`](../../build/pcuda-stage6/run-p3mq4nfy/environment/inventory.json)
сохранён отдельно: это снимок до COUNT-fix, а не текущие хеши device headers/PCH.
Полная конфигурация — `engine/CMakeCache.txt`, `compile_commands.json`,
`CMakeFiles/*/link.txt`; module argv — `.compile.command` / `.link.command`.
PCH effective driver argv сохранён в `engine/pcuda-jit/toolchain.manifest.dryrun.json`.
Итоговые representative module compile/link/LLD effective argv —
`post-count-v2/module-inspection/0/` (ранние — `module-inspection/0/`);
это offline dry-run, не replay и не дополнительная сборка в измеряемой цепочке.

| Артефакт | SHA-256 |
| --- | --- |
| `engine/db_cli` | `ea5886f466abaa2af5ca67557b40cdf7d3320bf70d8c97d68156ed4e0a1cac38` |
| PCH, 4 682 364 B после COUNT-fix/refactoring | `485b21459e0d4eb158745f72687839de3acf3f2c009767e22c5303aea9f9a531` |
| Toolchain manifest после COUNT-fix/refactoring | `0188f746987a83fc1870e133e50b2517ceb3d52cb16c34697e8e26528dc149b2` |

`db_cli` после изменения device headers остался побайтно тем же, но PCH и manifest
изменились. Один hash CLI не доказывает неизменность JIT-кода. Ранний PCH
`4e41c1bc…` (4 658 284 B) и manifest `3ea7ccad…` относятся к серии до COUNT-fix;
их исходные полные hashes сохранены в `environment/inventory.json`.

## 3. Методика

Сначала добавлялись воспроизводящие тесты, затем исправление. GPU-ожидания
независимы от device helpers. CLI-oracle работает только после завершения процесса.
Runtime/cache-проверки используют счётчики codegen/compiler/linker/load/symbols.
Новая схема типов может потребовать физического lowering; это не совместимая смена
одних значений. Готовый fallback не требует нового исходника модуля.

Timing model — `pcuda-modules-v1`, определения из [timing.md](../../timing.md).
`summary = total_query_ms` — непрерывный C++ `steady_clock` от передачи SQL в уже
загруженную БД до готового CPU QueryResult. Внутри — frontend, необходимые build/load,
runtime-аллокации, execution и fetch; снаружи — загрузка БД, oracle и печать.
Native compile/link wall/user/system измеряет Bash `time`; суммы параллельных команд
не заменяют прямой `build_elapsed_ms`. Python не является таймером или compiler proxy.

Диагностические серии: три cold-процесса на plain/traced, первоначально по 19
запросов (§4.2), после COUNT-fix — по 27. Добавлены смены порога HAVING COUNT и
отрицательная арифметика, Float64/typed NULL и возврат к готовой Int64-схеме. Порядок:
plain→traced, traced→plain, plain→traced. В каждом — новый SQL/module/device-cache
каталог, сохранённый внутри процесса; OS/HIP/driver caches не очищались. Частоты,
температуры и desktop-нагрузки не фиксировались. Oracle работает после выхода CLI.
Clang/LLD traces сохранены; анализ выполнялся после серии.

Это небольшой correctness/diagnostic набор по SUPPLIER SF10, не Q3.3 и не performance
приёмка этапа 8. n=3 недостаточно для заявления статистически подтверждённого
ускорения. PCH preparation и startup→первый результат отдельно здесь не измерены;
native wall всего процесса с 19/27 запросами не называется временем полного старта.

## 4. Результаты

### 4.1. Исторические функциональные и quality-проверки до COUNT-fix

| Проверка | Результат | Raw относительно каталога работы |
| --- | --- | --- |
| Release CTest | **13/13** | `ctest-host-final.stdout` |
| CPU-only без поиска AdaptiveCpp | **11/11** | `cpu-ctest-final.stdout` |
| Все current/current_heavy manifest + targeted/boundaries | **88/88** | `final-regression/manifest-plain-1dhr5or1/report.json` |
| 13 SSB ×3 | **36/39 с файлами**, только известный Q1.1 ×3; warm reuse подтверждён | `final-regression/ssb-plain-u7z8tsmx/report.json` |
| Целевая CLI-серия plain/traced | **114/114**, 78 ready-запросов без codegen/build/load/symbols | `cli-series/analysis.json`, шесть исходных `report.json` |
| Native SQL numeric/runtime fixtures | **167 попыток:** 165 независимых результатов и 2 ожидаемых отказа | `expanded-v8.stdout`, `expanded-v8.stderr`, `expanded-v8/` |
| Nullable CSV на GPU | **5/5** исходных запросов, через тестовые PCUDA-колонки | конец `expanded-v8.stdout` |
| Nullable CSV/TSV SQLite oracle | **5/5** | `nullable-oracle.json` |
| Прежние MHT/OR/NULL/ABI/memory, PHT и module-cache fixtures | Успех | `test_gpu_plan.*`, `test_pcuda_primitives_sql.*`, `test_pcuda_sql_modules.*` |
| Primitives/runtime hardware | **17 323 905** проверок; 21 sparse fixture с повторами/async error | `test_pcuda_primitives_gpu.*`, `test_pcuda_gpu_runtime.*` |
| ASan/UBSan host/lowering | Успех, не device sanitizer | `ub/host-sanitized-v2/` |
| Самопроверка общих ub-tools | **25/25** | `ub-selftests.stderr` |
| Targeted mutations | **9/9 host +8/8 GPU killed**, survivors/unresolved=0 | `ub/host-mutants-v2/report.json`, `ub/gpu-mutants-v1/report.json` |
| CRAP-region | **69** выбранных definitions/instantiations, max **9,560 <10** | `ub/crapper-v2/`, `ub/numeric-crapper-v1/`, `ub/lowering-crapper-v1/` |
| Dryer | Межфайловых whole-body пар rank ≥0,8 при ≥50 tokens не найдено | `ub/dryer-v2/report.json` |
| CLI runner unit tests | **3/3** | `runner-tests-final.stderr` |
| ELF/HCF/entries | **114 модулей /6 процессов**, один native kernel на `.so`, без конфликтов разных identities | `module-inspection/report.json` |

### 4.2. Историческая диагностика до COUNT-fix, не SLA

Исходный SQL: пять агрегатов над `s_nation + 0.5`, фильтр `s_region >= 0`,
ORDER BY 1; 20 000 строк SUPPLIER. Ready changed меняет `0.5` на `-0.5`.
Repeat возвращает исходный запрос после других числовых вариантов в сессии.
Все значения ниже — **мс, медиана [min; max]**, n=3 на строку. Это исходная
19-query серия до позднего COUNT-аудита, а не итоговый замер изменённых headers.

| Режим / фаза | **summary / ready total** | Build elapsed | Execution | Fetch |
| --- | ---: | ---: | ---: | ---: |
| Plain cold | **577,320 [576,456;581,636]** | 163,780 | 412,148 | 1,445 |
| Traced cold | **583,584 [580,393;586,227]** | 165,426 | 413,806 | 1,444 |
| Plain changed | **1,765 [1,751;1,809]** | 0 | 0,326 | 1,353 |
| Traced changed | **1,778 [1,658;2,295]** | 0 | 0,332 | 1,363 |
| Plain repeat | **1,736 [1,730;1,738]** | 0 | 0,324 | 1,351 |
| Traced repeat | **1,714 [1,678;2,123]** | 0 | 0,315 | 1,347 |

| Cold-режим | Compile Σ wall/user/system | Link Σ wall/user/system | Load | **summary** |
| --- | ---: | ---: | ---: | ---: |
| Plain | 318,811 /251,030 /64,291 | 272,207 /202,321 /86,653 | 0,519 | **577,320** |
| Traced | 323,369 /255,296 /62,637 | 266,611 /204,855 /90,363 | 0,466 | **583,584** |

Каждая ячейка — отдельная медиана; суммы команд и медианы стадий не складываются
для восстановления summary. Наблюдаемая trace−plain разница median summary
**+6,264 мс**, build **+1,646 мс**; это не чистое измерение overhead traces.
Ничего не вычиталось из raw. Первое execution/fetch может содержать device JIT;
его чистое время не выделено. В traced-процессах сохранены 57 Clang и 57 LLD traces.
Полные samples и разброс всех компонентов — `cli-series/analysis.json`.
**Цели 600/200/10/9 мс прототипа здесь неприменимы.**

### 4.3. Итоговые проверки после COUNT-fix и удаления дублирующего dispatcher

Новые raw сохраняются в `post-count-v2/`, не заменяя исторические файлы выше.

| Проверка | Результат | Raw относительно каталога работы |
| --- | --- | --- |
| Release CTest | **13/13** | `post-count-v2/ctest.stdout` |
| CPU-only без поиска AdaptiveCpp | **11/11** | `post-count-v2/cpu-ctest.stdout` |
| Native SQL numeric/runtime fixtures | **202 попытки:** 200 независимых результатов и 2 ожидаемых отказа | `post-count-v2/expanded.stdout`, `post-count-v2/expanded/` |
| Nullable CSV на GPU | **5/5**, входят в 202 попытки | конец `post-count-v2/expanded.stdout` |
| Прежние MHT/OR/NULL/ABI/memory, PHT и module-cache fixtures | Успех; PHT **7/7** | `post-count-v2/test_gpu_plan.*`, `post-count-v2/test_pcuda_primitives_sql.*`, `post-count-v2/test_pcuda_sql_modules.*` |
| Primitives/runtime hardware | **17 323 905** проверок; **21** sparse fixture с повторами/async error | `post-count-v2/test_pcuda_primitives_gpu.*`, `post-count-v2/test_pcuda_gpu_runtime.*` |
| ASan/UBSan host/lowering + свежее LLVM coverage | Успех, не device sanitizer/coverage | `post-count-v2/host-sanitized/report.json` |
| Targeted mutations | **13/13 host +10/10 GPU killed**, survivors/unresolved=0 | `post-count-v2/host-mutants/report.json`, `post-count-v2/gpu-mutants/report.json` |
| CRAP-region | **108** выбранных definitions/instantiations, max **9,560 <10** | `post-count-v2/{crapper,numeric-crapper,lowering-crapper,count-crapper}/report.json` |
| Dryer | Межфайловых whole-body пар rank ≥0,8 при ≥50 tokens не найдено; scope расширен numeric headers | `post-count-v2/dryer/report.json` |
| Самопроверка общих ub-tools / CLI runner | **25/25 /3/3** | `post-count-v1/ub-selftests.stderr`, `post-count-v1/runner-tests.stderr` |
| Current/current_heavy manifest + targeted/boundaries | **88/88** | `post-count-v2/regression/manifest-plain-zd0evjlm/report.json` |
| 13 SSB ×3 | **36/39 с файлами**, только прежний Q1.1 ×3; warm reuse подтверждён | `post-count-v2/regression/ssb-plain-ruz92k7t/report.json` |
| CLI plain/traced, 6 процессов ×27 запросов | **162/162**, **108** ready-запросов без codegen/compiler/linker/load/symbols | `post-count-v2/analysis.json`, шесть исходных `report.json` |
| ELF/HCF/entries | **138 модулей /6 процессов**, один native kernel на `.so`, без конфликтов разных identities | `post-count-v2/module-inspection/report.json` |

Offline-анализ завершённых процессов, samples и проверки raw success/counters:
[`post-count-v2/analysis.json`](../../build/pcuda-stage6/run-p3mq4nfy/post-count-v2/analysis.json).
SSB raw сохраняет `success:false`; это не общий «успех 39/39» после смены oracle.

### 4.4. Итоговая диагностика после COUNT-fix, не SLA

Тот же первый SQL и те же определения cold/changed/repeat, что в §4.2; PCH уже
с исправленным числовым кодом. По три plain/traced процесса, по 27 запросов,
порядок plain→traced /traced→plain /plain→traced. Все значения — **мс,
медиана [min; max]**; компоненты измерены отдельно и не складываются в summary.

| Режим / фаза | **summary / ready total** | Build elapsed | Execution | Fetch |
| --- | ---: | ---: | ---: | ---: |
| Plain cold | **578,106 [574,518;582,678]** | 162,202 | 410,921 | 1,389 |
| Traced cold | **582,986 [580,519;584,940]** | 164,559 | 412,855 | 1,459 |
| Plain changed | **1,745 [1,740;1,782]** | 0 | 0,325 | 1,337 |
| Traced changed | **1,764 [1,660;1,766]** | 0 | 0,329 | 1,338 |
| Plain repeat | **1,702 [1,694;1,771]** | 0 | 0,316 | 1,337 |
| Traced repeat | **1,664 [1,631;1,754]** | 0 | 0,324 | 1,287 |

| Cold-режим | Compile Σ wall/user/system | Link Σ wall/user/system | Load | **summary** |
| --- | ---: | ---: | ---: | ---: |
| Plain | 316,811 /239,227 /75,115 | 265,890 /198,773 /94,693 | 0,517 | **578,106** |
| Traced | 322,061 /246,963 /78,063 | 266,957 /196,757 /96,238 | 0,333 | **582,986** |

Наблюдаемая trace−plain разница median summary **+4,880 мс**, build **+2,357 мс**;
это не выделенное чистое время overhead traces. Сохранены **69 Clang и 69 LLD**
trace-файлов. Измерения до/после изменения headers не объединены в одну выборку,
исторические raw не переписаны. n=3 и нефиксированная desktop/thermal-нагрузка не
позволяют заявлять статистически подтверждённое ускорение. First execution/fetch
может включать device JIT, отдельно он не измерен. PCH preparation и прямой
startup→первый CPU-ready результат здесь не измерялись; process wall всех 27
запросов — другая величина. **Ни SLA SQL, ни цели прототипа этим не принимаются.**

## 5. Корректность

Подтверждены и исправлены потери дробной части/знака в агрегатах, потеря NULL-группы
с нулевым COUNT, неверная общая семантика CASE при переходе на tagged fallback,
переполнение оценки размера projection. Исходные неуспешные прогоны сохранены.

- SUM/MIN/MAX имеют Int64 либо Float64 по входному выражению, COUNT — UInt64,
  AVG — Float64. Проверены scalar, local-group, direct grouped atomic и MHT item
  updates, отрицательные значения, четверти/половины, HAVING и AVG после compaction.
- Четыре fixed-output kernel family проходят все 16 схем четырёх независимых
  Int64/Float64 параметров и их повторы: predicate, arithmetic, arithmetic+CASE
  и COUNT/HAVING с отрицательным промежуточным результатом. В каждой — ровно
  8 typed специализированных модулей +1 fallback; готовые typed сохраняются.
  Смена выходного Int64/Float64 layout может создать другой вариант, а не обязана
  уложиться в девять модулей на произвольную SQL-строку.
- Known typed NULL меняет validity без новой работы; первый NULL-only может
  потребовать новой схемы при замене числом. CLI проверяет точные INT64_MIN,
  INT64_MAX, ±9007199254740993 и последующие NULL. Общее преобразование целых в
  double не вводилось. Float64 CASE сохраняет общий тип даже для выбранной Int64-ветви.
- Проверены nested CASE, разные/равные/NULL-листья, домены 256/257, нулевые и большие
  выборки, LIMIT 0 и выше числа строк. Memory guard отвергает рост на готовом плане;
  следующий малый запрос успешно использует тот же кеш без новой компиляции.
- Q1.1 не подгонялся под файл. Независимый CPU-oracle заново получил
  SQL **4486422947100**, файл/1993-only **4474878955766**, вклад 1994-01-01
  **11543991334** (`q11-reference.stdout`, повтор после COUNT-fix —
  `post-count-v2/q11-reference.stdout`). Исходный SSB `success:false` сохранён.
- Manifest проверяет исполнение всех текущих форм, но не является независимым
  oracle каждой их ячейки; эту роль выполняют tiny/CSV/targeted/SSB проверки.
- COUNT/HAVING отдельно проверяет `COUNT(NULL)>-1`, `COUNT(NULL)+-1<0`, переходы
  Int64/Float64/typed NULL и возврат к готовой typed-схеме. Host-контракты включают
  UINT64_MAX/INT64_MIN, ±9007199254740993, умножение/деление с разными знаками,
  нулевые результаты и все пары Int64/UInt64/Float64 тегов на малых числах.

Границы: только существующие числовые типы; integer/NULL GROUP BY CASE; арифметика
с представимыми промежуточными Int64-состояниями, без новой политики integer
overflow/деления на ноль.
Итоговые CRAP/coverage — host-only и неполные: 38 из 108 выбранных definitions
покрыты не полностью, включая одну неисполненную простую legacy literal-функцию. Весь `visitor.cpp`
и device-код не объявляются покрытыми; для GPU использованы hardware/oracle/mutation
проверки. Sanitizer device-путь не проверял.

### Сохранённые неуспехи

`numeric-red.*` — исходный неправильный тип дробного SUM; `numeric-v3.*` — NULL-группа
COUNT(NULL); `expanded-v4.*` — неверное деление CASE в первом fallback;
`count-valid-red.*` — COUNT=0 ошибочно считался NULL в HAVING; `contracts-red.*` —
переполнение projection; `mixed-count-red.*` — unsigned promotion ломал
`COUNT(*)>-1`. Исправления не переписывают эти исходные неуспехи.

`post-count-v1/count-crapper/report.json` сохранил неуспешный gate: отдельные
constexpr-диспетчеры операций знак/модуль дублировали существующий dispatcher и
имели CRAP ≥10. Они удалены; value-type операции используют `numericOperation`.
Пороги и LLVM coverage не менялись. Предыдущие source copies, успешные 13/10
мутаций и 202 аппаратные попытки до этого рефакторинга сохранены в том же каталоге.

Сохранены также ошибки инфраструктуры/тестов: первоначальное quoting BASE_PATH,
попытка собрать новый target до reconfigure, HAVING без требуемого parser GROUP BY,
старые assertions о UInt64/первом kernel, первое неверное ожидание отсутствия plan
lowering для новой type schema. `host-coverage-v1` не прошёл из-за невалидного
CASE-column leaf в самом тесте; `crapper-v1` выявил непокрытый validateCaseTypes.
Добавлены честные проверки допустимых/недопустимых листьев, пороги не менялись.

## 6. Выводы и решения

Пользователь согласовал Int64/Float64 для SUM/MIN/MAX, UInt64 для COUNT и Float64
для AVG, а также нормализацию минуса только перед числовым литералом. Общий унарный
оператор над колонками не добавлялся. GROUP BY CASE остаётся integer/NULL.
Численные SLA и бюджет SQL→PCUDA ≤10 мс этой работой не объявляются выполненными.

**Подтверждено:** значения не определяют статические массивы и выбор требуемого
к генерации алгоритма; runtime-ёмкости и подготовленные ветви работают без
перекомпиляции. Числовые/NULL-ошибки не маскируются совпадением со старым выводом.
Все семь пунктов этапа 6 закрыты в документированных границах.

Не заявляются: завершение этапов 7–8, поддержка SF10 текстов/byte-флагов, общий
унарный минус, новые SQL-типы, отсутствие внутреннего device JIT на ready-запросе,
аппаратная проверка других GPU или ускорение относительно корректного baseline.

## 7. Воспроизведение и оставшаяся работа

Целевые targets: `test_pcuda_runtime_contracts`, `test_pcuda_runtime_parameters`.
CLI-проверки: `test/pcuda_stage6/validate_cli.py`; quality recipes:
`test/pcuda_stage6/host-recipe.json`, `test/pcuda_stage6/gpu-recipe.json`.
Используются общие инструменты `tools/ub`, без stage-specific копий.
Doxygen-комментарии находятся в коде; генерация HTML не запускалась.

Команды из корня проекта, для нового каталога `RUN`:

```bash
RUN="$PWD/build/pcuda-stage6/reproduce"
GCC=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake -S . -B "$RUN/engine" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC -DBASE_PATH=\\\"$PWD/data/\\\"" \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DCRYSTAL_PCUDA_COMPILER=/opt/AdaptiveCpp/bin/acpp \
  "-DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=$GCC" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUN/engine" -j4
ctest --test-dir "$RUN/engine" --output-on-failure
TMPDIR=/tmp/opencode "$RUN/engine/test_pcuda_runtime_parameters" "$RUN/gpu"
python3 test/pcuda_stage6/validate_cli.py --db-cli "$RUN/engine/db_cli" --output-root "$RUN"
python3 test/pcuda_stage6/validate_cli.py --db-cli "$RUN/engine/db_cli" --output-root "$RUN" --traces
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" --mode manifest --runs 1 --output-root "$RUN"
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" --mode ssb --runs 3 --output-root "$RUN"
```

Последняя команда возвращает failure из-за сохранённого Q1.1 mismatch; остальные
ошибки игнорировать нельзя. Финальная ротация и последовательные hardware/CLI
команды сохранены в `post-count-v2/run-regression.sh`, offline-анализ —
`post-count-v2/analyze.py`. Полные quality argv, причины, stdout/stderr, source
copies и hashes находятся в `post-count-v2/*/actions.jsonl` и `.command.json`;
исторические — в `ub/` и `post-count-v1/`. Для GPU recipe
нужны `CXXFLAGS`, `CRYSTAL_PCUDA_FLAGS`, `AdaptiveCpp_DIR`, `PATH` этого SDK;
мутации выполняются только в независимых копиях после успешной baseline-сборки.
Build failure и timeout не считаются kills.

Дальнейшие задачи — отдельно этап 7 (включая согласование SF10 decoding и ручные
SSB), затем этап 8 с полной приёмкой и согласованием SQL performance targets.
Чужие артефакты не очищались; commit/push не выполнялись.
