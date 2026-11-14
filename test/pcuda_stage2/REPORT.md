# Этап 2: явный физический план и контракт модулей

## 1. Статус и границы

**Этап 2 выполнен. Финальные CPU/GPU-проверки завершены.**
Дата работы — 2026-11-05. Этап не объявляется переносом GPU backend на PCUDA:
исполнение пока использует согласованный временный адаптер к существующим
SYCL-ядрам. Этапы 3–8 не выполнялись; performance-приёмка не проводилась.

Реализованы backend-independent физический план, checked size expressions,
интерпретатор, фиксированный C ABI v1 и раздельные kernel contracts/bodies.
Аллокации, порядок операций и материализация больше не являются host-программой
в сгенерированной библиотеке. Для сохранения рабочего SQL-пути временно
компилируется **одна legacy-библиотека**, содержащая отдельную обёртку каждого
фактического ядра. Это не модульный PCUDA codegen этапа 5.

### Разделение ответственности

| Компонент | Ответственность |
| --- | --- |
| `include/core/parameter_value.h`, `src/core/parameter_value.cpp` | Прежняя точная модель значений этапа 1, выделенная без изменения семантики и зависимости от SQL-плана |
| `include/core/module_abi.h`, `include/core/module_contract.h` | C-layout, стабильные статусы и проверки версии, backend, схемы, validity, диапазонов и launch |
| `include/core/size_expression.h`, `src/core/size_expression.cpp` | Неотрицательные размеры/счётчики с проверкой переполнения, не знаковая SQL-арифметика |
| `include/core/execution_plan.h`, `src/core/execution_plan.cpp` | Типизированные операции и host-интерпретатор без GPU SDK |
| `include/core/physical_planner.h`, `src/core/physical_planner.cpp` | Реестр буферов, привязки аргументов, сборка плана отдельно от его исполнения |
| `src/core/visitor.cpp` | Понижение существующих SQL-алгоритмов в операции плана и отдельные device bodies; монолит не перенесён целиком в новый файл |
| `src/core/legacy_kernel_codegen.cpp` | Только временные single-kernel обёртки и таблицы ABI, без query-аллокаций и host-алгоритма запроса |
| `include/core/legacy_plan_runtime.h`, `src/core/legacy_plan_runtime.cpp` | Удаляемый SYCL-адаптер: runtime-owned scratch/results, memory guard, копии, ожидания, проверка возможностей устройства |
| `src/core/execution.cpp` | Frontend/optimizer, legacy binding, компиляция bundle, владение загруженными библиотеками/entry points и вызов интерпретатора |

Идея, словарь и контракты описаны Doxygen-комментариями в перечисленных
заголовках. [Конфигурация](Doxyfile),
[сгенерированный HTML](../../build/pcuda-stage2/doxygen/html/index.html).

План содержит launch, ensure/fill/copy, alias для ping-pong, read-counter,
служебные counter prefix/reduce, columnar capacity, публикацию формы результата,
проверки границ, обоснованные wait, branch и ограниченный loop. Порядок блоков
задаёт зависимости на одной явной упорядоченной очереди. Операции и аргументы
можно исследовать через `describePlan`, не разбирая склеенный host C++.

Из MHT извлечены count, **пять отдельных prefix-sum запусков** и write;
aggregate finalization, projection count/scan/write, сортировки/Top-K также
представлены реальными launch и host-операциями. Fused scan/filter/probe/aggregate
сохранены; pipeline не приравнивается к единице компиляции.

Временный lowering использует явный реестр буферов и lexical capture их имён
в device body; динамические scalar captures заданы отдельно. Загрузчик не
угадывает аргументы по символам. Эффекты scratch консервативно read/write.
Полная типизированная генерация PCUDA и более точные эффекты — не результат этапа 2.

### Согласованный ABI v1

Имена нового ABI не содержат `Crystal`. Layout утверждён пользователем для
64-битного процесса; размеры и смещения закреплены C11/C++ static assertions.

| POD | Размер, байт | Содержание |
| --- | ---: | --- |
| `KernelArgumentSchemaV1` | 16 | kind, type, access, flags |
| `KernelScalarV1` | 8 | Точные Int64/UInt64/Float64 bits |
| `KernelArgumentV1` | 48 | Схема, borrowed pointer/extent, scalar payload, отдельная validity |
| `KernelLaunchConfigV1` | 40 | Трёхмерные grid/block и dynamic shared bytes |
| `KernelInvocationV1` | 80 | Версия/размер, backend, opaque stream, arguments и launch |
| `KernelModuleDescriptorV1` | 32 | Версия/размер/backend и неизменяемая схема модуля |
| `KernelErrorV1` | 208 | Статус, индекс аргумента и ограниченное диагностическое сообщение |

Целевые entry points: `db_kernel_descriptor_v1()` и
`db_kernel_launch_v1(const KernelInvocationV1*, KernelErrorV1*) -> int32_t`.
Модуль заимствует память и ставит ровно одно ядро; исключения не пересекают C-границу.
В legacy bundle применяются индексированные `db_legacy_kernel_descriptor_v1_N`
и `db_legacy_kernel_launch_v1_N`, backend — `DB_BACKEND_LEGACY_SYCL`.
Адрес SYCL queue никогда не интерпретируется как PCUDA stream.

Int64 не проходит через double, typed NULL сохраняет тип и validity.
Tagged-numeric schema допускает точный runtime tag, но **не реализует** bounded
typed/fallback-кеш или SQL runtime-параметризацию следующих этапов. В рабочем
SQL-пути literal-копия по-прежнему связывается **после оптимизации**;
`QueryTemplate::cacheKey()` не подменяет SQL-зависимый compiled-cache key.
Новый префикс старого кеша — `v41_explicit_plan_abi1`.

## 2. Окружение и конфигурация

Основной каталог артефактов:
[`build/pcuda-stage2/run-iilnso1m`](../../build/pcuda-stage2/run-iilnso1m/).
`build/` — локальное хранилище, не опубликованный переносимый архив.

- AMD Radeon 780M / `gfx1103`, Ryzen 7 7840HS; NVIDIA/Intel аппаратно не проверялись.
- AdaptiveCpp `25.10.0+git.5989e1cf.20260821.branch.HEAD`.
- Clang/LLVM 17.0.0, revision `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`;
  LLD 17.0.0, ROCm SDK `/opt/rocm_sdk_612`.
- CMake 4.4.5, Doxygen 1.18.0. Строго C++17; публичный ABI дополнительно собран как C11.
- Host Release: `-O3 -DNDEBUG`; test targets добавляют `-UNDEBUG`.
  Все семь CTest targets действительно исполняют assertions.
- Legacy JIT: штатный `acpp --acpp-targets=generic -O0 -fPIC -shared -fuse-ld=lld`,
  совместимый PCH; это ещё **не** `--acpp-pcuda`.
- Пути GCC/data/include ниже — конфигурация этого стенда. Унаследованный compiler
  с hardcoded GCC path, глобальным PCH и `/tmp`-именами **не модернизировался**:
  безопасные session-owned артефакты и build-time PCH остаются этапу 5.

Версии, существенные переменные, hashes toolchain и CMake caches:
[`environment-final.json`](../../build/pcuda-stage2/run-iilnso1m/environment-final.json).
Итоговые hashes файлов проекта —
[`final-sha256.json`](../../build/pcuda-stage2/run-iilnso1m/final-sha256.json).
Полные команды host-сборки — `engine/compile_commands.json` и `CMakeFiles/*/link.txt`;
параметры coverage — отдельный `coverage/CMakeCache.txt`.
Исходный Git HEAD — `ac5e7b9c086b2dfde8f793e4a151dd3541f8fde8`;
дерево изначально содержало значительную незакоммиченную работу.

Hashes исходников, production/coverage binaries и PCH, список trace-файлов и
сохранённые verdicts:
[`final-evidence.json`](../../build/pcuda-stage2/run-iilnso1m/final-evidence.json).
Полные argv финальных GPU-процессов и hashes на момент их запуска:
[`final-v2-gpu-commands.json`](../../build/pcuda-stage2/run-iilnso1m/final-v2-gpu-commands.json).
Compiler observer сохраняет и исходный, и effective argv: plain убирает
унаследованный `-ftime-trace`, traced добавляет Clang/LLD traces. Оптимизация и
алгоритмы между диагностическими режимами не меняются.

## 3. Методика

Это функциональная и архитектурная приёмка, не сравнение производительности.
SSB выполняются в одной CLI-сессии, все 13 запросов по три раза. Независимые
SF10-результаты нормализуются с сохранением multiplicity. Manifest запускается
с `current`/`current_heavy`, без автоматического включения `future`.
Отдельные последовательности проверяют смену значений, повтор SQL, CASE/HAVING,
projection, ORDER BY/LIMIT и точность Int64/Float64; CPU oracle работает после процесса.

Есть одинаковые plain/traced последовательности целевых запросов и tiny GPU
fixtures. Bash native `time` фиксирует wall/user/system процесса и acpp;
`LC_ALL=C`, исходные argv/stdout/stderr, Clang `-ftime-trace` и LLD `--time-trace`
сохраняются. Python организует процессы и выполняет offline-анализ, но не служит
таймером или compiler proxy. Работа Python-драйвера acpp входит в native wall.

Новый `ACPP_APPDB_DIR` создаётся на процесс и сохраняется внутри сессии;
OS/HIP/driver caches не очищаются. Cold процесса, запроса и device JIT не смешиваются.
Разность cold/warm не выдаётся за чистое время device JIT. GPU-прогоны последовательны:
runner резервирует глобальный `/dev/shm/jit_headers.h.gch` через exclusive-create и
архивирует принадлежащий ему PCH после процесса.

Timing model нового пути — **`explicit-plan-v1`**, см.
[`timing.md`](../../timing.md#этап-2-explicit-plan-v1--2026-11-05).
`generated_execute_ms` теперь измеряет host-интерпретатор через `steady_clock`,
включая runtime-работу и необходимые GPU waits; это не чистое время ядер.
Warm `codegen_ms` может включать повторное physical lowering без acpp/linker.
SSB runner требует явный `--timing-model explicit-plan-v1`, считает реальные
compiler events и проверяет нулевой warm `compile_ms`. Историческая default-модель
остаётся строгой; старые JSON и неудачные gates не переписываются.

`total_query_ms` остаётся прямым интервалом до CPU QueryResult. **`summary` как
performance-показатель здесь не агрегируется и не оценивается**: загрузка данных,
печать и oracle не включаются в query timer; process native time имеет другую
границу. Compile/link старого пути объединены. Нет новых раздельных native compile
и link SLA, полноценного process-to-first-result измерителя или чистого device-JIT
таймера. По одному traced/plain процессу недостаточно для оценки overhead/медиан;
из прежних результатов ничего не вычитается. Цели 600/200/10/9 мс ограниченного
PCUDA-стенда и бюджет SQL→PCUDA ≤10 мс этим этапом не проверялись и не изменялись.

## 4. Результаты

### Проверки окончательной реализации

| Проверка | Результат | `summary`, мс | Артефакт |
| --- | --- | --- | --- |
| Release build / CTest | Сборка успешна, 7/7 | Не измерялся | [команды](../../build/pcuda-stage2/run-iilnso1m/release-final-v2-commands.json), [CTest](../../build/pcuda-stage2/run-iilnso1m/release-final-v2-ctest.log) |
| Python stage 1 / stage 2 | 13/13 и 5/5 | Неприменимо | [команды/логи](../../build/pcuda-stage2/run-iilnso1m/python-final-commands.json) |
| ASan + UBSan, plan / ABI | Успех, leak detection включён | Не измерялся | [команды/логи](../../build/pcuda-stage2/run-iilnso1m/sanitized-final-commands.json) |
| C11 ABI / Doxygen | Успех, compiler `-Werror`, Doxygen `WARN_AS_ERROR=YES` | Неприменимо | [команды/логи](../../build/pcuda-stage2/run-iilnso1m/docs-abi-final-commands.json) |
| Stage-2 semantic mutants | 17 killed, 0 survived, 0 build-failed | Неприменимо | [итоговый raw](../../build/pcuda-stage2/run-iilnso1m/mutations-final/report.json) |
| Stage-1 mutants после extraction | 12 killed, 0 survived, 0 build-failed | Неприменимо | [итоговый raw](../../build/pcuda-stage2/run-iilnso1m/mutations-stage1-final/report.json) |
| CPU backend independence | 6 dependency-файлов без GPU SDK, optimizer без acpp runtime | Неприменимо | [проверка](../../build/pcuda-stage2/run-iilnso1m/cpu-dependencies-final.json) |
| SSB, 13 запросов ×3 | 36/39 с файлами; только Q1.1 ×3, подтверждён CPU oracle | Не оценивался | [raw report](../../build/pcuda-stage2/run-iilnso1m/sql-baseline-db5jjvoj/report.json) |
| Manifest + параметры | 85/85 checks: 84 SQL-результата и 1 ожидаемый отказ | Не оценивался | [raw report](../../build/pcuda-stage2/run-iilnso1m/cli-plain-3k_pcl9z/report.json) |
| Целевые plain / traced | 22/22 каждый: 21 oracle и 1 ожидаемый отказ | Не оценивался | [plain](../../build/pcuda-stage2/run-iilnso1m/cli-plain-lir3_g7m/report.json), [traced](../../build/pcuda-stage2/run-iilnso1m/cli-traced-4kqib6ul/report.json) |
| Tiny GPU fixtures, production | Plain и traced успешны | Не оценивался | [plain](../../build/pcuda-stage2/run-iilnso1m/gpu-fixture-251rhs0p/report.json), [traced](../../build/pcuda-stage2/run-iilnso1m/gpu-fixture-j3nv3o3e/report.json) |
| Tiny GPU fixtures, coverage host | Успех после исправления library-path окружения | Не оценивался | [raw report](../../build/pcuda-stage2/run-iilnso1m/gpu-fixture-zk94ng4i/report.json) |

Manifest охватывает 64 действующих файла / 66 SQL-операторов, плюс 19 целевых
checks; его произвольные результаты проверены на успешное исполнение и форму,
не все 66 имеют независимый oracle. Вывод ограничен 32 строками. В отдельных
22-check сериях лимит вывода 5000, поэтому результаты Top-K 2049/4096 проверены
полностью и с правильным порядком; повтор 4096 проверяет reuse и завершение.
При выделении host-цикла обнаружена граница прогресса старого local Top-K:
при K больше половины 4096-элементного chunk число кандидатов могло перестать
уменьшаться. Этот local-selector путь теперь ограничен K≤2048; вне границы
используется существующая полная GPU-сортировка, без нового GPU-алгоритма.
Tiny fixture в каждом успешном процессе проверяет 10 SQL-результатов и четыре
отрицательных сценария (ABI до side effects, memory budget, overflow, NaN reserve).

SSB runner сохранил `success:false`/exit 1 только из-за Q1.1. В новой явной модели
`compiler_cache_verified:true`: **один PCH + 13 compile/link**, warm `compile_ms=0`.
Повторное physical lowering не выдаётся за нулевой codegen. Финальные production
CLI/fixture source и binary hashes совпадают с моментом регрессии; более ранние
результаты не приписаны пересобранному бинарнику.

### Аналоги инструментов ub-mode

Установка зависимостей не требовалась. Clang AST JSON задаёт complexity;
реальные LLVM code regions — покрытие. Аналог CRAP:
`CC² × (1 − region_coverage)³ + CC`, **не statement coverage crapper**.
В семи новых компонентах учтено **191 AST definition/instantiation**, все <10,
максимум **9**:
[`quality-final/summary.json`](../../build/pcuda-stage2/run-iilnso1m/quality-final/summary.json).

| Компонент | AST definitions/instantiations | Максимум CRAP-region |
| --- | ---: | ---: |
| `parameter_value.cpp` | 9 | 7,456 |
| `size_expression.cpp` | 35 | 9 |
| `execution_plan.cpp` | 52 | 8,252 |
| `physical_planner.cpp` | 33 | 8 |
| `legacy_kernel_codegen.cpp` | 8 | 6,840 |
| `legacy_plan_runtime.cpp` | 41 | 8,385 |
| `module_contract.h` | 13 | 9 |

Полные тела ≥50 lexical tokens сравниваются с порогом similarity ≥0,8.
Внутри файлов пар нет; отдельный
[cross-file контроль 77 тел](../../build/pcuda-stage2/run-iilnso1m/quality-cross-file-final.json)
также не нашёл пар. Одинаковые template source spans не считаются копированием.
Это ограниченный whole-body аналог dryer, не поиск всех дубликатов проекта.
Большой существующий `visitor.cpp`, SDK и device kernels не входят в этот
191-function gate; 100% coverage или CRAP<10 всего движка не заявляются.

GPU adapter покрывался plain Clang host fixture, а не повреждённой AdaptiveCpp
instrumentation; реальные JIT `.so` продолжали собираться штатным acpp.
**LLVM coverage всё ещё сообщает `2 functions have mismatched data`**;
точная причина не установлена. Предупреждения сохранены в
[`quality-final.log`](../../build/pcuda-stage2/run-iilnso1m/quality-final.log),
сырые `.profraw`, merged profile, AST и coverage export сохранены. Приведённый gate
относится к доступным code-region данным, не доказывает полноту инструментации.
Существующие warnings `PARAMETER_REF` в legacy visitor также не скрыты.

### История неуспешных проверок

- TDD: [отсутствующий ABI](../../build/pcuda-stage2/run-iilnso1m/abi-red.log),
  [план](../../build/pcuda-stage2/run-iilnso1m/plan-red.log),
  [расширенные tests](../../build/pcuda-stage2/run-iilnso1m/expanded-red-tests.log),
  [конфликт local-memory layout](../../build/pcuda-stage2/run-iilnso1m/kernel-local-red.log).
  Последний тест сначала воспроизвёл принятие несовместимого тела с тем же именем;
  `sameKernelDefinition` теперь также сравнивает item и local arrays.
- [Mutations v1](../../build/pcuda-stage2/run-iilnso1m/mutations-v1/report.json)
  сохраняет первоначальные build failures. После extraction старые `.o` попадали
  в glob и давали duplicate definitions; оба драйвера теперь берут активные
  объекты из `compile_commands.json`. Build failure не считается killed.
- [Mutations v2](../../build/pcuda-stage2/run-iilnso1m/mutations-v2/report.json):
  survived для copy offset и zero-work launch. Добавлены поведенческие проверки
  ненулевых окон копии и отсутствия dispatch при grid=0; новый прогон не заменяет старый raw.
- Начальный GPU fixture отвергал block comments установленным SQL parser;
  nonce переведён в пробелы/табуляции без изменения AST. Старые каталоги
  `gpu-fixture-*` сохранены, включая неуспешные процессы.
- Прежние SSB серии использовали исторический gate `warm codegen_ms == 0`;
  их cache-gate failure не стирается после введения явного timing model.
  [Исходная серия](../../build/pcuda-stage2/run-iilnso1m/sql-baseline-1ejveefe/report.json),
  [следующая серия](../../build/pcuda-stage2/run-iilnso1m/sql-baseline-f1ea__1r/report.json).
- Первоначальный quality-анализ неверно сопоставлял C1/C2 constructor aliases,
  не учитывал destructor, AST reference nodes без `kind`, UTF-8 byte offsets и
  считал template instantiations копиями одного исходного тела. Модель исправлена
  с unit tests, а предыдущие `quality-v1`–`quality-v3` сохранены.
- Первый запуск окончательного host coverage binary не дошёл до `main`:
  [raw failure](../../build/pcuda-stage2/run-iilnso1m/gpu-fixture-cr598ibs/report.json),
  `libsqlparser.so` отсутствовал в runtime search path. У того же бинарника с
  явным `LD_LIBRARY_PATH` новый успешный прогон; исходный failure не изменён.
  [Точная команда и окружение повтора](../../build/pcuda-stage2/run-iilnso1m/coverage-host-final-retry-command.json).

## 5. Корректность и ограничения

CPU tests исполняют план через независимый host-memory backend: branches/loops,
prefix/reduce, alias snapshots, повторное исполнение, копии со смещением, пустой
launch, точные параметры/typed NULL, range и arithmetic overflow. ABI tests
проверяют version/backend/layout/schema, nullable pointers, alignment, extents и
launch volume. Все contracts проверяются до side effects; несовместимый descriptor
не приводит даже к scratch allocation.

`executePlan` ожидает завершения до возврата и при раскрутке ошибки. Host upload
sources удерживаются до wait; scalar/argument values копируются wrapper-ом в
kernel closure до возврата. Runtime проверяет FP64/atomic64, local memory и block
limits; нет предположения о warp size. Scratch/result memory управляется RAII и
High-Water Mark, memory guard учитывает фактически резидентные ёмкости, старые
typed allocations и временные буферы. Device-buffer allocation отвергает overflow
байтов и null allocation. Неперенесённые runtime/fetch helpers остаются SYCL.

Q1.1: независимый CPU oracle заново подтвердил **4486422947100** по текущему SQL
против **4474878955766** в `11.txt`; разность **11543991334** — включённый день
`1994-01-01`. Прочитано 59 986 217 fact rows:
[`q11-oracle.json`](../../build/pcuda-stage2/run-iilnso1m/q11-oracle.json),
oracle source — `test/pcuda_stage0/q11_reference.cpp`.
SQL и файлы эталонов не менялись, SSB raw должен сохранять `success:false`.

Остаются ограничения legacy backend, а не заявленный полный SQL-паритет:

- Прямой fast-path `column > NULL` отвергается с явной диагностикой; это ожидаемый
  отказ, не успешный SQL-результат. Старый float fast-filter и group-first output
  layout не расширялись до новой SQL-семантики.
- Tiny fixture использует собственный in-memory каталог/битовые карты, а не SQL DDL.
  Большой многоуровневый MHT fixture аппаратно не проверялся. В lowering есть
  явные bounds: MHT 1..256³ slots и Int32 rows/offsets, projection scan до 1024
  block sums; эти ограничения не скрываются произвольными аллокациями.
- Служебный `HostPrefix`/`HostReduce` переносит только счётчики, не query rows.
  Это сохранение существующего алгоритма, не новая host-материализация join.
- Отдельные PCUDA `.so`, value-independent module key, typed/fallback cache,
  runtime LIMIT/CASE selection, отсутствие повторного codegen при смене значений,
  build-time PCH и безопасные межсессионные temporary artifacts ещё не реализованы.
- Ручные SSB примеры не переносились; NVIDIA/Intel, полный GPU NULL/3VL и все
  возможности `future` не объявляются проверенными.

## 6. Выводы и решения

- **Факт:** host-порядок выполнения и аргументы представлены планом, а не scope
  переменных сгенерированного whole-query C++; старый `execute_query` не генерируется.
- **Факт:** backend-independent компоненты и CPU optimizer не требуют GPU SDK/runtime;
  legacy adapter ограничен явно названными файлами и не добавляет нового SYCL-алгоритма.
- **Согласовано пользователем:** временный legacy adapter, ABI layout без `Crystal`
  в именах, аналоги crapper/dryer/mutator без установки зависимостей, без commit/push.
- **Отрицательный результат:** исправный функциональный plan не доказывает ускорения.
  Native timers и traces диагностические; достижение PCUDA targets не заявляется.
- **Граница этапа:** самостоятельная загрузка/компиляция PCUDA-ядер и гарантия reuse
  обычных значений остаются этапам 3–6; начало следующего этапа требует поручения.

## 7. Воспроизведение, журнал и оставшаяся работа

Команды выполняются из корня проекта; пути соответствуют данному стенду:

```bash
cmake -S . -B build/pcuda-stage2/reproduce/engine \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  '-DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 -DBASE_PATH=\"/home/aidar/Проекты/practice/data/\"' \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DACPP_TARGETS=generic -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/pcuda-stage2/reproduce/engine -j4
ctest --test-dir build/pcuda-stage2/reproduce/engine --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test/pcuda_stage1 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test/pcuda_stage2 -p 'test_*.py' -v
doxygen test/pcuda_stage2/Doxyfile

python3 test/pcuda_stage0/sql_baseline.py \
  --db-cli build/pcuda-stage2/reproduce/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --runs 3 --timing-model explicit-plan-v1 --output-root build/pcuda-stage2/reproduce
python3 test/pcuda_stage1/validate_cli.py \
  --db-cli build/pcuda-stage2/reproduce/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --include-heavy --output-root build/pcuda-stage2/reproduce
python3 test/pcuda_stage2/validate_cli.py \
  --db-cli build/pcuda-stage2/reproduce/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --skip-manifest --output-limit 5000 --output-root build/pcuda-stage2/reproduce
python3 test/pcuda_stage2/run_fixture.py \
  --binary build/pcuda-stage2/reproduce/engine/test_gpu_plan \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp --output-root build/pcuda-stage2/reproduce
python3 test/pcuda_stage2/mutate.py \
  --build build/pcuda-stage2/reproduce/engine --output build/pcuda-stage2/reproduce/mutations \
  --cxx /opt/rocm_sdk_612/bin/clang++ \
  --flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
```

SSB-команда намеренно возвращает nonzero при Q1.1 mismatch. Последующие проверки
нужно выполнять отдельно, не связывать их с её успехом. Для диагностической пары
повторить целевой CLI и fixture с `--traces`; GPU-команды не запускать одновременно.
Sanitizer и plain-host coverage argv сохранены в
[`sanitized-final-commands.json`](../../build/pcuda-stage2/run-iilnso1m/sanitized-final-commands.json)
и [`coverage-host-final-command.json`](../../build/pcuda-stage2/run-iilnso1m/coverage-host-final-command.json).
Для запуска отдельно слинкованного coverage host нужен также `deps/lib` в
`LD_LIBRARY_PATH`; точный успешный env/argv, profile path и quality command —
в [`coverage-host-final-retry-command.json`](../../build/pcuda-stage2/run-iilnso1m/coverage-host-final-retry-command.json).

Журнал действий и причин:
[`actions.jsonl`](../../build/pcuda-stage2/run-iilnso1m/actions.jsonl).
Исходные пользовательские изменения сохранены в
[`initial/`](../../build/pcuda-stage2/run-iilnso1m/initial/),
[`initial.diff`](../../build/pcuda-stage2/run-iilnso1m/initial.diff) и
[`initial-sha256.json`](../../build/pcuda-stage2/run-iilnso1m/initial-sha256.json).
Исторические raw reports, PCH, binaries и traces не удаляются. Commit/push не выполняются.

Итоговый аудит относительно начала этапа:
[`preservation-final.json`](../../build/pcuda-stage2/run-iilnso1m/preservation-final.json).
Из 430 исходных файлов 411 побитно неизменны; 19 изменены в объявленной области
этапа, для каждого есть исходный snapshot. Начальный архив содержит 166 файлов,
для остальных 264 сохранены начальные hashes и подтверждена неизменность.
Исходные файлы не потеряны, Git HEAD и index не менялись, локальные ссылки/anchors
проверены, `git diff --check` успешен. Удалены только три созданных этой работой
Python `.pyc`, отсутствовавших в начальном inventory; пользовательские caches и
эксперименты не очищались. Изменения относительно исходного рабочего дерева, а не
общего Git HEAD: [stage2-only.diff](../../build/pcuda-stage2/run-iilnso1m/stage2-only.diff).

Критерий этапа выполнен: порядок исполнения, storage bindings и kernel arguments
определяет физический план. Оставшаяся миграция — PCUDA runtime/ресурсы (этап 3),
примитивы (4), single-kernel module codegen/cache/PCH (5), runtime-зависимые
алгоритмы/размеры (6), полный функциональный перенос и ручные примеры (7),
интегрированная приёмка (8). Их готовность не следует из закрытия этапа 2.
