# Этап 5: модульные SQL codegen, compiler и кеш PCUDA

## 1. Статус и границы

**Этап 5 выполнен 2026-11-07.** Реальный SQL-путь через `db_cli` использует
отдельные PCUDA-модули, runtime-параметры ABI v1 и кеш загруженного кода только
внутри процесса. При совместимой смене значений готовый план не понижается заново,
компилятор/линкер и `dlopen/dlsym` не вызываются. Это подтверждено счётчиками, а не
только коротким временем повторов.

Пользователь разрешил включить необходимые предпосылки этапа 6: без них
value-free ключ был бы некорректен для LIMIT и GROUP BY CASE. **Этап 6 целиком,
этапы 7–8 и весь перенос не закрыты.** Не перенесены полностью независимые ручные
SSB, не исправлен согласованный SF10 raw-prefix view текстовых/byte-полей, не
завершена общая приёмка типов/выходных layout и SQL performance targets. Replay,
фоновые JIT tiers и постоянный кеш SQL-модулей не добавлялись.

### Как устроена реализация

```text
SQL → resolve/extract parameters → структура + schema + каталог
    → готовый physical plan либо parameterized optimize/lowering
    → поиск отдельных модулей → codegen/compile/link только отсутствующих
    → PCUDA plan interpreter, runtime-owned buffers, один явный stream
    → типизированный CPU QueryResult
```

| Файлы | Ответственность |
| --- | --- |
| `src/core/visitor.cpp`, `expression_codegen.cpp` | Fused device bodies и явные операции; обычные значения не участвуют в генерации |
| `physical_planner.cpp`, `execution_plan.cpp` | Контракты аргументов, размеры, ветвления, копии, обеспечение ёмкости и исполнение плана |
| `module_codegen.cpp`, `code_symbols.cpp`, `content_hash.cpp` | Канонизация локальных имён аргументов, полный структурный ключ, SHA-256 identity, одна translation unit на одно ядро |
| `module_compiler.cpp` | Штатный acpp, отдельные compile/link, 1–4 независимые цепочки, private artifacts и native timers |
| `module_cache.cpp` | Загруженная `.so`, заранее разрешённые identity/descriptor/launch entries, проверка ABI/schema и транзакционная публикация кеша |
| `execution.cpp`, `pcuda_plan_runtime.cpp` | Готовые планы отдельно от текущих параметров/буферов; runtime adapter без SYCL launch-view |
| `pcuda_plan_services.cpp`, `pcuda_scan.cpp` | Заранее собранный многоуровневый prefix-scan; reusable workspace принадлежит runtime |
| `include/jit_headers.h`, `cmake/`, `tools/pcuda_pch_manifest.py` | Минимальный PCH, fingerprints и build/install dependencies, не SQL-path preparation |

ABI v1 layout не менялся. Native модули экспортируют `db_kernel_descriptor_v1`,
`db_kernel_launch_v1`, дополнительно `db_kernel_identity_v1`; kernel/lambda
находится в `db_module_<SHA256>`. Аргументы/буферы привязывает план, не загрузчик.
Одинаковые bodies со схемами/настройками могут переиспользоваться между планами,
даже когда исходные имена колонок/параметров различны. Это не разбиение fused
операторов на отдельные SQL-ядра.

Старый whole-query compiler и `legacy_kernel_codegen.cpp` удалены. Legacy adapter
сохранён только для исторического ABI-negative fixture; в `db_cli` не линкуется.
`bindLegacyPlan` остаётся совместимостью host-тестов, но не используется engine.

`DevicePrefix` заменяет прежние пять SQL-generated MHT scan kernels, три projection
scan kernels и host prefix агрегатной материализации. На host передаётся только
проверенный total для ёмкости/control flow; сами counts/offsets и строки остаются
на GPU. Для Int32 служба расширяет counts до UInt64, отвергает отрицательные counts
и total >INT32_MAX до narrowing; для UInt64 используется checked scan этапа 4.
Старый лимит MHT `256^3`/projection `1024 block sums` удалён; ограничения Int32
directory/row IDs и memory guard остаются явными.

LIMIT — execution-local size slot; необходимые sort-варианты заранее присутствуют
в плане. GROUP BY CASE использует фиксированное число листьев, а равные значения
и NULL объединяются с первым равным листом. Runtime min/max-домен потребовал бы
пересчёта диапазона/ёмкости для значений; компактное кодирование выбрано потому,
что не зависит от ширины диапазона и сохраняет ограниченную структурную ёмкость.

Владение: private `mkdtemp` directory живёт с compiled/loaded owners; завершение
использующей модуль GPU-работы подтверждается до `dlclose` и удаления артефактов.
Failed builds сохраняют диагностику, но не становятся кешем следующего процесса.
Временные артефакты разных сессий не имеют общих writable имён. QueryEngine/cache
и prefix workspace рассчитаны на одного writer; конкурентный доступ не обещан.

Doxygen-комментарии описывают идею, контракты и владение. **Doxygen не запускался,
HTML и другие страницы не генерировались.**

## 2. Окружение и конфигурация

Все локальные артефакты:
[`build/pcuda-stage5/run-i5pbu1e6`](../../build/pcuda-stage5/run-i5pbu1e6/).
Они не являются опубликованным переносимым архивом. Начальное грязное дерево
сохранено в `initial/`, `initial.diff`, `initial-sha256.json`, `initial-status.txt`;
пользовательские debug/PCH-эксперименты и результаты предыдущих этапов не очищались.

- AMD Radeon 780M Graphics (`gfx1103`), Ryzen 7 7840HS; runtime подтвердил FP64 и
  atomic64. NVIDIA/Intel аппаратно не проверялись.
- AdaptiveCpp `25.10.0+git.5989e1cf.20260821.branch.HEAD`; Clang/LLVM и LLD 17.0.0,
  ROCm LLVM revision `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`, SDK
  `/opt/rocm_sdk_612`. Установленный toolchain не изменялся.
- SF10: 59 986 217 fact-строк, существующее физическое хранилище; его text/byte
  ограничения не выдаются за корректное декодирование новых SQL-типов.
- Host — Release `-O3 -DNDEBUG`; тесты добавляют `-UNDEBUG`. Обычные host targets
  не запускают acpp. Prebuilt service targets используют PCUDA compiler launcher.
- SQL modules/PCH: `--acpp-targets=generic --acpp-pcuda -std=c++17 -O0 -g0 -fPIC
  -Wno-narrowing -Wno-deprecated-declarations`, project include directory и
  `--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1` для данного стенда.
  Link: отдельный `acpp --acpp-targets=generic --acpp-pcuda ... -fuse-ld=lld
  -Wl,-O0 -shared <module.o> -o <module.so>`. Пути/дополнительные flags конфигурируемы.
- PCH использует `-fpch-instantiate-templates -c -MD -MF ... -x c++-header`, модули
  — `-include-pch`. Размер build-tree PCH — **4 644 340 B**. Depfile не содержит
  `core/execution.h`, SQL parser, `crystal/utils.h`, `iostream`, `fstream` или
  `sycl/sycl.hpp`; внутренние libkernel headers самого PCUDA SDK не скрываются.
- В performance-процессах: `LC_ALL=C`, `TMPDIR=/tmp/opencode`,
  `ACPP_ADAPTIVITY_LEVEL=1`, `ACPP_ALLOCATION_TRACKING=1`; traces включены только
  в соответствующем режиме. Compiler-affecting environment фиксируется manifest
  и восстанавливается в дочерних compile/link, не меняя runtime environment CLI.

Версии, существенные env и полные hashes —
[`environment/inventory.json`](../../build/pcuda-stage5/run-i5pbu1e6/environment/inventory.json).
Измеренные SHA-256:

| Артефакт | SHA-256 |
| --- | --- |
| `engine/db_cli` | `58f7e7a5ca2b204a18331513e923c9dd8702fc91f70bb4aa72500788e8c69d94` |
| Build-tree PCH | `cfab0d5d9b0df6508ac44cac0395578b6d77f07b4ada519dd1c8eacd11392dc9` |
| Toolchain manifest | `2f6245516489832747525be2d412ab764807c6709da49eb14890dbd70764b89c` |

Команды host-сборки: `engine/CMakeCache.txt`, `compile_commands.json`,
`CMakeFiles/*/link.txt`. Все исходные acpp argv находятся в `.compile.command` /
`.link.command` каждого модуля. Effective Clang и полный LLD argv трёх
представительных модулей — `module-inspection/0/*-effective.*`, включая plugins,
CRT/stdlib/runtime libraries и rpath. Это dry-run диагностика после измерений,
**не replay**. Manifest содержит fingerprints зависимостей PCH/compiler/plugins/
runtime/config; linker version/hash также зафиксирован в inventory.

Установка проверена в независимых `installed-v1/` и `installed-v2/`. PCH готовится
заново относительно конечных установленных headers; `DESTDIR` relocation явно
отвергается. Последовательность lookup: env → explicit config → installed manifest
относительно executable → build default. При несовместимости входов — диагностика,
без скрытой подготовки PCH из SQL. Для GPU depfile tracking нужен CMake ≥3.20 или
Ninja; CPU-only режим не требует поиска AdaptiveCpp.

## 3. Методика

Timing model — **`pcuda-modules-v1`**, см. [timing.md](../../timing.md#этап-5-pcuda-modules-v1--2026-11-07).
SQL `summary` — один непрерывный `steady_clock total_query_ms` в
`DatabaseInstance::executeQuery`, от передачи SQL в уже загруженную БД до готового
CPU QueryResult. Внутри находятся parse/plan, codegen, необходимые build/load,
runtime allocations, execution и fetch. Oracle, форматирование, печать, startup,
загрузка исходных таблиц и подготовка PCH находятся снаружи.

Native compile/link wall/user/system сохраняет Bash `time`, секунды с шестью
знаками, `LC_ALL=C`; в итогах они переведены в мс. Собственная работа Python-driver
acpp включена, дополнительного Python timer/proxy внутри измеряемой цепочки нет.
`build_elapsed_ms` — прямой C++ интервал всех цепочек; command wall sums при overlap
не складываются с ним. Сохраняются Clang `-ftime-trace` и LLD `--time-trace`; Python
разбирает завершённые transcripts/traces, не анализирует их одновременно с GPU.

`codegen_ms` охватывает physical lowering на plan miss; wrapper emission,
канонизация/hash/cache bookkeeping, запись build script и разбор timers остаются
в прямом summary, но не все входят в component intervals. Их residual не обнуляется.
По этому полю **не заявляется** полный прямой SQL→PCUDA ≤10 мс.

Performance-серия: **7 независимых cold-процессов на plain/traced**, в каждом
1 исходный Q3.3 +8 обычных warm +8 с изменёнными параметрами. Порядок режимов
чередуется между sessions. Изменённые значения: города 235/235 и годы 1993–1996,
при той же SQL-структуре; первый такой запрос также обязан иметь нулевые build
events. Каждый процесс получает новый `ACPP_APPDB_DIR`, сохранённый внутри процесса;
SQL/module cache на входе пуст. OS/HIP/driver caches не очищались, частоты/температуры
не фиксировались; фоновые desktop-нагрузки возможны.

Корректность проверена **после** performance-серии: исходный Q3.3 — существующий
ручной SF10 reference, изменённый — его независимое подмножество по ключам групп.
Counter сохраняет multiplicity и не навязывает порядок без ORDER BY. Oracle не
работает в измеряемом CLI даже между warm-запросами. Всего **238 результатов**.

Первое execution/fetch может включать device JIT, но его чистое время отдельно
не наблюдалось. Prebuilt services могут впервые JIT-компилироваться runtime.
Native wall всего процесса включает data loading, 17 запросов и teardown:
это **не полный старт до первого результата**. Отдельные startup/data-loading и
process→first-result интервалы этой работой не измерены и не восстанавливаются
вычитанием/суммой стадий.

## 4. Результаты

### 4.1. SQL Q3.3 SF10

Основной raw набор:
[`sql-series-6ldbk47n/series.json`](../../build/pcuda-stage5/run-i5pbu1e6/sql-series-6ldbk47n/series.json),
offline [analysis.json](../../build/pcuda-stage5/run-i5pbu1e6/sql-series-6ldbk47n/analysis.json).
Все samples, порядок, native process timers, исходники, `.so`, команды и traces
сохранены в подкаталогах. Все времена ниже — **мс, медиана [min; max]**.

| Режим / фаза | n | **summary (cold) / total (ready)** | Build elapsed | Execution | Fetch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain, cold | 7 | **662,947 [659,449;666,457]** | 149,261 [148,191;151,382] | 441,325 [437,915;443,314] | 69,911 [68,894;71,452] |
| Traced, cold | 7 | **662,708 [658,921;668,560]** | 151,102 [147,851;152,921] | 440,314 [439,372;444,538] | 69,258 [67,973;70,593] |
| Plain, warm | 56 | **17,164 [16,829;18,624]** | 0 | 16,626 [16,384;17,884] | 0,380 [0,247;0,544] |
| Traced, warm | 56 | **17,108 [16,817;17,704]** | 0 | 16,564 [16,404;17,019] | 0,380 [0,254;0,495] |
| Plain, изменённые параметры | 56 | **16,385 [16,233;16,776]** | 0 | 15,901 [15,758;16,129] | 0,339 [0,231;0,655] |
| Traced, изменённые параметры | 56 | **16,376 [16,268;16,843]** | 0 | 15,887 [15,793;16,334] | 0,344 [0,231;0,811] |

Cold command sums и load, **каждая колонка — своя медиана**, n=7:

| Режим | Compile Σ wall/user/system | Link Σ wall/user/system | Load | **summary** |
| --- | ---: | ---: | ---: | ---: |
| Plain | 240,357 /183,766 /52,186 | 197,725 /144,855 /72,318 | 0,413 | **662,947** |
| Traced | 245,341 /192,787 /50,724 | 196,801 /147,874 /69,707 | 0,419 | **662,708** |

Compiler/linker sums **не elapsed** и не добавляются к summary. На Q3.3 выполняются
четыре fused launch, но компилируются только **три** уникальных `.so`: customer и
supplier build структурно одинаковы после канонизации имён и разделяют модуль.
Во всех cold-процессах ровно 3 compile, 3 link, 3 load, 9 symbol resolutions.
Все 224 ready-запроса имеют plan/module codegen, compiler, linker, load, symbols=0.
Размер трёх `.so`: plain **95 160 B**, traced **95 176 B**.

В traced-режиме суммы одноимённых фаз трёх компиляций, n=7: ExecuteCompiler
**82,174 [80,232;85,486]**, Frontend **39,330 [38,081;42,098]**, Backend
**34,899 [34,528;35,419]**, Optimizer внутри backend **16,387 [16,234;16,609]**,
ExecuteLinker **27,062 [25,896;27,643]**. Их summary — traced cold из таблицы;
родительские/дочерние фазы не складываются. Остаток между trace и native wall
не объявляется чистым временем Python/acpp driver.

Наблюдаемая разница trace−plain median: build **+1,841 мс**, compile Σ **+4,984 мс**,
link Σ **−0,924 мс**, summary **−0,239 мс**. Это оценка диагностического режима с
шумом, не утверждение, что traces ускоряют SQL; ничего не вычиталось из raw.

PCH preparation — отдельно, n=1 на установку: v1 native wall/user/system
**373,941 /291,415 /32,668 мс**, v2 **499,772 /453,608 /38,647 мс**. Это разные
установочные prefix/моменты, не controlled PCH speed comparison; данные —
`installed-v{1,2}/lib/crystal/pcuda-jit/pch.time`. В SQL summary они не входят.
Native process wall с 17 запросами: plain **4514,780 [4505,106;4689,184]**,
traced **4512,390 [4499,810;4520,389]**; это не startup→первый результат.

**Цели 600/200/10/9 мс этапа 0 не применяются к этой SQL-серии.** Здесь другой
домен/алгоритм общего SQL-пути, а не компактный 24-group prototype. Отдельная
современная SYCL comparison series не проводилась; backend/flags speedup и SQL SLA
из этих чисел не выводятся. Контроль прототипа и согласование полных SQL-целей
остаются этапу 8.

### 4.2. Функциональная и инструментальная регрессия

| Проверка | Результат | Raw относительно каталога артефактов |
| --- | --- | --- |
| Release CTest | **12/12** после итоговой build-rule правки | `ctest-final.stdout` |
| CPU-only без AdaptiveCpp | **10/10** | `cpu-configure.*`, `cpu-build.*`, `cpu-ctest.stdout` |
| Manifest/current/current_heavy + targeted/boundaries | **88/88** | `manifest-plain-960r1cu3/report.json` |
| 13 SSB ×3, один процесс | **36/39 с файлами**, warm reuse verified | `ssb-plain-cx46kw1m/report.json` |
| Установленный CLI, все 13 SSB | **12/13** на каждой из v1/v2, только Q1.1 | `ssb-plain-8qln62x2/`, `ssb-plain-w_lxh1ol/` |
| Native modules/параметры/cache/lifetime/prefix | Успех; ровно 8 typed +1 fallback, exact Int64, known typed NULL, pending-work unload | `modules-v6.stdout`, `modules-v6/` |
| Tiny MHT/OR/NULL/ABI/memory guard и PHT | Успех; 7 PHT results, независимые tiny oracles | `gpu-regression/report.json`, `test_gpu_plan.*`, `test_pcuda_primitives_sql.*` внутри него |
| Primitives/runtime | **17 323 905** per-element/assertion checks; 21 sparse fixtures с повторами и async-error checks | `gpu-regression/test_pcuda_primitives_gpu.*`, `test_pcuda_gpu_runtime.*` |
| Nullable CSV/TSV | **5/5**, независимый SQLite CPU oracle | `nullable-oracle.json` |
| ELF/HCF | **224 модулей /18 процессов**, один HCF/native kernel на `.so`, фиксированные C entries, нет конфликтов разных identities | `module-inspection/report.json` |
| ASan/UBSan host modules и plan interpreter | Успех, не device sanitizer | `host-sanitized.*`, `plan-sanitized.*` |
| Общие ub-tools | **25/25** | `ub-selftests-v2.stderr` |
| Targeted semantic mutations | **12/12 host**, **8/8 GPU killed**, survivors/unresolved=0 | `ub/host-mutants-v2/report.json`, `ub/gpu-mutants-v2/report.json` |
| CRAP-region | **73 module +22 expression definitions/instantiations**, max **9,584 <10** | `ub/crapper-v3/`, `ub/expression-crapper-v1/` |
| Dryer | Независимых whole-body пар rank ≥0,8 при ≥50 tokens не найдено, шесть новых host sources | `ub/dryer-v3/report.json` |
| Runner CPU checks | **5/5**: separate dump, metrics, parameters-first reuse, depfile escaping, raw statistics | `runner-final.stderr` |
| Toolchain PCH dependency | Старое правило не rebuild; новое rebuild; no-op сохраняет PCH | `pch-dependency-check-v2/report.json` |

## 5. Корректность и ограничения проверок

- Q1.1 не исправлялся под файл. Независимый `test/pcuda_stage0/q11_reference.cpp`
  заново прочитал четыре физические колонки: SQL **4486422947100**, 1993-only/file
  **4474878955766**, вклад 1994-01-01 **11543991334**. `q11-reference.stdout`
  сохраняет результат. SQL включает граничный день, reference — нет; исходные
  SSB `success:false` и все четыре mismatch-процесса (основной, прежний, две
  установки) не превращены в успешные заменой oracle.
- Manifest gate проверяет исполнение/форму всех current queries, но не является
  полным независимым oracle каждой их ячейки. Targeted, tiny и SSB результаты
  сравниваются независимо; future SQL-категории не выдаются за реализованные.
- CASE проверяет широкий диапазон, равные ветви и обе NULL, с правильным слиянием;
  HAVING/арифметика/селективность/LIMIT 0 меняются без нового codegen/build.
  Нужные алгоритмические варианты не специализируются по текущему LIMIT.
- Большое целое **9007199254740993** проходит через ABI и projection без double.
  Известный тип сохраняется при следующем SQL NULL. Первый NULL-only остаётся
  отдельной допустимой схемой. Четыре независимых predicate-параметра проходят
  16 Int64/Float64 schemas, затем повторы всех схем и typed NULL: ровно девять
  модулей для этого kernel family; прежние typed остаются доступны.
- Bounded family включает body, существенный output/buffer layout и ABI. Проверка
  не обещает девять модулей на произвольную SQL-строку независимо от изменения
  выходной схемы. Полная матрица fractional aggregate semantics/выходов и смены
  типов ещё не принята; SQL type system автоматически не расширена. Float oracles
  нового tiny module fixture преимущественно целочисленные по значению.
- Загруженный модуль проверяется по полному структурному ключу/identity, версии,
  backend и двусторонней совместимости schema. Отказ build/load не резервирует
  typed-вариант навсегда; повтор после исправления проходит. Отрицательные тесты
  покрывают несовместимые flags/PCH, отсутствующие symbols, ABI/schema/identity,
  неверный stream/block и память. Pending-work test освобождает compiler/cache/
  last module owner до чтения результата, проверяя completion перед unload.
- При аудите найдены одинаковые конкурентные обычные записи group keys. Они
  заменены PCUDA `atomicExch`, а требования atomic64 учитывает runtime. Аппаратная
  регрессия выполнена после исправления; device race sanitizer не запускался.
- Prefix service дополнительно проверен на 65 539 Int32 elements, partial blocks,
  in-place/empty, INT32_MAX total, overflow/negative counts, null pointers, UInt64
  значения выше 2^53. Огромные Int32 domains не аллоцировались ради формального
  теста лимита: большие размеры по-прежнему ограничивают checked arithmetic,
  launch validation и memory ledger.
- Coverage **host-only и неполное**: из 95 выбранных definitions 66 полностью
  покрыты, 29 — не полностью, включая две простые неисполненные definitions.
  Низкий CRAP не означает полное покрытие. GPU LLVM coverage не собиралось;
  `visitor.cpp`, весь runtime и device path не объявляются целиком покрытыми
  crapper. Для них здесь настоящие hardware/oracle/mutation проверки.

### Сохранённые неуспехи и изменения методики

Начальный parameter-plan red и `prefix-red.*` подтверждают ожидаемый fail до
интеграции. После замены scan старый optimizer assertion ожидал `read-counter`;
он заменён проверкой двух реальных `DevicePrefix`, при сохранении проверки fusion.
Все последующие CPU/GPU результаты относятся к собранным исправлениям.

`crapper-v1` отверг constructor CC=10 и недостаточный `storageType` coverage;
constructor разделён по ABI/schema responsibility, добавлены типовые тесты.
`crapper-v2` отверг неполный installation lookup; реальный copy-executable fixture
проверил installed и env override. Пороги не менялись. `plan-coverage-v1` с двумя
бинарниками получил LLVM **9 mismatched functions**, поэтому успешный coverage
не опубликован; отдельная однобинарная expression-коллекция `plan-coverage-v2`
прошла без diagnostics. Нельзя выдавать первый export за успешный.

Первую GPU mutation-кампанию `gpu-mutants-v1` прервал внешний 120-секундный лимит
команды; это **незавершённая инфраструктурная попытка, не kill**. Полная v2 выполнена
в новом каталоге без внешнего лимита, с per-build/test deadlines. Host/GPU mutations
делались только в независимых source copies после пересборки baseline.

Первый standalone plan sanitizer link не содержал definitions виртуальных AST
методов; повтор включает настоящие visitor/frontend objects, без отключения RTTI
или sanitizer. Оба build stderr сохранены.

В первом PCH dependency исправлении generated manifest не отслеживался CMake
после первоначальной генерации; честный fixture `pch-dependency-check` сохранил
`success:false`. Вместо второго parser/самоподдерживающейся configure dependency
выбран **один владелец списка**: manifest writer расширяет существующий depfile
эффективными toolchain inputs. V2 использует private compiler wrapper/config,
реально обнаруживает fingerprint input, меняет только собственный файл и
подтверждает rebuild/no-op. SDK/проектные headers не менялись. Наблюдение mtime
нужно для correctness, это не Python performance timer.

После dependency-only rebuild hashes `db_cli`, PCH и manifest **совпали с измеренными**:
[`dependency-rebuild-hashes.json`](../../build/pcuda-stage5/run-i5pbu1e6/dependency-rebuild-hashes.json).
Измеренный PCH/manifest дополнительно архивированы в `measured-pch/`.

## 6. Выводы и решения

**Подтверждено:** native отдельные модули работают на одном PCUDA stream;
неизменные bodies разделяются внутри/между планами; runtime-параметры не требуют
повторного SQL JIT; loader entries кешируются; PCH не создаётся на SQL-пути;
универсальный scan вынесен из SQL-компиляции; ресурсы и loaded lifetime не принадлежат
генерируемому модулю. Все десять пунктов этапа 5 проверены.

**Не подтверждено/не заявляется:** полный перенос всех примеров/storage types,
готовность этапов 6–8, SQL SLA, ≤10 мс полного frontend translation, отсутствие
device JIT на ready SQL, чистое разбиение его стоимости, аппаратная работа на
NVIDIA/Intel. Warm SQL Q3.3 этой серии около 17 мс, не 9–10 мс прототипа; это
разные реализации, и скрытого ослабления согласованных целей нет.

**Решения пользователя:** необходимые предпосылки этапа 6 включены; сам этап
не закрывается автоматически. Политика 8 typed +1 fallback, known typed NULL,
process-only cache и штатный acpp сохранены. Replay остаётся необязательным этапом
9 после этапов 1–8. Новые performance-пороги не утверждались.

Гипотеза, что большая часть первого execution/fetch — device JIT, правдоподобна,
но весь интервал ему не приписывается; специальной S2-инструментации здесь нет.
Малые различия diagnostic medians не объявляются статистически доказанным выигрышем.

## 7. Воспроизведение и оставшаяся работа

Команды выполняются из корня проекта. Машинные пути ниже — конфигурация данного
стенда; для другого SDK/GCC их нужно явно заменить. Каталоги quality/series должны
быть новыми, существующие raw не перезаписываются.

```bash
RUN="$PWD/build/pcuda-stage5/reproduce"
GCC=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake -S . -B "$RUN/engine" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC -DBASE_PATH=\"$PWD/data/\"" \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DCRYSTAL_PCUDA_COMPILER=/opt/AdaptiveCpp/bin/acpp \
  "-DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=$GCC" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUN/engine" -j4
ctest --test-dir "$RUN/engine" --output-on-failure
TMPDIR=/tmp/opencode "$RUN/engine/test_pcuda_sql_modules" "$RUN/modules"
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" \
  --mode manifest --runs 1 --output-root "$RUN"
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" \
  --mode ssb --runs 3 --output-root "$RUN"
```

SSB runner возвращает failure из-за исходного Q1.1 mismatch — это ожидаемый,
сохранённый результат сравнения с файлами, не повод игнорировать остальные ошибки.
Для новых данных нужен собственный независимый oracle, не перенос известной суммы.

```bash
python3 test/pcuda_stage5/measure_series.py --db-cli "$RUN/engine/db_cli" \
  --sessions 7 --runs 9 --output-root "$RUN"
# Подставить напечатанный новый sql-series-*:
python3 test/pcuda_stage5/analyze_series.py "$RUN/sql-series-IDENTIFIER"
python3 -m unittest discover -s test/pcuda_stage5 -p test_runner.py -v
python3 test/pcuda_stage5/test_pch_dependencies.py --output "$RUN/pch-deps" \
  --compiler /opt/AdaptiveCpp/bin/acpp --cxx /opt/rocm_sdk_612/bin/clang++ \
  "--flags=--gcc-install-dir=$GCC"
cmake --install "$RUN/engine" --prefix "$RUN/installed"
```

Общие `tools/ub/coverage.py`, `crapper.py`, `dryer.py`, `mutator.py` вызываются с
`host-recipe.json`, `plan-recipe.json`, `gpu-recipe.json`. Полные argv, sources/hashes,
stdout/stderr и причины — в `ub/*/actions.jsonl` и `.command.json`. Scope crapper:
`content_hash.cpp`, `code_symbols.cpp`, `module_codegen.cpp`, `module_compiler.cpp`,
`module_cache.cpp` плюс отдельный `expression_codegen.cpp`. Для GPU mutation recipe
нужны совместимые `CXXFLAGS`, `CRYSTAL_PCUDA_FLAGS`, `AdaptiveCpp_DIR`, `PATH`; никаких
новых копий quality tools не создано. ELF inspection воспроизводится
`inspect_modules.py <artifact-directories> --effective --output <new-directory>`.

Журнал действий — [`actions.jsonl`](../../build/pcuda-stage5/run-i5pbu1e6/actions.jsonl);
stage-only diff, исходные snapshots и preservation audit находятся в том же корне
артефактов. Commit/push и очистка чужих незакоммиченных работ не выполнялись.

**Дальше, отдельными задачами:** завершить оставшийся аудит runtime-dependent
arrays/эвристик/памяти и общую type/NULL матрицу этапа 6; исправить SF10 decoding и
перенести ручные SSB на этапе 7; затем окончательная функциональная/ошибочная
приёмка, прямой SQL→PCUDA budget, контроль прототипа и согласованные SQL cold/warm
цели этапа 8. Закрытие этапа 5 не поручает эти работы автоматически.
