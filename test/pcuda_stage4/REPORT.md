# Этап 4: GPU-примитивы PCUDA

## 1. Статус и границы

**Этап 4 выполнен и проверен 2026-11-06 в границах библиотеки примитивов.**
Перенесены загрузки/предикаты, PHT/MHT и атомарные
helpers; добавлены block reduction, block exclusive scan и runtime-workspace
для многоуровневого scan. Ни один новый device-заголовок не требует полного
`sycl/sycl.hpp`, host I/O, SQL parser или каталога.

Это **не завершение миграции SQL**. Генератор ещё выпускает SYCL launch wrappers,
а его reductions/scans остаются legacy до следующих этапов. Библиотека запроса,
SQL-зависимый кеш и runtime-параметризация не заменены. Новый `ExclusiveScan`
готов как отдельный prebuilt runtime-сервис, но не подменяет скрыто старый SQL
prefix-scan. ABI v1 и same-executor bridge этапа 3 сохранены.

### Идея реализации

| Компонент | Обязанность |
| --- | --- |
| `include/crystal/load.h`, `pred.h` | Striped load/gather, validity и маски; padded lanes обнуляются, смещения не сужаются до int |
| `include/crystal/hash.h` | Остаток `key-min` без signed overflow и отрицательного индекса; широкие interleaved offsets |
| `include/crystal/join.h` | Существующие perfect-domain PHT/MHT, публичные PCUDA CAS/add; единственный писатель каждой MHT-directory записи |
| `include/core/inline_math.h`, `device_atomics.h` | Отдельно чистая арифметика/bit codecs и device-scope relaxed атомики |
| `include/crystal/collectives.h` | Общая shared-memory reduction/scan без предположений о warp size |
| `include/core/scan_layout.h`, `pcuda_scan.h`, `src/core/pcuda_scan.cpp` | Checked layout, runtime-owned high-water scratch, все уровни scan и обратное добавление offsets в одном stream |
| `include/crystal/device.h` | Минимальный device preamble; `utils.h` остаётся host-only |
| `visitor`, `legacy_kernel_codegen`, `execution` | Явные occupancy-аргументы/буферы, nullable join keys и разрешение публичного PCUDA API в пока legacy compiler/PCH |

**Словарь:** tile — striped порция входа; padded lane — поток/элемент за концом
входа; occupancy — отдельное UInt32-слово занятости PHT-слота; directory MHT —
`[key, offset, count]`; scan level — проход по totals предыдущего уровня;
High-Water Mark — сохранённая ёмкость scratch, а не повторная аллокация на запрос.

Пользователь выбрал **явную occupancy-карту PHT**, а не перевод нулевых ключей
в MHT. Теперь 0 является обычным ключом. Карта очищается планом и передаётся
build/probe как аргумент. Payload/key публикует только выигравший CAS поток;
probe идёт следующим ядром того же stream, одновременный build/probe не поддержан.
Для ещё не перенесённых независимых ручных SSB оставлены совместимые nonzero-key
вызовы без occupancy; SQL ими не пользуется. Это тот же PCUDA алгоритм, не второй
backend. Полный перенос ручных программ остаётся этапу 7.

У многоуровневого scan нет фиксированного лимита числа уровней. `enqueue()`
проверяет layout/launch/capabilities и ставит всю иерархию без промежуточных wait.
`finish()` получает total/status на CPU и отвергает переполнение UInt64. После
ошибки частичной submission выход также недействителен. Буферы принадлежат runtime,
не ядрам; pending workspace нельзя переиспользовать до finish. Exact in-place scan
разрешён, частично перекрывающиеся диапазоны — нет.

Doxygen-комментарии описывают идею, preconditions, владение и ограничения.
**Doxygen не запускался, HTML/другие страницы не создавались.**

## 2. Окружение и конфигурация

Все артефакты этой работы:
[`build/pcuda-stage4/run-ecd8a23n`](../../build/pcuda-stage4/run-ecd8a23n/).
Это локальное хранилище, не опубликованный архив. Начальное незакоммиченное
дерево сохранено в `initial/`, `initial.diff`, `initial-sha256.json`; исходные
примитивы для сравнения — в `baseline/include/`.

Стенд: AMD Radeon 780M, AdaptiveCpp
`25.10.0+git.5989e1cf.20260821.branch.HEAD`, Clang/LLVM/LLD 17, C++17.
Версии и существенные переменные — `environment/`. NVIDIA/Intel не проверялись.
Конфигурация сборки — `configure.command.json`, `engine/CMakeCache.txt`,
`engine/compile_commands.json`, `engine/CMakeFiles/*/link.txt`.

Host — Release; assertions тестов включены через `-UNDEBUG`. Новый service
собирается acpp `--acpp-targets=generic --acpp-pcuda`; SQL compiler и его старый
PCH дополнительно получили только `--acpp-pcuda`, необходимый публичным PCUDA
headers/атомикам. Новый модульный compiler/cache/PCH этим не реализуется.
Пути SDK/GCC в командах — конфигурация стенда, не новые константы движка.

## 3. Методика

CPU-тесты проверяют настоящие headers: все сравнительные адаптеры, padded/empty
tiles, NULL validity, Int32 promotion, точность выше 2^53, signed hash extremes,
64-битные row IDs и checked scan layout. Hardware-fixture использует независимые
CPU-циклы для SUM/MIN/MAX, каждого prefix, PHT membership и MHT payload multiplicity.
Host coverage не выдаётся за device coverage. Для качества используются общие
локальные инструменты `tools/ub/`, не новые копии инструментов для этапа.

GPU-процессы последовательны. Каждый процесс получает новый `ACPP_APPDB_DIR`;
внутри процесса кеш сохраняется. OS/HIP/driver caches не очищаются. SQL runners
сохраняют старый PCH в собственные артефакты, не запускаются одновременно.
SQL timing model — `explicit-plan-v1`; повторный physical lowering допустим и
не считается новым вызовом acpp.

Performance-контроль — **prebuilt primitive microbench**, не SQL/JIT-модульный
конвейер: 1 048 589 Int32 входов, 65 536 PHT keys, блок 128. Сравниваются исходные
headers/SYCL и новые PCUDA primitives на одинаковых данных/stream. Операции:
load+predicate, PHT build/probe, block SUM, block exclusive scan (не глобальный
prefix всей колонки). Для PHT новое время честно включает occupancy init.

Host benchmark собран `-O3 -DNDEBUG`, kernel TU — штатным acpp `-O0`, без PCH.
Compile и link измеряются отдельно native Bash time, `LC_ALL=C`; сохраняются
Clang `-ftime-trace`, LLD `--time-trace`, dry-run/effective commands и plain-контроль.
Внутренние интервалы — `steady_clock` с completion на границе execute и после D2H.
`summary` измеряется одним интервалом: lazy output/scratch preparation, init,
launch, completion и CPU fetch. Startup, входной upload, compilation/linking
prebuilt бинарника, независимый oracle и печать в него **не входят**.
Это явно не SQL cold latency: parse/plan/codegen и dlopen здесь отсутствуют.

Операции/варианты ротируются; первый запуск нового primitive может включать
device JIT и использовать инфраструктуру/буферы предыдущей операции процесса.
Cold означает первый запуск этого primitive, не очищенный весь stack перед
каждым ядром. Разность cold/warm не называется чистым временем device JIT.
Python только организует native команды и разбирает готовые результаты.

Итоговая методика benchmark — **V2**: CPU-oracle выполняется только в четырёх
отдельных correctness-процессах, по 36 результатов каждый, до performance-серии.
В 28 измеряемых процессах oracle не выполняется даже между warm-интервалами.
На каждый вариант/diagnostic mode — 7 процессов × (1 первый + 8 warm) для каждой
операции; compile/link — три независимые сборки на комбинацию. Выгрузка CPU-результата
в summary сохранена. Температуры/частоты не фиксировались, OS/driver caches не
очищались; выводы ниже относятся к этому стенду и серии, не доказывают универсального
статистически значимого выигрыша.

## 4. Результаты

Первичный Release CTest после добавления occupancy: 9/10; старое ожидание трёх
scratch-буферов Q2.1 заменено проверкой шести (три таблицы + три occupancy).
После этого CTest — 10/10. Это изменение контракта, не ослабление correctness gate.

Первый аппаратный fixture выполнил **17 323 902** oracle assertions: signed/unsigned
SUM/MIN/MAX, FP64, block sizes 1/3/31/48/127/128/255/256/1023, empty/full/partial
входы, scan 16 777 217 элементов (четыре уровня), deep scan с блоком 3, in-place,
reuse, overflow, PHT и MHT. Последующие проверки/мутации включают дополнительные
CAS64 и sparse bitmap checks; их точные результаты сохранены отдельно.

SQL после исправления NULL probe: manifest/targeted **85/85**; дополнительные
границы Top-K — **22/22 plain** и **22/22 traced**; tiny MHT/OR/NULL и новый
PHT-fixture успешны. Последний проверяет семь результатов: negative/zero/absent
keys, zero build-filter, повтор запроса, изменение validity и отключение reuse.
SSB 13×3 — **36/39 с файлами**, только известный Q1.1×3; compiler-cache gate
успешен. Raw `success:false` SSB не переопределяется общим success.

### Итоговая регрессия и ub-mode

Финальные команды и return codes:
[`validation-verified.json`](../../build/pcuda-stage4/run-ecd8a23n/validation-verified.json),
сводка путей — `validation-summary.json`. Этот повтор выполнен **после**
выделения общих build/probe/scan helpers, предыдущие прогоны не перезаписаны.

| Проверка | Подтверждённый результат | Сырые данные |
| --- | --- | --- |
| Release CTest | 10/10, включая реальные boundary expressions плана для INT32_MAX / INT32_MAX+1 | `ctest-verified.stdout` |
| Hardware primitives | 3/3 процесса, в каждом 17 323 905 per-element/assertion checks, включая CAS64 и точный sparse bitmap bit | `gpu-verified-{0,1,2}.stdout` |
| SQL PHT zero/negative/NULL/reuse | 7/7 plain и 7/7 traced | `gpu-fixture-0hja9ioz/`, `gpu-fixture-8meh0zdd/` |
| Tiny MHT/OR/NULL/ABI/memory guard | Успех, независимые tiny-input oracles | `gpu-fixture-29gp4hfy/` |
| 13 SSB ×3 | 36/39 с файлами; только Q1.1×3, подтверждённый CPU oracle; 1 PCH +13 compile/link, warm compile=0 | `sql-baseline-a4gv915h/report.json` |
| Manifest и targeted | 85/85; 64 current/current_heavy файла, 66 SQL statements; future не включался | `cli-plain-asxgl2pu/report.json` |
| Top-K/targeted | 22/22 plain и 22/22 traced, по 21 независимому oracle + ожидаемый legacy rejection | `cli-plain-d920tyob/`, `cli-traced-gy229vim/` |
| NULL CSV/TSV | 5/5 SQLite CPU oracle, не GPU CSV loader | `nullable-oracle.json` |
| Ручные SSB | 13/13 syntax checks с `--acpp-pcuda`; только source compatibility, не аппаратная приёмка или полный перенос | `manual-syntax/report.json` |
| ASan/UBSan | Host primitive contracts проходят | `host-sanitizer-build.stderr`, `host-sanitizer-run.stderr` |
| Общие ub-tools | 25/25 self-tests | `ub-selftest-final.stderr` |
| Targeted mutants | Host 9/9, GPU 8/8 killed; 0 survived/build-failed/timeout/unresolved | `ub/host-mutants-final/report.json`, `ub/gpu-mutants-final/report.json` |
| CRAP-region | 299 выбранных host definitions/instantiations, max=6 <10; все coverage identities измерены | `ub/crapper-final/report.json` |
| Dryer | Ни одной независимой whole-body пары ≥0,8 при min=50 tokens; 193 device/host AST definitions в расширенном scope | `ub/dryer-device-verified/report.json`, `ub/dryer-host-final/report.json` |

Manifest gate проверяет исполнение/форму всех current queries, но не является
независимым oracle каждого их значения. Отдельные targeted/tiny результаты
сравниваются полностью. Миллионы аппаратных checks — преимущественно сравнение
каждого элемента scan, а не миллионы различных SQL-запросов.

Coverage неполное: из 299 host definitions/instantiations полностью покрыты 253,
есть частично/неисполненные template regions (минимум coverage 0). Низкий CRAP не
подменяет полное покрытие. **GPU LLVM region coverage не собиралось**; для device
пути здесь AST/token-анализ, настоящие аппаратные oracles и семантические мутации.
ASan/UBSan также не является device sanitizer. Scope не выдаётся за проверку
всего `visitor.cpp` или всего движка.

Реальный большой GPU scan — 16 777 217 элементов; размеры/смещения за 2^32 и
INT32_MAX проверены host arithmetic/layout/plan tests без таких GPU-аллокаций.
Это не заявление о наличии памяти или аппаратном прогоне миллиардов строк.

### Производительность базовых примитивов

Основные V2 raw: [`primitive-bench-nqjn403a/report.json`](../../build/pcuda-stage4/run-ecd8a23n/primitive-bench-nqjn403a/report.json).
В каждом каталоге — исходные argv, stdout/stderr, native timers, objects/binary,
Clang/LLD traces и отдельные device caches. Проверено **144/144** результатов в
отдельных correctness-процессах; performance-серия — 28 процессов, 1008 интервалов.
Численные SLA SQL или прототипа этапа 0 к этой таблице не применяются.

Первый вызов операции, **мс, медиана [min; max]**, plain, n=7:

| Операция | Вариант | **summary** | Execute с возможным device JIT | Fetch, медиана |
| --- | --- | ---: | ---: | ---: |
| Load+predicate | baseline | **79,247 [77,847;83,738]** | 77,354 [75,912;81,816] | 1,922 |
| Load+predicate | PCUDA | **77,323 [75,881;85,964]** | 75,140 [73,934;84,081] | 1,927 |
| PHT build/probe | baseline | **166,625 [165,027;197,907]** | 164,696 [163,119;194,413] | 1,955 |
| PHT build/probe + occupancy | PCUDA | **189,209 [185,586;226,580]** | 187,253 [183,661;224,151] | 1,964 |
| Block SUM | baseline | **154,860 [152,081;166,625]** | 154,633 [151,956;166,514] | 0,127 |
| Block SUM | PCUDA | **80,893 [78,715;85,945]** | 80,785 [78,494;85,738] | 0,125 |
| Block exclusive scan | baseline | **3,472 [2,480;155,687]** | 0,573 [0,417;153,794] | 2,064 |
| Block exclusive scan | PCUDA | **2,612 [2,324;83,682]** | 0,704 [0,503;81,770] | 1,895 |

SUM и scan используют общий launch site с runtime-флагом. Первый scan часто идёт
после SUM; его разброс включает разное внутрипроцессное состояние. Эти первые
интервалы **не являются изолированным cold device JIT каждого kernel**; низкую
медиану scan нельзя трактовать как отсутствие JIT или многократное ускорение cold.
Чистое время device JIT отдельно не наблюдалось.

Warm, **мс, медиана [min; max]**, plain, n=56:

| Операция | baseline execute | PCUDA execute | baseline **summary** | PCUDA **summary** |
| --- | ---: | ---: | ---: | ---: |
| Load+predicate | 0,252 [0,233;0,281] | 0,238 [0,221;0,530] | **1,804 [1,749;2,850]** | **1,793 [1,738;2,703]** |
| PHT | 0,324 [0,302;0,936] | 0,338 [0,319;0,650] | **1,895 [1,808;5,084]** | **1,889 [1,826;3,863]** |
| Block SUM | 0,455 [0,435;0,474] | 0,304 [0,290;0,359] | **0,624 [0,521;0,662]** | **0,468 [0,360;0,539]** |
| Block scan | 0,450 [0,439;0,767] | 0,537 [0,518;0,665] | **2,014 [1,953;3,334]** | **2,095 [2,021;4,446]** |

**Цена переносимости/корректности не скрыта:** в этой серии warm scan execute
медленнее примерно на 19%, PHT с occupancy — на 4%; SUM имеет меньшую медиану.
GPU-примитивы не объявляются безусловно быстрее baseline. В full summary больших
результатов заметен fetch, поэтому время ядра и полный путь не взаимозаменяемы.

Native prebuilt build, **мс, независимые медианы**, n=3; не часть summary:

| Вариант / режим | acpp -c wall / user / system | Link wall / user / system | **summary** |
| --- | ---: | ---: | --- |
| baseline plain | 3296,568 / 3128,339 / 147,114 | 70,679 / 57,849 / 28,042 | Не измеряется сборкой |
| baseline traced | 3599,623 / 3430,987 / 157,822 | 71,820 / 56,203 / 23,363 | Не измеряется сборкой |
| PCUDA plain | 485,451 / 435,754 / 46,570 | 70,550 / 51,740 / 26,712 | Не измеряется сборкой |
| PCUDA traced | 487,375 / 435,458 / 43,527 | 70,669 / 55,989 / 20,201 | Не измеряется сборкой |

Меньшее время компиляции относится к **этому prebuilt TU без PCH**, а не к полному
SQL JIT. Размер бинарника baseline — 593 688 B, PCUDA — 113 544 B. Все raw значения,
min/max/user/system, startup и native process wall —
[`benchmark-build-distributions.json`](../../build/pcuda-stage4/run-ecd8a23n/benchmark-build-distributions.json).
Startup main→runtime-ready median plain: baseline 264,060, PCUDA 270,153 мс;
data preparation/upload — 2,467 / 2,452 мс. Это не полный spawn→CPU-ready.
Native process wall охватывает все 36 запусков, печать и teardown; полный старт
до первого CPU-result отдельно не измерен. PCH/dlopen/dlsym здесь отсутствуют.

Влияние диагностики не вычитается. Наблюдаемая разность медиан compile wall
trace−plain: baseline +303,055 мс, PCUDA +1,924 мс; n=3 и неконтролируемое thermal
состояние не позволяют приписать всю разность только trace overhead.

| Операция | baseline **summary traced**, мс | PCUDA **summary traced**, мс | PCUDA warm execute traced, мс |
| --- | ---: | ---: | ---: |
| Load+predicate | **83,072 [79,770;86,463]** | **82,203 [76,344;90,888]** | 0,237 |
| PHT | **170,315 [165,821;201,355]** | **194,194 [188,246;210,754]** | 0,339 |
| SUM | **160,863 [154,480;172,087]** | **78,270 [76,490;91,549]** | 0,305 |
| Scan | **3,442 [2,374;191,193]** | **2,749 [2,339;101,632]** | 0,536 |

Остальные trace/plain распределения — в основном JSON. Сохранены 12 оригинальных
Clang/LLD trace-файлов; отдельный offline-разбор — `benchmark-trace-analysis.json`.
Вложенные Frontend/Backend/Optimizer, linker trace и native wall не складываются.

## 5. Корректность и ограничения

### Исправленные причины ошибок

- Старый `HASH` вычитал signed ключи до widening и мог дать отрицательный индекс.
  Теперь modulo считается без signed overflow, key compare отвергает probe alias.
- Нулевая PHT-ячейка ошибочно совпадала с ключом 0. Presence теперь независима от key.
- Несколько MHT-потоков писали одинаковые directory fields неатомарно. Теперь
  запись выполняет только поток с локальной payload-позицией 0.
- Nullable FK, не участвующий в SELECT/WHERE, не попадал в validity-регистры.
  После регистрации join nullability выявилась потеря декларации transient mask.
  Учёт написанных деклараций отделён от живых column bindings; CPU/codegen и SQL
  тесты фиксируют обе причины, NULL не считается нулевым ключом.
- Union type-punning double заменён bit-preserving memcpy; scalar arithmetic
  promotion централизован, без принудительного перевода Int64 в double.

PHT/MHT по-прежнему требуют perfect-domain build keys: разные build keys не
должны иметь один residue. PHT не предназначен для SQL multiplicity — её обслуживает
MHT. MHT directory/counts/row IDs остаются Int32, поэтому план отвергает build
rows >INT32_MAX **до kernels**; scan/table byte extents используют checked arithmetic.
Новый global scan шире, но legacy SQL scan limits (MHT до 256^3 slots и projection
до 1024 block sums) пока сохранены как явные Require, не молчаливое переполнение.
Размеры/выходная ёмкость и смена алгоритмов по параметрам остаются этапу 6.

У signed SQL Int64 арифметики здесь не вводится новая overflow/zero-divide политика:
`safe_*` означает прежнее promotion промежуточного Int32/Float32, а не произвольную
арифметику без границ. Bitmap atomics работают с выровненными raw GPU words;
atomic64/FP64 проверяются runtime. Межъядерная видимость обеспечивается stream,
а не relaxed atomics; concurrent build/probe не обещается.

Q1.1 проверен новым независимым C++ CPU oracle по 59 986 217 строкам:
SQL **4486422947100**, эталон/1993-only **4474878955766**, включённый SQL
`1994-01-01` даёт разницу **11543991334**. SQL, данные и эталон не менялись:
[`q11-oracle.json`](../../build/pcuda-stage4/run-ecd8a23n/q11-oracle.json).
SF10 legacy raw-prefix/zero-tail storage view этапа 3 также не менялся.

### История отрицательных проверок

- `host-red.stderr`: header load не был самодостаточен по uint64_t, отсутствовал
  новый hash/layout API; тест предшествовал реализации.
- `optimizer-red.stderr`: ещё отсутствующий occupancy-контракт.
- `gpu-fixture-xkvt6d7j`: пропущенная nullable FK mask; `gpu-fixture-8zak7bjo`:
  её необъявленный transient register. Оба raw failures сохранены.
- `ub/crapper-v1`: неинстанцированные comparison adapters; добавлены поведенческие
  проверки всех адаптеров, scope не сужался. `dryer-host-v1/v2` выявили повторение
  bounds/promotion; общие политики вынесены в transform/gather/mask helpers.
- `dryer-device-v1`: обычному Clang явно не передан public PCUDA include path.
  Следующие AST запуски передают его и `-fdeclspec`, не маскируют ошибку compiler.
- `dryer-device-v2/v3`: среди настоящих похожих build/probe loops были сравнения
  функции с её собственной вложенной lambda. Добавлен failing self-test и
  исключение только содержащихся byte ranges одного файла; разные lambda/functions
  продолжают сравниваться. Это не изменение порогов 0,8/50.
- `primitive-bench-quirejrs` и `primitive-bench-e2mwaab1`: соответственно missing
  dependent `template` в baseline adapter и `<stdexcept>` в минимальном PCUDA
  benchmark TU; исправлены входные sources, не compiler checks.
- `primitive-bench-rvxnfti_` — историческая **V1** серия: oracle был исключён из
  каждого summary, но выполнялся между warm интервалами внутри native процесса.
  Для строгого разделения сделана **новая V2**, без смешивания распределений.
  Старый report/commands/timers/binaries сохранены, V1 input sources восстановлены
  и проверены по сохранённым SHA-256 в `source-evidence/benchmark-v1-verification.json`.

## 6. Выводы и решения

Ресурсы остаются у runtime, fused алгоритмы сохранены. Примитивы готовы к отдельным
PCUDA kernels, но целевая модульная SQL-архитектура ещё не реализована. Явная
occupancy — согласованное решение пользователя; новые SQL-типы, replay, постоянный
module cache и новый SLA не добавлялись.

Числа 600/200/10/9 мс относятся только к стенду этапа 0 и здесь не принимаются
заново. Бюджет SQL→PCUDA ≤10 мс и полный SQL cold/warm SLA остаются интеграционной
проверке. Новые microbench timings нельзя подставлять вместо них.

## 7. Воспроизведение и оставшаяся работа

Сборка — командой `configure.command.json`, затем:

```bash
cmake --build build/pcuda-stage4/run-ecd8a23n/engine -j4
ctest --test-dir build/pcuda-stage4/run-ecd8a23n/engine --output-on-failure
build/pcuda-stage4/run-ecd8a23n/engine/test_pcuda_primitives_gpu
python3 test/pcuda_stage2/run_fixture.py \
  --binary build/pcuda-stage4/run-ecd8a23n/engine/test_pcuda_primitives_sql \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --output-root build/pcuda-stage4
python3 test/pcuda_stage4/benchmark.py \
  --output-root build/pcuda-stage4 --engine build/pcuda-stage4/run-ecd8a23n/engine \
  --baseline-include build/pcuda-stage4/run-ecd8a23n/baseline/include \
  --acpp /opt/AdaptiveCpp/bin/acpp --cxx /opt/rocm_sdk_612/bin/clang++ \
  --sdk /opt/AdaptiveCpp --gcc-install-dir /usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  --cold 7 --build-repeats 3
```

Общие `tools/ub/coverage.py`, `crapper.py`, `dryer.py`, `mutator.py` используют
`host-recipe.json` / `gpu-recipe.json`; paths toolchain передаются flags извне.
Мутации строятся с нуля в независимых копиях, build failures/timeouts не kills.
Полный журнал команд/причин — `actions.jsonl` и `ub/*/actions.jsonl`.

Полные хеши исходников, compiler/runtime, бинарников/PCH/SQL bundles и список
traces — `final-evidence.json`; собственный diff относительно начальной рабочей
копии — `stage4-only.diff`, аудит сохранности остальных исходников —
`preservation-final.json`. Первый optimizer TDD red использовал старый stage-3
build-каталог: только перезаписанные нами derived optimizer object/binary
восстановлены из начального snapshot с успешной проверкой, см.
`restore-own-red-build-v2/`. Старые raw reports/traces не переписаны.

Следующие этапы должны подключить эти primitives к модульному codegen, заменить
legacy SQL collectives, перенести runtime-dependent shapes и ручные примеры.
Этап 5 автоматически не начинался. Commit/push и очистка чужих артефактов не
выполнялись; исходная незакоммиченная работа не включается в отдельный commit.
