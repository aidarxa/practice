# PERF-COLD: компиляция PCUDA и холодный старт

## 1. Статус и границы

Исследование выполнено 2026-10-30 на AMD Radeon 780M: инвентаризация установленного
toolchain, разбор прежних результатов, однофакторные проверки, сочетания,
контрольные серии с/без traces и независимая проверка вычислений. **Это не перенос
движка на PCUDA.** Движок остаётся SYCL, этап 1 не начат, ABI не утверждён.
**2026-10-31 пользователь утвердил абсолютные показатели ограниченного стенда
Q3.3 SF10/V2 и закрыл этап 0.** Точное решение — [раздел 6](#stage0-agreement).
Replay отложен в необязательный этап 9, только если останется время после этапов
1–8; его исследовательские результаты не означают готовую интеграцию.

Основной практически подтверждённый результат: при готовом PCH независимые четыре
PCUDA-модуля выгоднее готовить **параллельно**, а не последовательно. В итоговой
серии **V2 с подготовкой запроса внутри таймера** `summary` прототипа снизился с
**890,400 до 469,289 мс**, с сохранением обычных вызовов `acpp`.
Экспериментальный replay полных effective команд Clang/LLD дал **371,793 мс**.
Ранние V1-границы и позднее замедление сохранены отдельно, не объявляются SLA и
не подставляются вместо полного V2 `summary`.

Все эти `summary` относятся к четырёхмодульному Q3.3 с 24 компактными группами,
**без SQL parse/plan/codegen**, после загрузки SF10. Это не end-to-end SQL latency
и не изолированное сравнение SYCL с PCUDA. SQL-генератор, `src/`, `include/`,
основная CMake-сборка и пользовательские debug/PCH-эксперименты не изменялись.
Постоянный кеш SQL-модулей не добавлялся, установленный toolchain не модифицировался.

### Полный реестр артефактов

Все пути ниже локальны; `build/` не является переносимым опубликованным хранилищем.
Каждый каталог содержит исходные команды, stdout/stderr, native timers,
PCH/объекты/библиотеки, Clang/LLD JSON и отдельные device-cache каталоги.

| Набор | Артефакт | Объём/назначение |
| --- | --- | --- |
| Начальная инвентаризация | [inventory-aum9g2iv](../../build/pcuda-compile-perf/inventory-aum9g2iv/inventory.json) | help, hidden options, dry-run, документация |
| Дополненная инвентаризация | [inventory-9dfa9bv9](../../build/pcuda-compile-perf/inventory-9dfa9bv9/inventory.json) | также gold, transitive runtime libraries, bitcode/headers hashes, AMD JIT sources |
| Исторический разбор | [history-audit.json](../../build/pcuda-compile-perf/history-audit.json) | три исходных отчёта этапа 0, без переписывания |
| Пробный запуск | [series-aoryfbzi](../../build/pcuda-compile-perf/series-aoryfbzi/report.json) | 5 вариантов × 2 diagnostics × 1 cold × 3 выполнения |
| Однофакторный screening | [series-6yi76ncc](../../build/pcuda-compile-perf/series-6yi76ncc/report.json) | 18 вариантов × 2 × 3 cold × 9 выполнений |
| Сочетания, подтверждение | [series-ydtefe2b](../../build/pcuda-compile-perf/series-ydtefe2b/report.json) | 6 вариантов × 2 × 7 cold × 9 выполнений |
| Runtime-настройки | [series-e_2b8uoh](../../build/pcuda-compile-perf/series-e_2b8uoh/report.json) | 4 варианта × 2 × 3 cold × 9 выполнений |
| Дополнительные линкеры/ELF | [series-9q99u37u](../../build/pcuda-compile-perf/series-9q99u37u/report.json) | 5 вариантов × 2 × 3 cold × 9 выполнений |
| Поздний повтор replay/adaptivity | [series-qe5drg9f](../../build/pcuda-compile-perf/series-qe5drg9f/report.json) | 2 варианта × 2 × 7 cold × 9 выполнений |
| Итоговая граница V2 | [series-0je32lqk](../../build/pcuda-compile-perf/series-0je32lqk/report.json) | 3 варианта × 2 × 7 cold × 9; подготовка запроса включена в summary |
| Примитивы и unload | [validation-dsg6peey](../../build/pcuda-compile-perf/validation-dsg6peey/report.json) | baseline/replay × 3, 7 модулей на набор |
| Int64/Float64/NULL | [types-bench-7p059jnq](../../build/pcuda-compile-perf/types-bench-7p059jnq/report.json) | lazy/tags/eager, 3 серии, correctness-only replay driver |
| Лимит 8 + fallback | [types-bench-g8422clm](../../build/pcuda-compile-perf/types-bench-g8422clm/report.json) | четыре параметра, 3 серии |
| SQL-контроли и 13 SSB | [sql-controls-0jigbesf](../../build/pcuda-compile-perf/sql-controls-0jigbesf/report.json) | порядок запросов, linker O0/default, runtime-log, 13 SSB |
| Полный offline-экспорт | [analysis-2ykgkys8/summary.json](../../build/pcuda-compile-perf/analysis-2ykgkys8/summary.json), [tables.txt](../../build/pcuda-compile-perf/analysis-2ykgkys8/tables.txt) | V1/V2 отмечены отдельно; все распределения, native user/system, PCH, trace-фазы, исправленная агрегация SQL |
| Экспорт до V2 | [analysis-tot36nqz](../../build/pcuda-compile-perf/analysis-tot36nqz/summary.json) | все V1-серии, не полный итог V2 |
| Более ранний offline-экспорт | [analysis-_8plkaoa](../../build/pcuda-compile-perf/analysis-_8plkaoa/summary.json) | сохранён до дополнительных linker/replay-серий; не полный итог |
| Повторный Q1.1 CPU oracle | [q11-reference.stdout](../../build/pcuda-compile-perf/analysis-_8plkaoa/q11-reference.stdout), [команда/хеш](../../build/pcuda-compile-perf/analysis-_8plkaoa/q11-reference-command.json) | независимая проверка расхождения с 11.txt |
| Проверка файлов V1 | [final-check-5k5qzbg1](../../build/pcuda-compile-perf/final-check-5k5qzbg1/checks.json) | прежняя проверка и сохранённые V1 host sources |
| Итоговая проверка V1/V2 | [final-v2-check-4cy45uer](../../build/pcuda-compile-perf/final-v2-check-4cy45uer/checks.json) | syntax, ссылки, module/source hashes с учётом версии, V2 timing coverage, CPU test logs и copies sources |

Итого PCUDA Q3.3: **326 новых процессов, 2874 проверенных результата**
(V1: 284/2496; V2: 42/378).
Одинаковые варианты разных серий не объединяются в одну статистику.

## 2. Окружение, конфигурация и каталог кандидатов

- AdaptiveCpp `25.10.0+git.5989e1cf.20260821.branch.HEAD`, checkout
  `5989e1cfa4e70d55c65e786a4c71acb124ce54ff`, generic PCUDA; HIP/OpenMP backends.
- Clang/LLVM **17.0.0**, ROCm LLVM revision
  `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`; LLD **17.0.0**.
  Установленные BFD/gold версии и help сохранены в inventory. `mold` отсутствует;
  ничего не устанавливалось. HIP SDK `/opt/rocm_sdk_612`, HIP 6.1.40093.
- AMD Ryzen 7 7840HS, 8 ядер/16 потоков, AMD Radeon 780M (`gfx1103`), около 30 GiB RAM.
  Runtime подтвердил FP64/atomic64. NVIDIA/Intel аппаратно не проверялись.
- SF10: 59 986 217 fact-строк; загружаются десять нужных int32-колонок.
  Device resident budget Q3.3 — 963 884 644 байт. Pinned/device память и stream — RAII.
- GPU-модули: C++17, `--acpp-targets=generic --acpp-pcuda -fPIC -g0 -O0`;
  явно выбран `--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1`.
  Это параметры данного запуска, не новые hardcoded настройки движка.
- Host: `-O3 -DNDEBUG`. Минимальный PCH: `test/pcuda_stage0/pch.hpp`,
  `-fpch-instantiate-templates -x c++-header`, затем `-include-pch`.
  PCH создаётся с совместимыми flags отдельно от запроса; размер baseline — 4 237 904 байт.
- Базовая линковка: отдельный `acpp -shared -fuse-ld=lld -Wl,-O0` для каждого `.o`.
  `-Wl,-O1` меняет ELF linker optimization, а не уровень оптимизации GPU-кода.
- `LC_ALL=C`; для новых PCUDA performance-серий явно
  `ACPP_ADAPTIVITY_LEVEL=1`, `ACPP_ALLOCATION_TRACKING=1`, кроме соответствующих кандидатов.
  Прочие существенные переменные, версии, kernel/CPU/load snapshots — в inventory.
  Частоты/температуры не фиксировались для каждого прогона; power/thermal-состояния
  между ранними и поздними сериями не доказаны одинаковыми.

SHA-256 compiler/plugin/runtime, транзитивных HIP/HSA-зависимостей, PCUDA headers и
device bitcode — в дополненном inventory. Hashes исходников, host-бинарников,
использованных данных — в `report.json.sha256`; каждого модуля/PCH — в session.
Полные исходные/effective команды: `commands.json`, `*-compile-dryrun.txt`,
`*-link-dryrun.txt`, `*-lld-dryrun.txt`, фактически исполняемый `build.sh` каждой session.

### Что именно означает replay

`--acpp-dryrun` сохраняет полный Clang argv, включая PCUDA macros, include paths,
`-mllvm -acpp-sscp -acpp-sscp-pcuda`, `-disable-O0-optnone`, оба plugin flags.
Для LLD команда извлекается из Clang `-###`, с CRT, порядком runtime/stdlib libraries
и rpath. При dry-run ещё не существующего объекта подставляется готовый host.o,
после разбора меняется **только** этот operand обратно на нужный module.o.

Replay исключает повторную работу acpp/Clang-драйверов в выбранных стадиях; он не
изменяет ядра и не означает, что `acpp` стал компилировать быстрее. Его compile/link
колонки — **native Clang/LLD wall**, не `acpp wall`. Команды подготавливаются до
запуска измеряемого host, как экспериментальная преднастроенная конфигурация.
Стоимость этой подготовки не выдаётся за runtime-ускорение SQL; прототип не делает
codegen. Для интеграции нужны version/hash/schema checks и генерация argv под
реальный модуль, а не буквальное копирование Linux CRT-путей в движок.

### Источники

Прочитаны все help/hidden inventories установленного acpp, Clang driver/cc1/LLVM
и LLD, справка BFD/gold, установленный acpp-driver и исходники соответствующего commit.
Каталог ниже группирует потенциально относящиеся к latency опции, а не обещает
перебор декартова произведения всех LLVM flags. Полные исходные списки сохранены.

- [AdaptiveCpp PCUDA](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/5989e1cfa4e70d55c65e786a4c71acb124ce54ff/doc/pcuda.md),
  [performance](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/5989e1cfa4e70d55c65e786a4c71acb124ce54ff/doc/performance.md),
  [environment](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/5989e1cfa4e70d55c65e786a4c71acb124ce54ff/doc/env_variables.md).
- [Clang 17 UsersManual](https://releases.llvm.org/17.0.1/tools/clang/docs/UsersManual.html),
  [flags](https://releases.llvm.org/17.0.1/tools/clang/docs/ClangCommandLineReference.html),
  [PCH internals](https://releases.llvm.org/17.0.1/tools/clang/docs/PCHInternals.html).
- [LLD 17 ELF options](https://github.com/llvm/llvm-project/blob/llvmorg-17.0.0/lld/ELF/Options.td).
  Копии документации с URL/SHA-256 сохранены в `inventory-*/docs/`.

| Область / flags | Поддержка и решение |
| --- | --- |
| `-O0`, `-O1/-O2/-O3`, `-Os/-Oz` | O0 — обязательный baseline. Прежний O3/PCH контроль сохранён в этапе 0; runtime generic JIT оптимизирует отдельно. Новая полная O-matrix не повторялась. |
| `-g0`, `-g`, split DWARF, `-fpch-debuginfo` | Baseline без debug info. Удалять отсутствующую debug info нечего; PCH-debuginfo здесь неприменим. |
| `-include-pch`, минимальные headers | Проверены с/без PCH; большой эффект на frontend. Не подключать `sycl/sycl.hpp`. |
| `-fpch-instantiate-templates` | Проверено включение/отсутствие. Без него отложенные инстанцирования возвращаются в каждый модуль. |
| PCH `q33_build.hpp` вместо `pch.hpp` | Проверено расширение preamble: дополнительного убедительного выигрыша нет. |
| `-fpch-codegen`, explicit instantiation / `extern template` | Не измерялись: требуют отдельного PCH object/контракта; module-local kernel/lambda identity не становится общей заранее известной специализацией. После обычного PCH инстанцирования уже малы. |
| `-fno-implicit-templates`, отключение PCH validation | Не допустимые ускорения: риск пропущенного device-кода/невалидного PCH. Проверки совместимости не отключались. |
| `-fmodules*`, C++ modules/header units | Clang modules потребуют отдельного подтверждения AdaptiveCpp/PCH/HCF совместимости; не проверены. C++20 modules/header units не входят в строго C++17. |
| `-fno-exceptions`, `-fno-rtti` | Проверены только на не использующих их GPU-модулях, не на host/runtime. Убедительного latency-выигрыша нет; не production-рекомендация. |
| `-fno-unwind-tables -fno-asynchronous-unwind-tables` | Проверены: небольшое уменьшение ELF, не убедительное ускорение; цена — диагностика/unwinding. |
| `-fomit-frame-pointer` | Проверен, выигрыша нет. |
| `-fno-vectorize`, `-fno-slp-vectorize`, unroll/inline thresholds | Не отдельная новая серия: обычный host O0 уже не имеет O3-проходов; flags stage 1 не управляют отдельным runtime O3. Произвольные hidden-pass tweaks не подтверждены полезными. |
| `-Xclang -disable-llvm-passes`, `-mllvm -opt-bisect-limit` | Не допустимо вырезать обязательные AdaptiveCpp target-separation/registration passes ради короткой компиляции. |
| `-mllvm -acpp-sscp-preoptimize` | Проверен отдельно и в сочетании. Help прямо помечает developer-only/NOT supported; увеличивает stage-1 optimizer, не даёт cold-выигрыша. |
| `--acpp-targets=generic --acpp-pcuda` | Сохранены. Другие targets не поддерживают этот PCUDA-путь; выбор GPU arch при stage 1 generic невозможен. |
| `--acpp-pcuda-chevron-launch` | Исключён: добавляет preprocessing и не нужен native `pcudaParallelFor`. |
| `--acpp-export-all`, explicit multipass | Не нужны одному kernel+wrapper/TU; export-all расширяет экспортируемый device-код, не является ускорением этого случая. |
| `--acpp-stdpar*`, GPU/OpenMP legacy backend flags | Не относятся к измеряемому native PCUDA generic-пути; не подменять ими задачу. |
| `--acpp-use-accelerated-cpu` | Не целевой тюнинг AMD generic PCUDA: в effective GPU-командах нет legacy CPU compilation flow. |
| `--acpp-clang`, GCC/include/config paths | Выбор совместимого toolchain, не самостоятельное ускорение. Clang 17 + GCC16 несовместимость не обходится ослаблением проверок. LLVM upgrade требует отдельного стенда. |
| `--acpp-save-temps`, deps `-MD/-MF` | Сохранение/диагностика, не ускорение. Dependency checks оставлены одинаковыми для baseline/candidates. |
| `-fintegrated-cc1`, integrated assembler | Установка уже использует обычный integrated путь. External assembler добавит процесс; такой вариант не измерялся и не объявляется ускорением. |
| `-flto`, ThinLTO, linker LTO jobs | Baseline без host LTO; добавлять дорогую link-time оптимизацию не обосновано. Это не управление HCF/device JIT. |
| `-fuse-ld=lld/bfd/gold` | Все три измерены. mold отсутствует; его преимущество не утверждается. |
| `-Wl,-O0/-O1` | Измерены. Не путать с GPU O0/O3. |
| `-Wl,--threads=1/4` | Измерены; в полном acpp link driver overhead доминирует, отдельный большой выигрыш не доказан. threads=1 включён в лёгкий replay-профиль для ограничения потоков. |
| `--gc-sections`, `--icf=safe`, `--strip-all`, `--hash-style=sysv` | Измерены; уменьшение размера не превратилось в убедительное cold-ускорение. GNU hash — исходный default. |
| `--icf=all`, `-Bsymbolic`, visibility/interposition, `-fno-use-cxa-atexit` | Не переносились: могут менять identity/экспорт/teardown semantics. Нельзя ломать registration ради ELF-оптимизации. |
| `--no-eh-frame-hdr`, relax knobs | Не измерялись: диагностический/target-specific компромисс при малых wrappers; более сильного основания после traces нет. |
| `-nostdlib*`, `-nodefaultlibs`, `-nostdinc++` | Не сохраняют требуемые headers/runtime/CRT автоматически. В replay полный link argv сохранён, библиотеки не удалены. |
| `-Ofast`, `-ffast-math`, unsafe/finite math, no-signed-zeros | Исключены как общие SQL-настройки: недопустимо менять точность, NaN/Inf/порядок арифметики ради скорости. |
| `ACPP_ADAPTIVITY_LEVEL=0/1/2`, `ACPP_ALLOCATION_TRACKING=0/1` | Измерены отдельно, adaptivity=0 также с replay. Это runtime-настройки, а не ускорение acpp frontend. |
| Device JIT cache, предварительный запуск ядра | Не скрывать работу перед summary. Новый device-cache на процесс; никакого предварительного прогрева исследуемых ядер. |
| Replay compiler/linker argv, параллельность 4 | Измерены отдельно, затем в сочетаниях; остаются экспериментом, не новой production-конфигурацией. |
| `-ftime-trace`, LLD `--time-trace`, debug logging/strace | Диагностика с overhead, не ускоряющие flags. strace не использовался в новых performance-сериях. |

В исходниках установленного commit runtime `LLVMToBackend::optimizeFlavoredIR`
строит O3 pipeline. AMD lowering выбирает hipRTC либо clang по **build-time**
`ACPP_HIPRTC_LINK`; `getClangPath()` использует build-time configuration.
Не обнаружен поддерживаемый пользовательский переключатель «S2/JIT O0».
Изменение/пересборка этой части LLVM/AdaptiveCpp не проводились и остаются отдельной
гипотезой, а не скрыто применённым флагом. Не следует из этого считать первое
execution чистым временем JIT или приписывать ему конкретные внутренние доли.

## 3. Методика

### Таймеры и границы

`q33_cold.cpp`, **итоговый V2**: выбор устройства/startup → read/H2D десяти колонок
с ожиданием → **один непрерывный `steady_clock` summary**: проверка
ключей/aggregate bounds, расчёт доменов и подготовка query-specific result/scratch
буферов → native build script → четыре
`dlopen/dlsym` → memset/build/build/build/fused probe+aggregate → ожидание → D2H →
ожидание готового CPU-результата. Форматирование и проверка SF10 выполняются после.
SQL parse/plan/codegen отсутствуют. Исходники ядер готовы до запроса, PCH заранее.

Build script вызывает реальные компиляторы/линкеры внутри summary. Bash native
`time`, `TIMEFORMAT` с шестью десятичными знаками секунд, `LC_ALL=C` сохраняет
wall/user/system **каждой команды**. Собственный Python-драйвер acpp включён
полностью. Дополнительного Python-таймера/прокси в измеряемой цепочке нет.
Python организует серии и разбирает JSON/traces только после завершения серии.

- `compile_wall_sum_ms`, `link_wall_sum_ms`: суммы native wall четырёх команд;
  CPU user/system также сохраняются. Для parallel это **не elapsed**.
- `build_elapsed_ms`: прямой C++ интервал от spawn Bash script до завершения всех
  compile/link. Включает shell/spawn/wait и промежутки, не анализ traces.
- `load_ms`: четыре dlopen/registration/dlsym, cached launch pointers дальше.
- Первое `execute_ms`: четыре memset, три build-ядра, fused probe/aggregate,
  один финальный stream wait; возможный device JIT внутри. Нет wait между ядрами.
- `fetch_ms`: D2H и wait перед CPU-чтением. Warm `total_ms` измеряется напрямую
  от submission до CPU-ready; warm execute — отдельный более узкий интервал.
- `summary_ms` V2: непрерывный первый query path после data-ready,
  **включая preparation**, не сумма.
- `first_result.cpp`: начинает parent steady_clock **перед posix_spawn host**,
  заканчивает при CPU-ready marker по pipe. Учитывает loader/startup, данные,
  подготовку и первый запрос, но не ожидание пользовательского ввода/печать/teardown.
  Есть небольшая неизбежная доставка marker/планирование parent; отдельное
  `main_to_first_result_ms` позволяет не смешивать границы.
- Полный native process wall включает восемь warm-повторов и teardown и поэтому
  **не называется полным стартом до первого результата**.

Параллельно запускаются четыре независимые compile→link цепочки; одна `.so` на
одно ядро сохранена. Sum wall команд и непрерывный elapsed показаны раздельно.
`summary_minus_components_ms` при overlap не является проверкой покрытия. Проверка
покрытия V2 — разность summary и **preparation + build elapsed + load + first total**, а не
принудительное обнуление остатка или сложение медиан.

### Кеши, порядок, диагностика

Для каждого варианта/diagnostics/cold — новый процесс и отсутствующий до старта
`ACPP_APPDB_DIR`; внутри девяти выполнений он сохраняется. SQL/module cache пуст
при первом запросе; четыре модуля загружаются один раз. OS/filesystem caches не
очищались. Более ранние прогоны могли прогреть системные библиотеки/файлы.
HIP/driver caches отдельно не очищались: пустой AdaptiveCpp appdb не является
обещанием пустоты всех нижележащих кешей устройства.
Пробный запуск не исключён из истории, но не включён в подтверждающую статистику.

Порядок вариантов циклически сдвигается между сериями, порядок trace/plain
чередуется. Screening: три независимых cold на режим, warm 24; подтверждение:
семь cold, warm 56. Поздний replay-повтор — отдельная серия с тем же числом,
не дополнение «лучших» значений предыдущей. CPU/GPU clocks не закреплялись,
фоновые desktop-нагрузки возможны. Для небольших отличий n=3 недостаточно для
заявления статистически подтверждённого выигрыша; bootstrap/p-value не заявлялись.

Clang `-ftime-trace=<owned file>`, LLD `--time-trace=<owned file>`, granularity
не переопределялась. Для BFD/gold режима LLD trace нет по определению, Clang trace
есть; эти строки не являются измерением LLD. Runtime DEBUG=3 — отдельный SQL
диагностический запуск, не performance baseline. Trace-parent/child интервалы
никогда не суммируются между собой; ниже суммы только **одноимённых стадий разных
компиляций**, не elapsed параллельной сборки.

## 4. Результаты

Все времена — **мс**. Если не оговорено иначе: медиана; `[min; max]` — наблюдаемый
разброс, не доверительный интервал. Все отдельные samples и отсутствующие значения
сохранены в raw reports и полном offline-экспорте из раздела 1.

**История границ:** PCUDA-таблицы 4.2–4.6 относятся к V1: их raw `summary_ms`
начинался после query-specific preparation. Это прямой, но **более узкий**
prepared-buffer интервал, а не полный обязательный `summary` первого запроса.
Полный `summary` у V1 не измерен; его нельзя восстанавливать суммой стадий.
Сами raw поля/числа не переписаны. Итоговый **V2 — раздел 4.9**: таймер начинается
сразу после data-ready, preparation входит внутрь. SQL-таблица 4.7 всегда измеряла
полный SQL total напрямую и этой коррекции не требует.

### 4.1. Исторические расхождения: проверено по исходным данным

| Исторический SQL Q3.3 O0/PCH | Native acpp compile + link | summary = cold total_query_ms | Host fetch |
| --- | ---: | ---: | ---: |
| `sql-matrix-oe5qo0yx`, n=3 | 2775,087 [2772,573; 2812,463] | 3447,548 [3435,102; 3472,623] | 90,113 [87,288; 91,074] |
| `baseline-series-g4b_dmw4`, n=3 | 2771,103 [2767,512; 2823,871] | 3368,742 [3352,251; 3408,510] | 0,582 [0,535; 0,583] |

Это близкая скорость **одного SQL compile+link пути**, не многократное расхождение
компиляции. Проверены все шесть subreports: совпадают сохранённые hashes исходников,
совпадает binary SHA-256
`2b27a7a1913cc60acc73fb911625f057b90f52a67d47749b15cbc0027132f321`,
текущие `src/include` совпадают с этими snapshots. Все 12 соответствующих compiler/
linker trace-файлов доступны; их hashes/интервалы — `historical_crosscheck` offline-экспорта.
В matrix Q3.3 первый запрос, в baseline перед ним восемь запросов; первый большой
grouped fetch там приходится на Q2.1. Matrix задаёт `-Wl,-O0`, baseline — нет.

Исторические **1027,076 мс PCUDA Q3.3** — `cold_module_path_ms`, сумма компонентов,
а не прямой summary: compile 330,841, link 320,877 (медианы отдельно, их нельзя
складывать для восстановления медианы cold). Прототип использует 24 компактные
группы и не имеет SQL-стадий. Сравнение с 3368,742 мс SQL **не изолирует** flags,
алгоритм либо смену API. Старое поле не переименовано.

### 4.2. V1: сочетания с подготовленными буферами, 7 cold / 56 warm, без traces

Полный обязательный `summary` для V1 **не измерен**; ниже сохранён более узкий
raw prepared path (`summary_ms` в историческом JSON), не подстановка вместо него.

| Конфигурация | Compile Σ wall | Link Σ wall | Build elapsed [min; max] | **V1 prepared path [min; max]** |
| --- | ---: | ---: | ---: | ---: |
| baseline: acpp/acpp, последовательно | 275,445 | 272,720 | 551,220 [542,734; 570,101] | **847,645 [837,816; 909,745]** |
| fast-link: acpp/LLD, threads=1 | 274,111 | 55,186 | 333,434 [323,693; 357,896] | **627,376 [621,276; 685,295]** |
| parallel4: acpp/acpp | 266,586 | 264,265 | 136,147 [135,273; 137,259] | **436,327 [428,878; 440,686]** |
| parallel-fast-link: acpp/LLD | 268,366 | 49,402 | 82,799 [81,848; 84,515] | **381,150 [375,378; 385,568]** |
| replay-parallel: Clang/LLD | 89,218 | 48,817 | 38,606 [37,568; 42,452] | **337,226 [332,240; 388,390]** |
| + unsupported preoptimize | 115,653 | 49,811 | 55,062 [52,726; 55,903] | **352,174 [349,565; 359,658]** |

Первый/второй driver указаны для compile/link. Σ wall при parallel характеризует
работу команд, не latency. Все четыре `.so` в обычных вариантах вместе — **61 696
байт**, preoptimize — **64 384**. Один и тот же алгоритм/аргументы/порядок ядер.

| Конфигурация | Load | Первое execute | Fetch | Warm execute [min; max] | Warm total |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0,497 | 297,266 | 0,042 | 7,835 [6,235; 9,171] | 7,871 |
| fast-link | 0,319 | 295,721 | 0,045 | 8,069 [6,293; 9,170] | 8,141 |
| parallel4 | 0,324 | 299,698 | 0,042 | 7,926 [6,287; 9,195] | 7,961 |
| parallel-fast-link | 0,337 | 297,109 | 0,043 | 7,993 [6,280; 9,149] | 8,035 |
| replay-parallel | 0,322 | 298,972 | 0,043 | 7,843 [6,238; 9,196] | 7,896 |
| + preoptimize | 0,672 | 296,125 | 0,042 | 7,915 [6,292; 9,159] | 7,951 |

Native CPU totals показаны отдельно; каждый компонент — собственная медиана,
`user+system` не подменяет wall и не прибавляется к нему:

| Конфигурация | Compile Σ wall / user / system | Link Σ wall / user / system |
| --- | ---: | ---: |
| baseline, acpp/acpp | 275,445 / 200,220 / 72,831 | 272,720 / 204,973 / 99,983 |
| parallel4, acpp/acpp | 266,586 / 195,887 / 68,597 | 264,265 / 193,987 / 94,135 |
| replay-parallel, Clang/LLD | 89,218 / 51,965 / 38,729 | 48,817 / 24,965 / 23,302 |

Разница native acpp wall и Clang trace — **не доказанное чистое время Python**:
есть driver/startup, загрузка plugin/libraries, I/O, scheduling и неохваченные интервалы.
Сопоставление replay подтверждает выгоду устранения повторных driver invocations,
не точное разбиение каждой миллисекунды остатка.

### 4.3. V1: PCH и полный старт процесса

| Конфигурация | PCH prepare wall [min; max], отдельно | Startup | Data loading | Preparation | **V1 prepared path** | Spawn → CPU-ready [min; max] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 284,189 [265,372; 347,753] | 241,220 | 174,619 | 33,253 | **847,645** | 1305,084 [1287,491; 1359,083] |
| parallel4 | 280,300 [260,397; 350,045] | 240,554 | 172,879 | 33,216 | **436,327** | 889,234 [876,109; 897,218] |
| replay-parallel | 265,555 [260,422; 345,286] | 240,653 | 173,264 | 33,231 | **337,226** | 783,712 [780,680; 918,867] |

В этих V1-строках preparation находится **вне** узкого raw summary, но **внутри**
прямого process-start интервала. PCH функционально один и тот же prebuilt input, его случайный разброс
подготовки не означает оптимизацию разными runtime-профилями. PCH native user/system
для **всех** вариантов — в offline JSON. Стоимость не прибавляется к каждому query.
Все значения process-start измерены напрямую; медианы столбцов не складываются.
Они относятся к запуску подготовленного host с существующими source/PCH/scripts,
не к запуску всего Python-организатора/первой установке toolchain.

### 4.4. Все однофакторные результаты

Screening `series-6yi76ncc`, n=3 cold/режим. В таблице compile/link/warm/bytes —
plain; V1 prepared path показывает plain **и** trace. Полный `summary` не измерен.
Полный min/max каждого компонента,
fetch/load/PCH и native user/system сохранён в `tables.txt`/offline JSON.
Разброс существенно шире малых отличий; эти отличия не объявляются ускорением.

| Фактор | Compile Σ | Link Σ | **V1 prepared plain [min; max]** | **V1 prepared trace [min; max]** | Warm exec | `.so` Σ bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 276,369 | 272,013 | **849,583 [840,379; 983,633]** | **850,954 [850,815; 941,708]** | 7,832 | 61696 |
| без PCH | 1028,662 | 322,118 | **1733,985 [1483,799; 1739,539]** | **1709,621 [1487,322; 1771,148]** | 8,707 | 61696 |
| PCH без instantiate | 426,341 | 306,751 | **1111,450 [977,803; 1124,904]** | **1099,627 [972,486; 1168,375]** | 8,426 | 61696 |
| PCH q33 preamble | 304,458 | 310,278 | **972,862 [842,956; 984,971]** | **998,800 [841,262; 1021,113]** | 7,994 | 61696 |
| no exceptions | 302,155 | 310,564 | **968,939 [839,350; 969,455]** | **956,306 [842,479; 990,809]** | 9,011 | 61696 |
| no RTTI | 292,800 | 300,454 | **936,468 [846,567; 985,689]** | **960,372 [843,774; 1026,955]** | 8,043 | 61696 |
| no unwind tables | 295,192 | 301,133 | **949,021 [844,336; 1030,068]** | **960,136 [848,748; 986,308]** | 8,417 | 60992 |
| omit frame pointer | 323,372 | 319,968 | **996,386 [847,087; 1041,073]** | **1041,333 [838,776; 1048,239]** | 8,169 | 61968 |
| SSCP preoptimize | 336,063 | 301,976 | **978,269 [882,347; 1042,146]** | **954,152 [873,660; 1052,855]** | 8,392 | 64384 |
| linker O1 | 277,128 | 280,435 | **866,138 [850,311; 1028,580]** | **855,728 [850,615; 1014,204]** | 8,140 | 61568 |
| linker threads=1 | 279,524 | 275,581 | **863,896 [841,012; 931,539]** | **892,330 [862,814; 926,556]** | 8,102 | 61696 |
| linker threads=4 | 299,130 | 303,853 | **948,326 [852,033; 983,794]** | **943,903 [863,625; 1010,871]** | 8,335 | 61696 |
| linker gc-sections | 275,911 | 275,537 | **849,790 [843,613; 1000,500]** | **857,207 [851,755; 1002,683]** | 7,979 | 60848 |
| BFD вместо LLD | 277,966 | 351,632 | **937,179 [920,341; 1112,251]** | **932,646 [923,459; 1171,465]** | 8,062 | 87600 |
| replay только Clang link | 273,755 | 95,702 | **667,275 [665,294; 831,179]** | **678,975 [671,133; 766,556]** | 7,913 | 61696 |
| replay только LLD link | 272,910 | 59,280 | **636,400 [628,729; 761,061]** | **647,548 [643,757; 751,418]** | 8,183 | 61696 |
| replay только Clang compile | 96,604 | 279,090 | **686,213 [666,247; 795,706]** | **678,018 [676,086; 817,142]** | 8,080 | 61696 |
| parallel4, acpp/acpp | 264,635 | 270,720 | **434,070 [429,955; 509,818]** | **440,000 [437,932; 514,456]** | 8,430 | 61696 |

Дополнительный ELF/linker screening `series-9q99u37u`, n=3. Это отдельный временной
блок, не объединённый со старым baseline:

| Фактор | Compile Σ | Link Σ | **V1 prepared plain [min; max]** | **V1 prepared trace [min; max]** | Warm exec | bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 275,568 | 274,473 | **850,495 [847,283; 1069,266]** | **855,011 [844,215; 1093,037]** | 8,236 | 61696 |
| gold | 299,189 | 310,286 | **947,519 [862,992; 1086,920]** | **955,989 [867,850; 1080,919]** | 8,199 | 68528 |
| ICF safe | 271,739 | 272,682 | **849,355 [841,140; 1043,583]** | **856,377 [843,090; 1026,051]** | 8,322 | 61696 |
| strip-all | 274,655 | 273,064 | **854,858 [838,612; 1046,656]** | **856,241 [844,472; 1043,697]** | 7,880 | 52200 |
| SysV hash | 271,745 | 274,875 | **851,481 [845,339; 1025,749]** | **851,611 [851,521; 1018,264]** | 8,251 | 62016 |

Вывод: PCH и предварительная инстанциация действительно устраняют frontend-работу.
Расширение preamble и остальные малые flags не показали сопоставимого выигрыша.
BFD/gold не улучшили этот cold path. Strip уменьшил ELF, но не latency; symbols для
диагностики теряются, поэтому его не выбираем. Файлы и неудачные гипотезы сохранены.

### 4.5. Что видно в traces; влияние диагностики

Суммы одноимённых Clang trace-интервалов четырёх модулей, медианы n=3 screening:

| Вариант | ExecuteCompiler | Frontend | Backend | Optimizer (внутри backend) | Pending instantiations (внутри frontend) |
| --- | ---: | ---: | ---: | ---: | ---: |
| без PCH | 789,995 | 748,648 | 27,386 | 11,643 | 106,685 |
| PCH без instantiate | 192,459 | 153,583 | 25,660 | 11,530 | 116,593 |
| baseline PCH + instantiate | 54,316 | 17,654 | 24,897 | 12,001 | 2,339 |
| PCH q33 preamble | 59,605 | 17,131 | 27,862 | 12,943 | 2,741 |
| preoptimize | 88,612 | 18,830 | 56,566 | 41,879 | 2,367 |

В подтверждающей серии baseline ExecuteCompiler Σ=54,596 против native compile
wall Σ=276,271; LLD ExecuteLinker Σ=37,190 против native acpp link Σ=274,197.
Replay+parallel: ExecuteCompiler Σ=50,266, LLD Σ=25,362; **не** следует складывать
эти суммы с elapsed=39,264, особенно при параллельной работе.

Парные серии с/без traces, n=7, показаны целиком; это оценка наблюдаемой разницы
диагностических режимов, не поправка к историческим данным:

| Вариант | **V1 prepared plain [min; max]** | **V1 prepared traced [min; max]** | Δ median path | Δ compile Σ | Δ link Σ |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | **847,645 [837,816; 909,745]** | **854,496 [843,863; 938,349]** | +6,851 | +0,826 | +1,477 |
| fast-link | **627,376 [621,276; 685,295]** | **635,059 [626,419; 640,056]** | +7,683 | +2,839 | −1,345 |
| parallel4 | **436,327 [428,878; 440,686]** | **432,439 [429,244; 440,536]** | −3,888 | +2,765 | +1,911 |
| parallel-fast-link | **381,150 [375,378; 385,568]** | **381,056 [376,166; 428,185]** | −0,094 | +2,345 | +0,942 |
| replay-parallel | **337,226 [332,240; 388,390]** | **335,558 [332,588; 377,624]** | −1,668 | +2,076 | +0,797 |
| + preoptimize | **352,174 [349,565; 359,658]** | **352,268 [348,674; 360,509]** | +0,094 | +2,615 | +0,610 |

Отрицательные Δ не означают, что traces ускоряют запрос: execution/scheduling
шум может перекрывать диагностический overhead. Ничего не вычитается из старых
измерений. Для screening/runtime/поздних серий такой же парный полный экспорт
сохранён в raw reports и `tables.txt`, не смешан с этой таблицей.

### 4.6. Runtime и поздняя повторяемость

`series-e_2b8uoh`, n=3, warm=24/режим:

| Runtime setting | **V1 prepared plain [min; max]** | **V1 prepared traced [min; max]** | First exec plain | Warm exec plain |
| --- | ---: | ---: | ---: | ---: |
| adaptivity=1, tracking=1 | **846,660 [846,350; 864,336]** | **852,512 [845,893; 861,918]** | 295,388 | 7,933 |
| adaptivity=0 | **833,578 [832,041; 844,544]** | **838,176 [835,329; 843,114]** | 289,046 | 7,916 |
| adaptivity=2 | **850,859 [841,916; 853,003]** | **848,640 [841,711; 850,106]** | 298,118 | 7,870 |
| tracking=0 | **853,862 [839,625; 855,529]** | **858,758 [842,173; 864,491]** | 295,916 | 8,057 |

Малый выигрыш adaptivity=0 проверен ещё раз **вместе с replay** в позднем блоке,
n=7, warm=56. Сравнивать его нужно с современным ему replay, не с 337 мс:

| Вариант | Build plain | **V1 prepared plain [min; max]** | **V1 prepared traced [min; max]** | First exec plain | Warm exec plain |
| --- | ---: | ---: | ---: | ---: | ---: |
| replay, adaptivity=1 | 43,828 | **423,693 [399,874; 439,451]** | **422,490 [417,982; 432,047]** | 378,276 | 8,696 |
| replay, adaptivity=0 | 42,036 | **411,427 [404,273; 429,639]** | **407,812 [378,367; 421,061]** | 369,842 | 8,757 |

Compile Σ plain=103,919/102,682; link Σ=52,962/53,358. Размеры остались 61 696 байт.
В этом блоке первое выполнение **существенно медленнее ранних серий**, хотя build
остаётся коротким. Точная причина не установлена; возможное power/thermal/desktop
влияние — гипотеза. Доказательства одинакового hardware-state между блоками нет.
Первое выполнение включает JIT, но его отличие от warm не называется чистым JIT.
Отключение adaptivity не предлагается как общая SQL production-настройка: тест
фиксированных параметров не покрывает цену отключения специализации других запросов.

Пробная n=1 серия также сохранена полностью: baseline V1 prepared path plain=1448,723,
trace=981,936; preoptimize=1029,990/1055,476; Clang-link=826,524/841,856;
LLD-link=790,633/773,691; replay-parallel=417,421/418,568.
Разные границы прогрева общих OS/runtime компонентов и n=1 не позволяют считать
эту таблицу доказательством overhead traces или ускорения flags. Её native timers,
первое/warm, размеры и PCH доступны в полном экспорте, не скрыты и не усреднены с n=7.

### 4.7. SQL-контроль порядка запросов и linker O0

Три повторные сессии на конфигурацию, тот же Release `db_cli`, O0/PCH, compiler
traces включены, strace выключен. Здесь native колонка — **compile + link**, а
`summary` — реальный SQL `total_query_ms` до готового CPU QueryResult.

| SQL Q3.3 / контекст | Native acpp compile+link | **SQL summary [min; max]** | Fetch [min; max] |
| --- | ---: | ---: | ---: |
| первый, linker default | 2221,728 | **2773,762 [2771,485; 2789,129]** | 71,891 [71,630; 74,513] |
| первый, `-Wl,-O0` | 2200,492 | **2759,195 [2754,424; 2854,623]** | 74,847 [73,603; 75,212] |
| после Q2.1, `-Wl,-O0` | 2231,336 | **2720,531 [2696,978; 2727,177]** | 0,941 [0,903; 1,152] |

Различие `-Wl,-O0` не объясняет историческое большое изменение fetch. Контроль
порядка воспроизводит его при неизменном linker flag. В отдельном DEBUG=3 запуске
общий fetch kernel из `DatabaseInstance::executeQuery` получает один cache MISS
и пять последующих cache hits; код соответствует `CompactSparseAggregateResult`
в `src/db/database_instance.cpp`. См. `sql-baseline-83src1b9/stderr.txt`, строки
340–358 (первый miss) и 693 (hit при первом Q3.3). Это подтверждает reuse общего
служебного ядра, но не измеряет, сколько из fetch точно потрачено только на JIT:
инициализация/scratch/scheduling также не выделены отдельно.

Нынешние native времена около 2,2 с **не заменяют** исторические около 2,77 с.
Flags/бинарник/исходники сохранены, но физическое состояние машины между датами
не выровнено. Нельзя приписывать это ускорение исследуемым PCUDA flags: SYCL-путь
не изменён. Полной повторной SYCL optimization matrix не проводилось.

### 4.8. Представительные примитивы

Дополнительная correctness-серия baseline/replay, по три набора, с Clang/LLD
traces. Ниже native command medians, мс; `summary` здесь **не измерен**, модули
собирались перед fixture-host, CPU oracle между fixtures исключён из execute.
Это проверка переносимости способа сборки, не новая сквозная latency-матрица.

| Модуль | baseline compile / link | replay Clang / LLD | `.so` bytes | summary |
| --- | ---: | ---: | ---: | --- |
| seed | 84,457 / 103,993 | 27,047 / 16,597 | 13528 | не измерен |
| block_scan | 130,786 / 126,901 | 45,240 / 22,243 | 14264 | не измерен |
| signed block_reduce | 136,537 / 130,854 | 48,204 / 24,804 | 18184 | не измерен |
| atomic_probe | 134,368 / 130,290 | 51,181 / 24,790 | 16688 | не измерен |

Все семь модулей, их native min/max/user/system и trace-фазы — `validation`
в offline JSON; нельзя делать вывод о величине выигрыша по разным временным
блокам только из четырёх приведённых примеров.

### 4.9. Итоговый V2: подготовка первого запроса внутри summary

При финальном аудите обнаружена слишком узкая V1-граница: query-specific
domain/bounds preparation и аллокации выполнялись после загрузки данных, но до
таймера. Это нельзя выдавать за полную cold-задержку первого запроса.
Изменена **только host-инструментация**, GPU-ядра/алгоритм/данные/flags не менялись.
V1 source и его SHA-256 сохранены в `final-check-5k5qzbg1/sources/`; V1 raw reports
не изменены. Снята новая независимая rotated серия V2, а не добавлена постфактум
сумма preparation к старым значениям. `method_version=2`,
`preparation_in_summary=true` явно записаны в JSON.

`series-0je32lqk`: n=7 cold/режим и 56 warm. Здесь **summary — требуемая полная
граница прототипа** от data-ready до CPU-ready, по-прежнему без SQL parse/plan/codegen.

| Профиль | Compile Σ wall | Link Σ wall | Build elapsed [min; max] | **summary plain [min; max]** | **summary traced [min; max]** |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline acpp/acpp | 274,545 | 276,327 | 551,874 [544,708; 632,129] | **890,400 [873,552; 1044,218]** | **890,413 [880,326; 1002,324]** |
| parallel4 acpp/acpp | 266,179 | 267,376 | 136,522 [136,062; 159,053] | **469,289 [464,130; 530,568]** | **467,326 [460,606; 546,252]** |
| parallel replay Clang/LLD | 87,616 | 49,598 | 37,972 [37,006; 40,006] | **371,793 [364,140; 422,587]** | **391,296 [366,831; 424,058]** |

Остальные колонки ниже — plain, каждый показатель измерен отдельно:

| Профиль | Query preparation (внутри summary) | Load | First execute | Fetch | Warm execute [min; max] | Warm total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 33,237 | 0,523 | 304,722 | 0,043 | 8,402 [6,365; 14,124] | 8,445 |
| parallel4 | 33,244 | 0,424 | 299,343 | 0,042 | 8,343 [6,240; 14,216] | 8,381 |
| replay | 33,292 | 0,328 | 301,199 | 0,042 | 8,541 [6,343; 11,592] | 8,583 |

| Профиль | Compile Σ wall / user / system | Link Σ wall / user / system | PCH wall / user / system (отдельно) |
| --- | ---: | ---: | ---: |
| baseline | 274,545 / 200,603 / 74,590 | 276,327 / 213,502 / 97,158 | 279,363 / 254,850 / 28,973 |
| parallel4 | 266,179 / 197,462 / 66,025 | 267,376 / 201,796 / 91,303 | 304,223 / 272,394 / 32,109 |
| replay | 87,616 / 46,389 / 39,734 | 49,598 / 24,283 / 24,859 | 272,901 / 240,484 / 28,969 |

| Профиль | Startup | Data loading | **summary** | Spawn → CPU-ready [min; max] |
| --- | ---: | ---: | ---: | ---: |
| baseline | 250,844 | 202,417 | **890,400** | 1335,327 [1290,919; 1525,019] |
| parallel4 | 250,678 | 174,500 | **469,289** | 889,802 [879,051; 1023,789] |
| replay | 240,035 | 184,557 | **371,793** | 792,835 [776,820; 940,743] |

Preparation **не прибавляется к summary повторно**. Медианы стадий не складываются
для получения полного старта. Query/allocation время теперь не спрятано в startup.
Медианный остаток `summary - preparation - build_elapsed - load - first_total`
около 0,0002 мс; это coverage check, не доказательство наносекундной точности.
Размер четырёх `.so` прежний — 61 696 байт, все 378 результатов V2 совпали с SF10.

Наблюдаемая разница diagnostic median summary (trace−plain): baseline +0,013,
parallel4 −1,962, replay +19,503 мс. У replay median first execution trace=315,246
против plain=301,199; нельзя объявлять всю разницу чистым overhead traces.
Δ compile Σ: +2,038 / +4,263 / +2,626; Δ link Σ: −1,493 / +1,022 / +1,033.
Полные traced min/max/user/system/PCH/first/warm — в новом offline-экспорте.

## 5. Корректность и ограничения измерителей

- **2874/2874** новых PCUDA Q3.3 результатов совпали с независимым ручным SF10
  `test/queries/result/scale10/33.txt`, с перестановкой строк, но с сохранением
  multiplicity. Проверка Python Counter выполняется после измеренной серии,
  внутри summary нет CPU oracle. Все значения агрегата integer, без double-потери.
- **450** модульных checks (432 fixtures + 18 pending-work unload) для baseline/
  replay; **990** collective fixtures. Empty/partial blocks, несколько уровней scan,
  signed/unsigned SUM/MIN/MAX, значения выше 2^53, poison guards, блоки 32/48/64/128/256,
  32/64-bit atomics. Проверки не основаны на assert; host собран Release/NDEBUG.
- У 42 primitive-модулей проверены уникальные HCF IDs, по одному kernel name,
  отсутствие full SYCL dependencies. Q3.3 sessions также проверяют четыре разных
  HCF IDs и dependency files. Explicit stream, RAII и waits перед unload/free
  сохранены; промежуточные таблицы не материализуются на host.
- Дополнительный Bash correctness-only Clang replay driver проверил **1971**
  mixed Int64/Float64/typed-NULL/independent-slot checks и **1305** checks лимита
  8 typed + fallback. В bounded — ровно 9 compile/link/load на процесс;
  значения/validity и возврат к готовому типу не добавляли сборки. Эти отчёты
  используют прежние field names, но их «compile» — wrapper→Clang, **не acpp wall**;
  их performance не используется для вывода об ускорении acpp.
- Новый полный `db_cli` прогон всех **13 SSB × 3**: **36/39** совпадений с файлами,
  три известных Q1.1 mismatch. Raw `success:false` оставлен. Q1.1 снова проверен
  независимым CPU oracle: SQL=4486422947100, файл/1993-only=4474878955766,
  разница 11543991334 — 1994-01-01. SQL и эталон не изменены. Всего с отдельными
  SQL-контролями 81 результат, 78 совпадений с файлами; незаметно переопределённого
  общего `success` нет. Warm всех новых SQL-контролей: codegen/compile=0,
  compiler events дают ожидаемое число cold сборок + отдельный PCH.
- Для Q3.3 host build script вызывается один раз, реально записаны 4 compile и
  4 link timers, четыре load/dlsym, затем восемь warm повторов без новых команд.
  Это не доказательство ещё не реализованной параметризации SQL-движка.
- Пройдено 11 существующих CPU-тестов этапа 0 и 6 новых CPU-тестов измерителя:
  native locale/ошибки, несуммирование вложенных traces, raw stats,
  готовность/отсутствие first-result marker, точное разделение SQL-вариантов,
  включение preparation в V2 summary (V1/исключённая preparation отвергаются).
  `test/test_optimizer.cpp` не менялся: optimizer/SQL codegen не изменялись.
- Перед V2 проверены синтаксис восьми новых Python-скриптов и 286 Bash-файлов,
  локальные ссылки, `git diff --check`; hashes всех measured host sources и
  модулей совпадают, исходники движка совпадают с историческим snapshot.
- После V2: восемь Python и 328 Bash syntax checks, 17 CPU unit tests,
  ссылки/whitespace и `git diff --check`; hashes всех модулей и соответствующих
  сохранённых V1/текущих V2 host sources совпали. Все 42 V2 execution records
  прошли отдельную проверку включения preparation и непересекающегося покрытия.
- Manifest-регрессия всей OLAP-функциональности, полный SQL NULL/3VL/nullable-data
  и перенос ручных примеров на PCUDA этим исследованием **не выполнялись**.
  Typed-NULL микротест не подменяет SQL 3VL. Это остаётся работой миграции.

При offline-проверке обнаружены **две ошибки агрегации, не вычислений**:

1. Первый `sql_controls.py` выбирал suffix `q33-O0`, смешав также `q21-q33-O0`:
   исходная агрегированная строка имеет n=6 и не используется. Raw session records
   верны. Исправлено точное имя варианта и добавлен CPU regression test.
2. В первом `validate.py` поле session.modules со списком ELF metadata было
   перезаписано одноимённым fixture-result. Бинарники, timers, symbols, traces и
   correctness results сохранены. Metadata восстановлена **отдельным offline
   экспортом**, будущее имя — module_artifacts.

Исходные отчёты не переписаны. `analysis-2ykgkys8/summary.json` явно фиксирует обе
коррекции и hashes входных JSON. Неисправные агрегаты не стали задним числом
«успешными измерениями», GPU/SQL прогоны ради смены агрегации не повторялись.

Документация этапа 0 сведена в [единый REPORT.md](../pcuda_stage0/REPORT.md),
архивы A–D. Проверены тексты после нормализации только headings/internal links и
каждая исходная строка таблиц: [итоговая проверка](../../build/pcuda-compile-perf/doc-consolidation-0pqokm5t/content-check-final.json).
Первая проверка обнаружила три ошибки ручного переноса строк collective-таблицы;
они исправлены **до удаления дублей**, исходная неуспешная проверка также сохранена
в том же каталоге (`content-check.json`). Это исправление документа, не новый
GPU-прогон и не изменение исторических сырых измерений.

## 6. Выводы, история предложений и согласованные решения

**Подтверждено в пределах стенда:**

1. Минимальный заранее подготовленный PCH + `-fpch-instantiate-templates` — основной
   frontend-выигрыш. Следующий bottleneck stage 1 — многократная подготовка команд/
   процессов, а не многосотмиллисекундный frontend одного PCUDA-модуля с PCH.
2. Четыре независимые compile/link цепочки можно параллелить, сохраняя `.so` на ядро
   и fused probe/aggregate. Это даёт большой cold-выигрыш даже без replay.
3. Replay сокращает host build ещё сильнее, не меняя размер/результат обычных модулей.
   Но это другой compile/link driver boundary и требование к обслуживанию toolchain,
   не переносимый «магический флаг acpp».
4. После сокращения host build первое GPU execution (с возможным device JIT)
   доминирует: наблюдались примерно 299–378 мс по разным подтверждающим сериям.
   Усиливать host flags бесконечно не устранит эту часть. Нельзя объявлять эти
   интервалы чистым device JIT без новой инструментации.
5. No-exceptions/RTTI/unwind/frame, linker O/threads/GC/ICF/hash/strip не дали
   сопоставимого подтверждённого cold-выигрыша. Preoptimize увеличивает работу
   optimizer и unsupported; не рекомендуется. Нейтральные результаты сохранены.

**Гипотезы/непроверенное:** эффект LLVM/toolchain upgrades, внутреннего JIT linker
и его O3 pipeline, нового механизма in-process compilation; частотные/thermal
причины позднего замедления. Ничего из этого не реализовано и не выдаётся за факт.

### Исходное предложение от 2026-10-30 — историческое, не действующие требования

Ниже сохранено предложение **до согласования**, не SLA и не утверждённые требования.
На 2026-10-30 предлагались gates для того же
SF10 PCUDA-прототипа с **полными V2-границами из раздела 4.9**, на этом AMD-устройстве,
ready PCH, OS caches не очищены,
пустой device-cache на процесс, не менее 7 cold, парный современный baseline:

| Профиль | Предлагаемый build elapsed median | Предлагаемый **summary median** | Warm |
| --- | ---: | ---: | --- |
| штатный acpp, до 4 независимых задач | ≤ 200 мс | ≤ 600 мс | total и execute ≤ 1,10 своего парного baseline |
| optional version-bound replay | ≤ 70 мс | ≤ 500 мс | то же |

Порог 400 мс для replay **не предлагается как устойчивый**: поздний n=7 его не
выполняет. Для одного из этих четырёх отсутствующих O0/PCH модулей Q3.3 возможен
отдельный gate native acpp compile median ≤ 100 мс; это не порог для всех
collectives или произвольного сложного SQL.
Ориентир пользователя «500 мс компиляции» не приравнивается ни к SQL summary,
ни к одному trace-полю. Из доступного troubleshooting `build.sh` и `q33.json`
видны разные границы (source→exe и ExecuteCompiler); точное происхождение
именно 500 мс не установлено и не додумано.

Для **реального SQL после интеграции** нужны отдельные согласованные per-query
критерии на всех 13 SSB/действующем manifest, совместимых изменённых параметрах
и современном SYCL baseline. Числа прототипа не подставляются в них автоматически.
На момент предложения пользователю также предстояло выбрать судьбу replay.
Принятое позже решение приведено ниже; эти процентные и replay-предложения
не стали действующими требованиями.

<a id="stage0-agreement"></a>

### Решение пользователя от 2026-10-31: этап 0 закрыт

Пользователь потребовал абсолютные числа вместо процентов и **явно утвердил**
следующие медианные цели для четырёхмодульного Q3.3 SF10/V2 на AMD Radeon 780M,
со штатным `acpp`, заранее подготовленным совместимым PCH и до четырёх независимых
compile/link цепочек. Это показатели ограниченного стенда, не SLA произвольного SQL.
Источник правил дальнейшей приёмки: [todo.md](../../todo.md#performance-targets).

| Показатель, мс | Утверждённая цель | Факт V2, acpp/parallel4 без traces | Простое объяснение |
| --- | ---: | ---: | --- |
| Cold summary | ≤600 | 469,289 | Первый запрос до CPU-результата после загрузки БД |
| Build elapsed | ≤200 | 136,522 | Вся параллельная компиляция и линковка четырёх модулей |
| Warm total | ≤10 | 8,381 | Повторный запрос целиком до CPU-результата, без новой компиляции |
| Warm execute | ≤9 | 8,343 | Стадия исполнения повтора, без fetch |

Все четыре согласованные цели выполнены в сохранённой серии
`series-0je32lqk`, раздел 4.9. Согласование не является новым экспериментом:
измерения сделаны 2026-10-30, решение принято 2026-10-31.

Методика: минимум **7 cold-процессов и 8 warm-повторов в каждом**; медианы,
не максимум каждого запуска. Полный V2 summary включает query preparation и
аллокации; startup/data/PCH preparation/oracle/печать отдельно. Данные, порядок,
flags и кеши сопоставимы; новый appdb на процесс, сохранение внутри warm,
OS/HIP/driver caches не очищались. Основная серия без traces, сопоставимые
диагностические trace-серии отдельно. Сырые samples/разброс и определения интервалов
сохраняются. Другие устройства, масштабы и SQL-формы не наследуют эти пороги автоматически.

**SQL→PCUDA ≤10 мс** согласован как отдельный **целевой бюджет для проверки при
интеграции**, не как уже измеренный результат. Он охватывает host frontend:
parse/разрешение ссылок и параметров, планы/оптимизацию и codegen. Compile/link,
load, GPU-аллокации, execution и fetch сюда не входят. Нужен отдельный прямой
интервал; текущий SQL prepare_ms содержит runtime-работу и не является чистой
трансляцией. Из полного SQL summary эти затраты не вычитаются.

Полные SQL cold/warm-цели согласовать отдельно на интегрированном пути до
окончательной приёмки этапа 8; по решению пользователя это **не блокирует закрытие
этапа 0**. Прежние процентные предложения, условные 500/70 мс для replay и 100 мс
для отдельной компиляции не утверждены как требования.

**Replay — только необязательный [этап 9](../../todo.md#stage9-replay), если останется
время после реализации и приёмки этапов 1–8.** В roadmap перенесены измеренные
136,522 → 37,972 мс build и 469,289 → 371,793 мс prototype summary, техника
effective-command replay и риски совместимости. До этапа 9 основной путь —
штатный acpp; replay не является условием завершения основной миграции.

Согласование закрыло последний пункт этапа 0 и ответвление PERF-COLD. Этап 1
разблокирован, но в рамках этой документальной фиксации не начат. Исторические
предложения, V1/V2 и все raw reports/traces сохранены без изменения результатов.

## 7. Воспроизведение и оставшаяся работа

Из корня проекта, выбрать свои пути compiler/GCC/data/device. Скрипты не требуют
пересборки/переустановки основного движка и не используют SQL cache прошлой сессии.

`test/pcuda_compile_perf/inventory.py`:

```bash
python3 test/pcuda_compile_perf/inventory.py \
  --acpp-source /home/aidar/AdaptiveCpp \
  --extra-flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
```

`test/pcuda_compile_perf/run.py`, основные подтверждающие варианты:

```bash
python3 test/pcuda_compile_perf/run.py --data data/10 --device-name 780M \
  --extra-flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  --variants baseline,fast-link,parallel4,parallel-fast-link,replay-parallel,preopt-replay-parallel \
  --series 7 --runs 9
```

Для точного воспроизведения остальных наборов взять `arguments.variants/series/runs`
из соответствующего raw report раздела 1; `--diagnostics both` — default.
Однофакторная матрица содержит варианты `baseline,no-pch,pch-no-instantiate,
pch-preamble,no-exceptions,no-rtti,no-unwind,omit-frame,preoptimize,link-O1,
link-threads1,link-threads4,link-gc,link-bfd,clang-link,lld-link,clang-compile,parallel4`.
Runtime: `baseline,adaptivity0,adaptivity2,no-tracking`.
Дополнительные ELF: `baseline,link-gold,link-icf-safe,link-strip,link-sysv-hash`.
Поздний replay: `replay-parallel,replay-adaptivity0`, 7 серий.

`test/pcuda_compile_perf/validate.py`, `sql_controls.py`:

```bash
python3 test/pcuda_compile_perf/validate.py --device-name 780M --series 3 \
  --extra-flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
python3 test/pcuda_compile_perf/sql_controls.py \
  --db-cli build/pcuda-stage0/engine-2trThb/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp --series 3
```

SQL controls пользуется старым `/dev/shm/jit_headers.h.gch` только с exclusive
ownership; не запускать одновременно другие старые db_cli. При существующем
чужом PCH runner откажется его перезаписывать.

`test/pcuda_compile_perf/make_replay_driver.py`, correctness-only type checks:

```bash
python3 test/pcuda_compile_perf/make_replay_driver.py \
  --extra-flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  --output build/pcuda-compile-perf/correctness-replay/acpp
python3 test/pcuda_stage0/types_bench.py \
  --acpp "$PWD/build/pcuda-compile-perf/correctness-replay/acpp" \
  --extra-flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  --device-name 780M --series 3 --output-root build/pcuda-compile-perf
```

Второй type run: та же команда с `--bounded-four`. Драйвер только для проверки,
не заменять им production acpp и не считать его wall временем acpp.

`test/pcuda_compile_perf/analyze.py`, CPU tests:

```bash
python3 test/pcuda_compile_perf/analyze.py
PYTHONDONTWRITEBYTECODE=1 python3 test/pcuda_stage0/test_tools.py
PERF_LAUNCHER="$PWD/build/pcuda-compile-perf/series-ydtefe2b/first_result" \
  PYTHONDONTWRITEBYTECODE=1 python3 test/pcuda_compile_perf/test_tools.py
```

Analysis всегда создаёт новый каталог и не переписывает raw. Все измеренные
команды, stdout/stderr, timings и traces остаются в артефактах; никаких новых
`RESULTS.md`/«продолжений» для этих прогонов не требуется.

**На 2026-10-31 этап 0 закрыт явным решением пользователя**, не одним только
завершением экспериментов. Дальнейшая работа: этапы миграции 1–8, проверка бюджета
SQL→PCUDA ≤10 мс и отдельное согласование абсолютных показателей полного SQL до
приёмки. Replay — лишь дополнительный этап 9 при наличии времени.
Отдельная S2/JIT-инструментация/LLVM-оптимизация остаётся непроверенной гипотезой,
а исходная граница раннего пользовательского ориентира 500 мс не установлена;
это не препятствия к согласованному закрытию этапа 0 и не новые обещания производительности.

**Последующее отдельное исследование, 2026-10-31:**
[JIT-TIER](../pcuda_jit_tiering/REPORT.md) проверяет внутренние S2 pipeline,
первый базовый вариант и фоновую подготовку следующего поколения. В нём есть
прямые JIT timers, PoC и ограничения общего cache mutex/runtime teardown.
Оно не изменяет приведённые здесь исторические измерения, согласованные цели
или состояние интеграции движка.
