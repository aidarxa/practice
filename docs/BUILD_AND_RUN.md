# Сборка и запуск Crystal

<a id="quick-start"></a>

## 1. Быстрый старт

Команды выполняются **из корня репозитория**. Нужны Linux, GPU-драйвер,
совместимый AdaptiveCpp с PCUDA (`acpp`, `clang++` и `ld.lld` в `PATH`),
CMake 3.20+, Make и Python 3.10+. [Подробнее о зависимостях](#requirements).

**Только для свежего клона без `deps/`:** установите библиотеки проекта.

```bash
bash install-deps.sh
```

**Настройте сборку — одна команда.** Пример для проверенного стенда:
Clang 17 из ROCm SDK и установленный GCC 14.3.1. Для другого SDK см.
[варианты конфигурации](#toolchain).

```bash
cmake -S . -B build/pcuda-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DACPP_TARGETS=generic \
  -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  -DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
```

**Соберите — одна команда:**

```bash
cmake --build build/pcuda-release -j4
```

Это полная сборка: дальнейшие команды SQL и SSB не требуют отдельной сборки
бинарников или запуска тестов.

**Создайте данные один раз.** Если `data/ssb/sf10/dataset.manifest` уже есть,
пропустите этот шаг. Для SF10 нужно около 5 ГиБ на постоянное хранение и больше
места во время генерации. Если памяти или диска мало, [возьмите SF1](#data).

```bash
bash bench/prepare_ssb.sh
```

**Запустите запрос:**

```bash
./build/pcuda-release/db_cli < test/queries/sql/q33.sql
```

Или откройте SQL-консоль:

```bash
./build/pcuda-release/db_cli
```

Готово. Не нужно задавать переменные окружения или создавать конфиг. После
загрузки базы Q3.3 выводит таблицу с результатом; на проверенном SF10 — 24 строки.

---

<a id="choose"></a>

## Дальше — только по необходимости

- [SQL-файлы, консоль и все 13 SSB](#run)
- [Другой масштаб или каталог данных](#data)
- [Необязательные настройки](#config)
- [Зависимости и нестандартный toolchain](#requirements)
- [Разработчикам: варианты сборки и тесты](#build)
- [Диагностика и измерения](#timing)
- [Установка](#install)
- [Типичные проблемы](#troubleshooting)
- [Что проверено](#verified)

<a id="run"></a>

## 2. SQL и SSB

Здесь и ниже используется полная сборка `build/pcuda-release` из быстрого старта.
Не переходите в build-каталог: относительные пути к данным считаются от текущего
рабочего каталога.

### Консоль и SQL-файлы

Внутри `db_cli` SQL заканчивается `;`, многострочный ввод поддерживается:

```sql
SELECT COUNT(*) FROM ddate;
```

Мета-команды вводятся отдельной строкой без `;`:

| Команда | Назначение |
| --- | --- |
| `help` | Справка |
| `\q` или `.exit` | Выход; EOF тоже завершает сессию |
| `\timing on` / `off` | Подробные времена и module counters |
| `\limit all` | Показать все строки результата |
| `\limit N` | Показать последние N строк; это не SQL `LIMIT` |
| `\dump on` / `off` | Генерировать код вместо исполнения / исполнять SQL |
| `\o filename` | Сохранить generated code при `\dump on`, не обычный результат |

SQL-файл передаётся через stdin, как в быстром старте. Для сохранения результата:

```bash
./build/pcuda-release/db_cli < test/queries/sql/q33.sql > build/q33.stdout 2> build/q33.stderr
```

У CLI нет ключей `--help` или `--data`: используйте `help` в консоли и
[настройку каталога данных](#data). CLI продолжает сессию после ошибки SQL,
поэтому exit code 0 без результата ещё не доказывает успешное выполнение.

<a id="ssb"></a>
<a id="ssb-sql"></a>

### Проверить все 13 SSB

Необязательная проверка SQL и ручных реализаций с независимым CPU-oracle:

```bash
python3 test/ssb_native_data/validate.py --build build/pcuda-release --timing-model pcuda-modules-v1
```

По умолчанию используются SF10 и два запуска каждого запроса. Команда сама
создаёт отдельный каталог `build/ssb-native-validation/native-*`. В конце ожидаются
`SQL 26/26; manual 13/13; success=True`: у каждого из 13 manual-клиентов
проверяются оба результата, то есть всего **26 SQL +26 manual результатов**.

Полная сборка из быстрого старта уже содержит всё необходимое. Если ранее
собирался только `db_cli`, сначала выполните полную команду сборки из раздела 1:
runner требует также все `ssb_qNN` и `bench/loader/ssb_reference` и сам их не собирает.

Дополнительные опции:

| Опция | Что меняет |
| --- | --- |
| `--sql-only` | Пропустить manual; `db_cli` **и** `ssb_reference` всё равно нужны |
| `--trials 1` | По одному исполнению без проверки повторного использования модулей |
| `--output-root build/sql-ssb` | Другой родительский каталог результатов |
| `--tables /path/to/tables` | Дополнительно сверить все 58 колонок с исходными `.tbl` |
| `--manifest` | Добавить current/current_heavy и targeted cases; нужен NumPy |
| `--traces` | Сохранить диагностические Clang/LLD traces |

SQL идут в одной сессии, manual — отдельным процессом на запрос. Повторный SQL
проверяется не только по результату, но и по отсутствию codegen/compile/link/load
events. Это функциональная проверка, **не серия независимых cold-измерений**.
GPU-клиенты не следует запускать одновременно друг с другом.

В напечатанном каталоге находятся `report.json`, `reference.json`, SQL transcript
в `sql/`, вывод manual в `qNN/`, команды, native process timings и JIT-артефакты.
Они сохраняются отдельно для каждого запуска.

**Q1.1:** исходный SQL включает `1994-01-01`, manual — только 1993 год.
Runner проверяет обе формы раздельно свежим oracle, без исключения для mismatch.
Старые числовые `test/queries/result/scale10` не являются Text-oracle; исторические
неуспешные сравнения в отчётах не переписываются.

<a id="ssb-manual"></a>

### Ручной SSB без SQL JIT

```bash
./build/pcuda-release/ssb_q33 data/ssb/sf10 2
```

Формат: `ssb_qNN [data-directory [trials]]`, default trials — 4, допустимо 1–1000.
Все 13 бинарников уже входят в полную сборку. Это независимые заранее собранные
PCUDA-алгоритмы, а не вызов SQL-генератора. Они требуют nonnullable input и
уникальные dimension keys. Результаты между `__SSB_RESULT_BEGIN__` и
`__SSB_RESULT_END__` — JSON-массивы с настоящими Text-строками.
`Query CPU-ready ms` не включает SQL parse/codegen/compile; внутренний device JIT
при первом запуске возможен и для заранее собранного кода.

<a id="data"></a>

## 3. Другой масштаб или каталог данных

Например, меньший набор SF1:

```bash
bash bench/prepare_ssb.sh --scale 1
CRYSTAL_DATA_DIR=data/ssb/sf1 ./build/pcuda-release/db_cli < test/queries/sql/q33.sql
```

Если SF1 уже подготовлен, первую команду пропустите. Вместо переменной можно
сохранить `data_dir = ./data/ssb/sf1` в `crystal.conf`.
Для SSB runner каталог передаётся обычным аргументом:

```bash
python3 test/ssb_native_data/validate.py --build build/pcuda-release --data data/ssb/sf1 --timing-model pcuda-modules-v1
```

Пересборка при смене SF не нужна. Размеры таблиц читаются из manifest выбранного
набора; они не зашиты в бинарник.

### Дополнительные способы подготовки

Каждый `--output` должен быть **новым каталогом**:

```bash
bash bench/prepare_ssb.sh --scale 1 --output data/ssb/new-sf1 --keep-tbl
```

`--keep-tbl` сохраняет исходные таблицы в напечатанном `build/ssb-tools/prepare-*/tables`
для проверки через `--tables`. По умолчанию сгенерированные `.tbl` удаляются после
подготовки. Для уже имеющихся исходных таблиц:

```bash
bash bench/prepare_ssb.sh --scale 1 --from-tbl /path/to/tables --output data/ssb/imported-sf1
```

Входной каталог не удаляется. Требуются `lineorder.tbl`, `supplier.tbl`,
`customer.tbl`, `part.tbl`, **`date.tbl`**: pipe-разделитель, завершающий `|`,
без заголовков. `--jobs N` меняет параллельность host-сборки; default — 4.

Подготовка не требует GPU, AdaptiveCpp, SQL parser или сборки движка: скрипт сам
собирает host-only C++17 importer и закреплённый `ssb-dbgen`. При первом получении
submodule нужна сеть. Изменённый submodule не перезаписывается. Значения/PK
проверяются до атомарной публикации всего набора; `--force` нет. При ошибке не
публикуется частичный dataset. Это не обещание crash durability при потере питания.

Для обычного SSB используйте **целый SF ≥1**. Дробный SF ≥1 отклоняется, а SF <1
масштабирует также календарь DDATE и годится только для smoke fixtures.
Очень большой SF ограничен Int32-ключами dbgen, диском и памятью устройства.

### Native-формат

Набор содержит 58 колонок и обязательный `dataset.manifest` версии
`CRYSTAL_SSB_NATIVE_1`: схема, SF, generator identity и фактические row counts.
Manifest и размеры проверяются до GPU allocations. Схема —
[`include/storage/ssb_schema.h`](../include/storage/ssb_schema.h):

- Int32 little-endian, четыре байта на строку; flags — ASCII `'0'`/`'1'`.
- FixedText декодируется до bounded NUL или на всю ширину, с проверкой UTF-8;
  пробелы значимы. Все исходные категории сохраняются как Text, без словаря.
- DDATE имеет 17 колонок, полный `Wednesday` занимает 9 байт; brand — до 10 байт.
- Статистика `.crystal-numeric-stats-v1` привязана к выбранному набору;
  read-only dataset работает без записи кеша.

Пример SF10: LINEORDER — 59 986 217 строк, SUPPLIER — 20 000, CUSTOMER — 300 000,
PART — 800 000, DDATE — 2 557. Это пример, не формула расчёта runtime-размеров.
Старый `data/10/`, encoded loaders `load_modified.c`/`convert.py` и numeric
references сохранены только как история, не native-вход. Не конвертируйте данные
поверх старых файлов. [Отчёт интеграции](../test/ssb_native_data/REPORT.md).

<a id="config"></a>

## 4. Необязательные настройки

Для стандартного запуска конфиг создавать не нужно. Если нужны изменения,
пример ключей находится в [`crystal.conf.example`](../crystal.conf.example);
собственные настройки можно записать в `crystal.conf` в корне проекта.

Читается первый существующий файл:

1. путь из `CRYSTAL_CONFIG`, если он задан;
2. `./crystal.conf`;
3. `./crystal-sycl.conf`;
4. `$HOME/.config/crystal-sycl/config.conf`.

Явные env overrides имеют приоритет. Они нужны только для соответствующего
изменения, не являются обязательной подготовкой:

| Что изменить | Config | Env override |
| --- | --- | --- |
| Dataset | `data_dir` | `CRYSTAL_DATA_DIR` |
| Выбрать GPU по подстроке имени | `device_name` | `CRYSTAL_DEVICE_NAME` |
| Выбрать другой конфиг | — | `CRYSTAL_CONFIG` |
| Выбрать нестандартный PCH manifest | `pcuda_manifest` | `CRYSTAL_PCUDA_MANIFEST` |
| Включить исследовательский overlap | `jit_overlap_experiment` | `CRYSTAL_JIT_OVERLAP_EXPERIMENT` |

Пустое имя GPU выбирает первое GPU; неоднозначное или отсутствующее совпадение —
ошибка, не CPU fallback. Остальные полезные config-ключи:

| Ключ | Default / смысл |
| --- | --- |
| `jit_jobs` | 4, допустимо 1–4 compile/link-цепочки |
| `jit_keep_artifacts` | off; сохранять commands, sources, `.so` и native timings |
| `jit_artifact_dir` | Пусто — системный temporary directory |
| `jit_traces` | off; диагностические Clang/LLD traces |
| `dump_generated_code` | off; on заменяет исполнение генерацией кода |
| `extended_timing` | off; аналог `\timing on` |
| `output_limit` | 1000; `all`/`off` отключают обрезку вывода |
| `reuse_scratch_buffers` | true; workspace текущего совместимого плана |
| `memory_guard_enabled` | on; не выключать для обхода OOM |
| `memory_guard_reserve_fraction` / `memory_guard_reserve_bytes` | 0.10 / 512MiB |
| `jit_overlap_experiment` | off; обычный путь — `pcuda-modules-v1` |

Boolean: `on/off`, `true/false`, `yes/no`, `1/0`. Неизвестный ключ — ошибка.
Compiler flags задаются в CMake, не в runtime config. PCH и manifest обычной
build-сборки выбираются автоматически; сохраняемые SQL-модули не становятся
постоянным кешем для следующих процессов.

<a id="requirements"></a>

## 5. Зависимости и нестандартный toolchain

### Зависимости

- C++17 и совместимый с AdaptiveCpp Clang/LLVM; произвольный Clang из системы
  не обязательно совместим с compiler plugins установленного SDK.
- CMake 3.20+ и Make либо Ninja. Проект допускает CMake 3.16+, но GPU PCH
  при CMake <3.20 требует генератор Ninja (`-G Ninja`).
- Python 3.10+, Bash, Git. Для `install-deps.sh` нужны также `gcc`, `make`, `ar`
  и сеть; для расширенных Text/manifest oracles — NumPy.
- AdaptiveCpp с `--acpp-targets=generic --acpp-pcuda`, LLD и GPU runtime/драйвер.
  [Документация PCUDA](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/pcuda.md).

`install-deps.sh` **не устанавливает GPU SDK**. В свежем клоне он создаёт
`deps/include/SQLParser.h`, `deps/include/linenoise.h`,
`deps/lib/libsqlparser.so`, `deps/lib/liblinenoise.a`. Уже подготовленный `deps/`
не нужно пересобирать. Скрипт клонирует и удаляет рабочие `sql-parser/` и
`linenoise/`: не запускайте его поверх каталогов с собственной работой.
Revisions этих двух библиотек скриптом не закреплены.

Окружение runtime берётся из установки SDK. Если библиотеки не находятся,
проверьте `ldd build/pcuda-release/db_cli` и инструкции поставки; не смешивайте
случайные HIP/HSA/AdaptiveCpp версии через `LD_LIBRARY_PATH`.

<a id="toolchain"></a>

### Другой SDK или compiler

Меняйте только нужные параметры в команде `cmake -S … -B …` из быстрого старта:

| Если… | Что указать |
| --- | --- |
| CMake не нашёл пакет AdaptiveCpp | `-DAdaptiveCpp_DIR=/path/to/AdaptiveCpp/lib/cmake/AdaptiveCpp` |
| `clang++` по умолчанию не тот | `-DCMAKE_CXX_COMPILER=/path/to/compatible/clang++` |
| Нужный `acpp` не найден в `PATH` | `-DCRYSTAL_PCUDA_COMPILER=/path/to/AdaptiveCpp/bin/acpp` из той же установки, что и пакет CMake |
| Clang выбирает совместимые C++ headers сам | Уберите обе опции с `--gcc-install-dir` |
| Clang выбирает несовместимые C++ headers | Укажите каталог установленного совместимого GCC в `CMAKE_CXX_FLAGS` и `CRYSTAL_PCUDA_FLAGS` |

На проверенном стенде Clang 17 автоматически выбирал GCC 16 и сборка падала.
В показанной команде GCC 14.3.1 задаётся для обычной сборки и для PCH/JIT.
При смене compiler/SDK или генератора используйте новый build-каталог и тот же
путь передавайте для сборки и в `--build` валидатора. Старый PCH не переносите.

### CMake-параметры

| Параметр | Назначение |
| --- | --- |
| `CRYSTAL_BUILD_GPU` | GPU runtime/CLI/manual; default ON |
| `CMAKE_BUILD_TYPE` | Release/Debug host-кода; динамические SQL-модули остаются `-O0 -g0` |
| `ACPP_TARGETS=generic` | Generic-путь заранее собираемых AdaptiveCpp targets, не C++ macro |
| `CRYSTAL_PCUDA_COMPILER` | `acpp` для PCH и SQL-модулей; default — поиск в PATH |
| `CRYSTAL_PCUDA_FLAGS` | Дополнительные PCH/module flags, CMake list через `;` |
| `CRYSTAL_PCUDA_LINK_FLAGS` | Default `-fuse-ld=lld;-Wl,-O0`, тоже CMake list |
| `CRYSTAL_INCLUDE_DIR` / `CRYSTAL_DEPS_INCLUDE_DIR` | По умолчанию `include` / `deps/include` проекта |
| `CRYSTAL_PCUDA_JIT_DIR` | По умолчанию `<build>/pcuda-jit` |
| `CMAKE_EXPORT_COMPILE_COMMANDS=ON` | Compile database для инструментов разработчика |

`-j4` управляет сборкой приложения, `jit_jobs=4` — runtime-подготовкой SQL-модулей.
PCH/services готовятся при build/install, а не безусловно при запуске CLI.

<a id="build"></a>

## 6. Для разработчиков: сборки и проверки

Эти действия **не нужны для обычного запуска**. Release из раздела 1 уже готов.

### Host-тесты существующей GPU-сборки

```bash
ctest --test-dir build/pcuda-release --output-on-failure
```

Assertions активны и в Release. GPU-клиенты/SSB не запускаются этим CTest набором.

### Только host-код, без AdaptiveCpp и GPU

Нужны C++17 compiler и подготовленный `deps/`; данные не нужны:

```bash
cmake -S . -B build/pcuda-cpu -DCRYSTAL_BUILD_GPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/pcuda-cpu -j4
ctest --test-dir build/pcuda-cpu --output-on-failure
```

Это тестовая сборка, **не CPU-версия SQL**: `db_cli` и `ssb_qNN` в ней нет.

### Debug

```bash
cmake -S . -B build/pcuda-debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DACPP_TARGETS=generic \
  -DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 \
  -DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake --build build/pcuda-debug -j4
```

Запуск — `./build/pcuda-debug/db_cli`. Debug относится к host-коду, не меняет
динамические SQL-модули на `-O3` или debug symbols.
Для сборки только CLI есть `--target db_cli`, но этого недостаточно для полного
SSB validator: ему требуется также CPU-oracle, а в default режиме — manual targets.

### Отдельно importer, без SQL/GPU-зависимостей

```bash
cmake -S bench/loader -B build/ssb-loader -DCMAKE_BUILD_TYPE=Release
cmake --build build/ssb-loader -j4
ctest --test-dir build/ssb-loader --output-on-failure
```

<a id="regression"></a>

### Полная GPU-регрессия

Нужны полная Release-сборка, данные и NumPy. Не запускайте одновременно другие
GPU-проверки. Текущие runners используют `/tmp/opencode`; он должен быть доступен
вашему пользователю:

```bash
mkdir -p /tmp/opencode
python3 test/pcuda_stage8/validate.py --build build/pcuda-release --data data/ssb/sf10 \
  --output-root build/full-regression --timing-model pcuda-modules-v1
```

Проверяются GPU/resources/ABI/NULL/MHT/OR, Text, native SQL/manual SSB и
current/current_heavy manifest. Это не обещание независимого oracle каждой ячейки
произвольного manifest-запроса. Категории `future` не становятся реализованными.
Исторический `test/pcuda_stage7/validate.py storage` проверял numeric-only этап с
Text rejection; это не текущая полная Text-регрессия.

### Python-клиенты и quality tools

```bash
python3 -B -m unittest discover -s test/ssb_native_data -p test_clients.py -v
python3 -B -m unittest discover -s tools/ub/tests -v
```

Некоторые client integration cases по умолчанию пропускаются; условия включения
указаны в самих тестах. Coverage/CRAP, cross-file DRY и mutations выполняются
существующими [`tools/ub`](../tools/ub/README.md), с новым каталогом артефактов
для каждого запуска. Native SSB recipe —
[`host-recipe.json`](../test/ssb_native_data/host-recipe.json).
Host coverage не является device coverage и не заменяет GPU/oracle проверки.

<a id="timing"></a>

## 7. Диагностика и измерения

### Повторный запрос в одном процессе

```bash
{
  printf '%s\n' '\dump off' '\timing on' '\limit all'
  cat test/queries/sql/q33.sql
  printf '\n'
  cat test/queries/sql/q33.sql
  printf '\n%s\n' '\q'
} | ./build/pcuda-release/db_cli > build/q33-twice.stdout 2> build/q33-twice.stderr
```

У второго запроса ожидаются нулевые `plan_codegen`, `module_codegen`, `compiler`,
`linker`, `loads`, `symbols`; `hits` может быть ненулевым. Plan/module/entry cache
существует только внутри процесса. AdaptiveCpp device-cache — другой слой;
его каталог можно отдельно выбрать `ACPP_APPDB_DIR`, но для обычного запуска
это не требуется. Новый каталог не очищает OS/HIP/driver caches.

### Как читать времена

- `Total query time` первого запроса — прямой **summary** от SQL в уже загруженной
  БД до CPU-результата. Startup, загрузка данных и печать сюда не входят.
- `Generated execute time` включает runtime и необходимые ожидания, возможно
  device JIT; это не чистое время GPU kernel.
- `Module build elapsed` — параллельный build interval; native compile/link sums
  нельзя складывать с ним как независимые затраты.
- `SQL to PCUDA time` заканчивается до compile/GPU allocations, перекрывается
  с prepare/codegen и не добавляется к component sum.

Подробные границы — [`timing.md`](../timing.md). Для диагностики включайте
`jit_keep_artifacts=on`/`jit_traces=on` в отдельном конфиге или `--traces` у native
validator. Clang/LLD traces — не ускоряющие флаги и не plain baseline.

### CSV/логи, без независимого payload oracle

```bash
python3 olap_bench/tools/run_olap_bench.py --db-cli build/pcuda-release/db_cli \
  --manifest olap_bench/manifest.json --class 07_ssb_standard \
  --mode single-session --warmups 0 --runs 2 --limit all --no-dump-code \
  --timing-model pcuda-modules-v1 --out build/ssb-csv
```

Создаются `summary.csv`, `aggregate_summary.csv`, `logs/`, `outputs/`.
Это проверка ошибок исполнения, **не** замена SSB-oracle. `single-session`
сохраняет процесс и кеши; `isolated` создаёт новые процессы и не измеряет warm
того же процесса. Смешанное mean/first агрегирование не является cold-median серией.

### Overlap — только явный эксперимент

Default — off. Исследовательский путь можно выбрать отдельно:

```bash
CRYSTAL_JIT_OVERLAP_EXPERIMENT=on ./build/pcuda-release/db_cli < test/queries/sql/q42.sql
```

Модель — `pcuda-modules-overlap-exp-v1`; build/load/execution перекрываются,
component sums/residuals — null. Для native validator укажите соответствующий
`--timing-model`: он сам задаёт env override. Общему CSV runner нужны **оба**.
Это не replay, parallel S2 или jobs>4. [Результаты и границы 8-ANALYSIS](../test/pcuda_stage8_analysis/REPORT.md).

### Performance-приёмка

Numeric-SF10 measure scripts/results этапа 8 — история, не baseline новой Text-схемы.
Функциональные проверки не утверждают Text speedup, полные SQL SLA или закрытие
этапа 8. Нагрузка и абсолютные цели native SQL согласуются отдельно.

Для сопоставимых серий нужны непрерывный summary, native process timing,
раздельные cold/warm и состояния всех кешей; минимум семь независимых cold-процессов
с восемью warm в каждом, plain/traced серии с ротацией и offline oracle после
всей серии. Не запускайте рядом сборки, GPU-клиенты или online-анализ. См.
[отчёт этапа 8](../test/pcuda_stage8/REPORT.md) и обязательную методику в
[`AGENTS.md`](../AGENTS.md). Числа прототипов не переносятся на полный SQL.

<a id="install"></a>

## 8. Установка

Укажите **конечный абсолютный prefix** вместо примера:

```bash
cmake --install build/pcuda-release --prefix /absolute/path/to/crystal
/absolute/path/to/crystal/bin/db_cli < test/queries/sql/q33.sql
```

Устанавливаются CLI, SQL parser, headers и заново подготовленный installed
PCH/manifest. Manual/test binaries не устанавливаются. Compiler, SDK и GPU runtime
остаются необходимыми; это не standalone-бинарник.

Installed manifest выбирается автоматически. Приоритет: `CRYSTAL_PCUDA_MANIFEST`
→ config `pcuda_manifest` → `../lib/crystal/pcuda-jit/toolchain.manifest` от бинарника
→ встроенный build-tree manifest. Старый явный override перекрывает автопоиск.
Clang PCH хранит абсолютные пути: перенос prefix/`DESTDIR` этим recipe не поддержан;
готовьте PCH в конечном расположении. Данные по относительному пути по-прежнему
требуют запуска из корня проекта либо явного `data_dir`.

<a id="troubleshooting"></a>

## 9. Типичные проблемы

| Симптом | Решение |
| --- | --- |
| CMake не нашёл AdaptiveCpp/acpp | Проверить установку SDK; для нестандартного пути — configure из раздела 5 |
| Clang plugin/version mismatch | Выбрать совместимый с AdaptiveCpp compiler, новый build-каталог |
| Ошибки `<string>`/`<type_traits>` | Проверить выбор совместимого [GCC](#toolchain) |
| Нет `SQLParser.h`, `linenoise.h`, `-lsqlparser` | Проверить `deps/`; installer нужен один раз, не заменяет GPU SDK |
| `libsqlparser.so`/`libacpp-rt.so` not found | `ldd build/pcuda-release/db_cli`, RPATH и runtime setup установленного SDK |
| PCH/manifest несовместим | Проверить explicit overrides и fingerprint mismatch; пересобрать `cmake --build build/pcuda-release --target pcuda_jit_pch`; при другом toolchain — новый build |
| Не найден `ssb_q11` / `ssb_reference` | Нужна **полная** сборка того же каталога, который передан в `--build`, не только target `db_cli` |
| Нет `dataset.manifest` / неверный размер колонок | Выбрать native dataset; старый `data/10` не подходит. `CUSTOMER7` — Text 10 bytes/row, не Int32 |
| `Output already exists` при подготовке | Набор уже есть: использовать его или выбрать **новый** `--output`, не удалять данные |
| Memory guard/OOM | Освободить память, не запускать конкурирующие клиенты или выбрать SF1; guard не выключать |
| Код вместо результата | `\dump off`; при старом пользовательском конфиге проверить `dump_generated_code` |
| Вывод обрезан | `\limit all` или увеличить `output_limit` |
| Native validator завершился с 1 | Смотреть `report.json`: ошибка исполнения/oracle/reuse; mismatch Q1.1 не допускается |
| Старый пример пытается писать `/sql-ssb` | Убрать старые shell-подстановки: текущий validator сам выбирает output, либо `--output-root build/sql-ssb`; sudo не нужен |
| Ошибка `/tmp/opencode` в runner | Создать доступный своему пользователю каталог; не менять права чужих файлов |

Не используйте старый `test_db_cli.sh` как основной quick runner: он меняет cwd
и не сравнивает payload. Сохраняйте `test/pcuda_*`: там исходники действующих
targets и oracles. Новые сборки/логи/PCH храните в игнорируемом `build/`.

<a id="verified"></a>

## 10. Проверка инструкции и границы

Команды быстрого старта проверены в новом build-каталоге на AMD Radeon 780M,
AdaptiveCpp 25.10/git и Clang 17. Q3.3 вернул 24 строки на готовом SF10, проверка SSB
подтвердила 26 SQL и 26 manual результатов. NVIDIA/Intel аппаратно не проверены.
[Команды, артефакты и история проверок](../test/ssb_native_data/REPORT.md#quick-start-verification).
Это функциональная проверка, не performance/SLA-приёмка.
