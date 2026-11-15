# Сборка, запуск и быстрая проверка SSB

Инструкция для текущего **Crystal-PCUDA**, Linux и Bash. Все команды, если явно
не сказано иначе, выполняются **из корня репозитория**. Движок использует C++17,
AdaptiveCpp generic PCUDA и GPU; CPU-only вариант ниже собирает тесты, а не
CPU-версию SQL-движка.

Первичная аппаратная проверка выполнена на AMD Radeon 780M. NVIDIA/Intel этой
инструкцией не объявляются аппаратно проверенными. SQL SLA и окончательная
приёмка этапа 8 остаются открытыми.

## Содержание

- [Что выбрать](#choose)
- [Зависимости и переменные окружения](#requirements)
- [Варианты сборки](#build)
- [Какие нужны данные](#data)
- [Запуск CLI и конфигурация](#run)
- [Быстро проверить SSB](#ssb)
- [Диагностика и измерения](#timing)
- [Установка](#install)
- [Типичные проблемы](#troubleshooting)
- [Что проверено при обновлении инструкции](#verified)

<a id="choose"></a>

## 1. Что выбрать

| Задача | Вариант | Нужны GPU / SF10-данные |
| --- | --- | --- |
| Проверить host-код без GPU-стека | `CRYSTAL_BUILD_GPU=OFF`, затем CTest | Нет / нет |
| Запускать SQL | GPU Release, target `db_cli` | Да / да |
| Получить все обычные targets и тесты | GPU Release, сборка без `--target` | Для сборки нужен SDK; для SQL — GPU и данные |
| Отлаживать host-код | Отдельная GPU Debug-сборка | Для SQL — да / да |
| Проверить Q3.3 без SQL-компилятора | Target `ssb_q33` | Да / да |
| Проверить все 13 SQL с эталонами | [SSB runner](#ssb-sql) | Да / да |
| Проверить все 13 ручных реализаций | [Manual runner](#ssb-manual) | Да / да |
| Прогнать широкую регрессию | [Functional acceptance](#regression) | Да / да; дополнительно NumPy |

**Короткий маршрут:** подготовить зависимости → собрать GPU Release → разместить
готовые данные в `data/10/` → выполнить один `q33.sql` → запустить SSB runner.
Не начинайте с большой performance-серии: это не быстрый smoke test.

<a id="requirements"></a>

## 2. Зависимости и переменные окружения

### 2.1. Что должно быть установлено

- C++17 compiler. Для GPU-сборки рекомендуется совместимый с установленным
  AdaptiveCpp Clang/LLVM; произвольный системный Clang не обязательно совместим.
- CMake **3.20+**, Make либо Ninja. Проект допускает CMake 3.16+, но для GPU PCH
  при CMake <3.20 необходимо выбрать генератор Ninja (`-G Ninja`).
- Python **3.10+**, Bash, Git; `gcc`, `make`, `ar` для скрипта зависимостей.
- `hyrise/sql-parser` и `linenoise` в локальном `deps/`.
- Для GPU: AdaptiveCpp с поддержкой `--acpp-targets=generic --acpp-pcuda`,
  соответствующий GPU runtime/драйвер и LLD. По умолчанию SQL-модули линкуются
  с `-fuse-ld=lld;-Wl,-O0`.

Установка AdaptiveCpp/ROCm **не входит** в `install-deps.sh`. Сначала настройте
совместимый стек по документации вашей поставки:
[PCUDA в AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/pcuda.md).
Сверяйте возможности именно установленной версии.

Для быстрых SSB runners достаточно стандартной библиотеки Python. NumPy нужен
для некоторых расширенных независимых oracles, включая manifest/Text-регрессию.

### 2.2. Локальные зависимости проекта

Проверьте наличие:

```bash
test -r deps/include/SQLParser.h
test -r deps/lib/libsqlparser.so
test -r deps/include/linenoise.h
test -r deps/lib/liblinenoise.a
```

Если они уже подготовлены, **не запускайте установщик повторно поверх них**.
Скрипт клонирует библиотеки, перемещает заголовки и удаляет рабочие каталоги
`sql-parser/` и `linenoise/`. Безопаснее выполнять его в новом каталоге, а готовый
`deps/` копировать только если в проекте его ещё нет:

```bash
ROOT="$PWD"
mkdir -p "$ROOT/build"
if [ -e "$ROOT/deps" ] || [ -L "$ROOT/deps" ]; then
  echo 'deps/ уже существует: проверьте файлы, не перезаписывайте каталог.'
else
  DEPS_WORK=$(mktemp -d "$ROOT/build/deps-XXXXXXXX")
  (cd "$DEPS_WORK" && bash "$ROOT/install-deps.sh") &&
    cp -a "$DEPS_WORK/deps" "$ROOT/deps"
fi
```

Нужны сеть и доступ к GitHub. Скрипт не фиксирует revision сторонних библиотек:
для воспроизводимости сохраняйте версии/хеши фактически собранного `deps/`.
Не удаляйте имеющиеся зависимости ради повторения инструкции.

### 2.3. Пути compiler/SDK

Пример **для проверенного стенда**, не обязательные пути для другой машины:

```bash
ROOT="$PWD"
ACPP=/opt/AdaptiveCpp
LLVM=/opt/rocm_sdk_612
CXX="$LLVM/bin/clang++"
BUILD="$ROOT/build/pcuda-release"
CPU_BUILD="$ROOT/build/pcuda-cpu"
HOST_FLAGS=''
PCUDA_FLAGS=''
export PATH="$ACPP/bin:$LLVM/bin:$PATH"
export ACPP_ADAPTIVITY_LEVEL=1
export ACPP_ALLOCATION_TRACKING=1

"$CXX" --version
"$ACPP/bin/acpp" --version
"$LLVM/bin/ld.lld" --version
cmake --version
```

Замените `ACPP`, `LLVM` и `CXX` на свои. В примерах пакет AdaptiveCpp находится
в `$ACPP/lib/cmake/AdaptiveCpp`; у другой установки путь может отличаться.
Runtime-библиотеки должны находиться загрузчиком: используйте окружение вашей
поставки SDK, а при ошибках проверяйте `ldd "$BUILD/db_cli"`. Не смешивайте
случайные версии `libacpp-rt`, HIP/HSA и compiler plugins через `LD_LIBRARY_PATH`.

**Clang и GCC headers:** на проверенном стенде Clang 17 автоматически выбирал
слишком новые GCC headers. Для него перед конфигурацией требуется явный GCC 14:

```bash
GCC_INSTALL=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
HOST_FLAGS="--gcc-install-dir=$GCC_INSTALL"
PCUDA_FLAGS="--gcc-install-dir=$GCC_INSTALL"
```

На другой машине не копируйте этот путь вслепую. Если выбор standard library
корректен, оставьте оба значения пустыми. Важно передать совместимый выбор
**и host compiler, и PCUDA/PCH compiler**, а не только одному из них.

<a id="build"></a>

## 3. Варианты сборки

Для разных compiler, генераторов, Release/Debug и GPU/CPU-only используйте
разные build-каталоги. Не переносите старый PCH между ними.

### 3.1. GPU Release: обычный вариант

После настройки переменных из раздела 2:

```bash
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DCMAKE_CXX_FLAGS=$HOST_FLAGS" \
  -DCRYSTAL_BUILD_GPU=ON \
  -DACPP_TARGETS=generic \
  "-DAdaptiveCpp_DIR=$ACPP/lib/cmake/AdaptiveCpp" \
  "-DCRYSTAL_PCUDA_COMPILER=$ACPP/bin/acpp" \
  "-DCRYSTAL_PCUDA_FLAGS=$PCUDA_FLAGS" \
  '-DCRYSTAL_PCUDA_LINK_FLAGS=-fuse-ld=lld;-Wl,-O0' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "$BUILD" -j4
ctest --test-dir "$BUILD" --output-on-failure
```

Полная сборка создаёт `db_cli`, все `ssb_qNN`, служебные GPU-библиотеки, PCH и
test targets. CTest запускает host-проверки; отдельные GPU/SSB-клиенты в него
автоматически не входят. Assertions в тестах включены и в Release.

**Если сейчас нужен только SQL CLI**, вместо полной сборки:

```bash
cmake --build "$BUILD" --target db_cli -j4
```

Этого достаточно для SQL SSB runner. Но остальные тестовые бинарники и ручные
`ssb_qNN` могут ещё отсутствовать; перед CTest/широкой регрессией соберите все targets.

### 3.2. Только host-тесты: без AdaptiveCpp и GPU

Нужен C++17 compiler и `sql-parser` в `deps/`; SF10-данные не нужны:

```bash
cmake -S "$ROOT" -B "$CPU_BUILD" \
  -DCRYSTAL_BUILD_GPU=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DCMAKE_CXX_FLAGS=$HOST_FLAGS"
cmake --build "$CPU_BUILD" -j4
ctest --test-dir "$CPU_BUILD" --output-on-failure
```

На машине без SDK задайте `CXX` своим host compiler и подходящие `HOST_FLAGS`.
В этом варианте нет `db_cli` и `ssb_qNN`: это **не CPU SQL fallback**.

### 3.3. Debug host-кода

Отдельная конфигурация с тем же toolchain:

```bash
BUILD="$ROOT/build/pcuda-debug"
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DCMAKE_CXX_FLAGS=$HOST_FLAGS" \
  -DCRYSTAL_BUILD_GPU=ON -DACPP_TARGETS=generic \
  "-DAdaptiveCpp_DIR=$ACPP/lib/cmake/AdaptiveCpp" \
  "-DCRYSTAL_PCUDA_COMPILER=$ACPP/bin/acpp" \
  "-DCRYSTAL_PCUDA_FLAGS=$PCUDA_FLAGS" \
  '-DCRYSTAL_PCUDA_LINK_FLAGS=-fuse-ld=lld;-Wl,-O0'
cmake --build "$BUILD" --target db_cli -j4
```

Без `--target db_cli` собираются и остальные Debug-targets.
После подготовки данных Debug CLI запускается как `"$BUILD/db_cli"`; если в shell
уже задан `CRYSTAL_PCUDA_MANIFEST`, выберите manifest именно этого build-каталога.
При возвращении к Release восстановите `BUILD="$ROOT/build/pcuda-release"`.
Debug относится к host-сборке; динамические SQL-модули по-прежнему используют
предусмотренные проектом `-O0 -g0`, а не автоматически host `-O3` или debug symbols.

### 3.4. Что означают основные настройки

| Настройка | Назначение |
| --- | --- |
| `CRYSTAL_BUILD_GPU` | Создавать GPU runtime, CLI и GPU-клиенты; default `ON` |
| `CMAKE_BUILD_TYPE` | Оптимизация/отладка host-сборки; обычно `Release` |
| `ACPP_TARGETS=generic` | Generic-путь для заранее собираемых AdaptiveCpp targets |
| `CRYSTAL_PCUDA_COMPILER` | Исполняемый `acpp` для PCH и SQL-модулей |
| `CRYSTAL_PCUDA_FLAGS` | Дополнительные флаги PCH/SQL-модулей, **CMake list** через `;` |
| `CRYSTAL_PCUDA_LINK_FLAGS` | Флаги линковки SQL `.so`, тоже CMake list |
| `CRYSTAL_INCLUDE_DIR` | Заголовки движка для JIT; default `<source>/include` |
| `CRYSTAL_DEPS_INCLUDE_DIR` | Совместимая настройка пути `deps/include`; не заменяет установку локальных библиотек |
| `CRYSTAL_PCUDA_JIT_DIR` | Каталог подготовленного PCH/manifest; default `<build>/pcuda-jit` |

Число `-j4` у CMake управляет сборкой приложения. Runtime-настройка `jit_jobs=4`
отдельно ограничивает параллельные compile/link-цепочки SQL. Это не одна настройка.
PCH готовится при build/install, а не заново при каждом запуске CLI.

<a id="data"></a>

## 4. Какие нужны данные

### 4.1. Текущий формат — готовый SF10

Данные не скачиваются при сборке и не входят в Git. По умолчанию CLI открывает
`data/10/` **относительно текущего рабочего каталога**, не каталога бинарника.
Текущий обычный CLI настроен на фиксированный SF10:

| Таблица | Строк | Файлы |
| --- | ---: | --- |
| LINEORDER | 59 986 217 | `LINEORDER0` … `LINEORDER16` |
| SUPPLIER | 20 000 | `SUPPLIER0` … `SUPPLIER6` |
| CUSTOMER | 300 000 | `CUSTOMER0` … `CUSTOMER7` |
| PART | 800 000 | `PART0` … `PART8` |
| DDATE | 2 557 | `DDATE0` … `DDATE16` |

Всего **58 файлов** без расширений. Это не `.tbl` и не CSV: схема находится в
[`include/storage/ssb_schema.h`](../include/storage/ssb_schema.h).

- Числа — Int32 little-endian, ровно четыре байта на строку, без заголовка.
- Flags — ASCII `'0'`/`'1'`, один байт на строку.
- FixedText — фиксированная ширина из схемы, декодирование до bounded NUL либо
  на всю ширину; содержимое должно быть валидным UTF-8.
- City/nation/region и manufacturer/category/brand уже представлены числовыми
  кодами физической схемы. Остальные Text-колонки нельзя заменять Int32-словарём.

Быстрая проверка трёх характерных размеров:

```bash
stat -c '%n: %s bytes' \
  data/10/LINEORDER5 data/10/CUSTOMER7 data/10/DDATE16
```

Для указанного SF10 ожидаются **239944868**, **3000000**, **2557** байт
соответственно. Это только предварительная проверка, не валидация всех данных.
Остальные файлы, размеры и Text также проверяются загрузчиком.

Если готовый набор находится вне проекта, а `data/10` ещё не существует:

```bash
mkdir -p data
test ! -e data/10 && test ! -L data/10 &&
  ln -s /absolute/path/to/converted-sf10 data/10
```

Не заменяйте существующий набор, не удаляйте его ради smoke test.

### 4.2. Ограничение старых инструментов подготовки

`bench/ssb-dbgen` — отдельный submodule; для его исходников нужен
`git submodule update --init bench/ssb-dbgen`. Но получение `.tbl` через dbgen
**ещё не создаёт** описанный выше набор бинарных колонок.

**Не используйте `bench/loader/load_modified.c` как готовую подготовку данных
для текущего Text-движка:** он пишет `CUSTOMER7` через `strtol`/`sizeof(int)`,
тогда как нужен FixedText шириной 10. Опция `--encode-mktsegment` у
`bench/loader/convert.py` также не подходит этому контракту. В этом руководстве
не заявляется проверенный end-to-end импорт из dbgen; используйте совместимый
подготовленный набор. Исправление legacy loader — отдельная работа.

У CLI нет опции `--data`, а изменение окружения `BASE_PATH` не меняет бинарник:
`BASE_PATH` — compile definition. `--data` у некоторых Python/manual-клиентов
не перенастраивает путь загрузки уже собранного SQL CLI. Другой SF нельзя
включить простым runtime-флагом.

<a id="run"></a>

## 5. Запуск CLI и конфигурация

### 5.1. Изолировать настройки smoke test

Сначала вернитесь к Release-build и создайте новый каталог результатов:

```bash
cd "$ROOT"
BUILD="$ROOT/build/pcuda-release"
mkdir -p "$ROOT/build/smoke" /tmp/opencode
RUN=$(mktemp -d "$ROOT/build/smoke/run-XXXXXXXX")
printf 'Artifacts: %s\n' "$RUN"
cp crystal.conf.example "$RUN/crystal.conf"
export CRYSTAL_CONFIG="$RUN/crystal.conf"
export CRYSTAL_PCUDA_MANIFEST="$BUILD/pcuda-jit/toolchain.manifest"
export CRYSTAL_JIT_OVERLAP_EXPERIMENT=off
export ACPP_APPDB_DIR="$RUN/device-cache"
export LC_ALL=C
```

`RUN` нужен для следующих примеров; сохраните напечатанные runners пути.
Некоторые текущие runners сами используют `/tmp/opencode`, поэтому этот каталог
должен существовать и быть доступен вашему пользователю. Чужие права не меняйте.
Новый `ACPP_APPDB_DIR` изолирует device JIT-cache; OS/HIP/driver caches не очищаются.

При запуске сначала создаются runtime/каталог и **загружаются все базовые колонки**
на GPU, затем появляется консоль. Нужны несколько ГБ памяти под данные, staging
и workspace. Даже маленький SQL-запрос не отменяет eager loading. Первое
построение статистик также может занять время; это не зависание на SQL parser.

### 5.2. Интерактивно

```bash
"$BUILD/db_cli"
```

Внутри консоли:

```text
help
\dump off
\timing on
\limit all
SELECT COUNT(*) FROM ddate;
\q
```

SQL заканчивается `;`, многострочный ввод поддерживается. Мета-команды вводятся
отдельной строкой без `;`. `.exit`, `\q` или EOF завершают сессию.

Полезные команды:

| Команда | Действие |
| --- | --- |
| `\timing on` / `off` | Подробные времена и module counters |
| `\limit N` | Показывать последние N строк результата; это не SQL `LIMIT` |
| `\limit all` | Не обрезать CPU-вывод; используйте для сравнения SSB |
| `\dump on` / `off` | Генерировать код **вместо исполнения** / исполнять SQL |
| `\o filename` | Записывать generated code в файл при `\dump on`, не перенаправлять обычные результаты |

Ключи запуска вроде `db_cli --help`/`--data` не реализованы: текущий `main()`
не разбирает аргументы. Для справки используйте `help` внутри консоли.

### 5.3. SQL-файл и несколько запросов

Один запрос, без смены рабочего каталога:

```bash
"$BUILD/db_cli" < test/queries/sql/q33.sql
```

Два одинаковых запроса **в одном процессе**, чтобы увидеть reuse:

```bash
{
  printf '%s\n' '\dump off' '\timing on' '\limit all'
  cat test/queries/sql/q33.sql
  printf '\n'
  cat test/queries/sql/q33.sql
  printf '\n%s\n' '\q'
} | "$BUILD/db_cli" > "$RUN/q33-twice.stdout" 2> "$RUN/q33-twice.stderr"
```

Во втором запросе должны быть нулевые `plan_codegen`, `module_codegen`,
`compiler`, `linker`, `loads`, `symbols` в `Module events`. `hits` нулевым быть
не обязан. Новый процесс снова теряет SQL plan/module cache; device cache —
другой слой и может сохраняться в `ACPP_APPDB_DIR`.

CLI перехватывает ошибки SQL и продолжает сессию: exit code 0 сам по себе не
доказывает успешный запрос. Проверяйте `[ERROR]`, наличие результата или используйте
runner, который разбирает transcript и сравнивает ожидаемые данные.

### 5.4. Где берутся настройки

Читается первый существующий config:

1. путь из `CRYSTAL_CONFIG`;
2. `./crystal.conf`;
3. `./crystal-sycl.conf`;
4. `$HOME/.config/crystal-sycl/config.conf`.

Затем явные env overrides имеют приоритет. `CRYSTAL_DEVICE_NAME` выбирает
единственное GPU с подходящей без учёта регистра подстрокой, например:

```bash
CRYSTAL_DEVICE_NAME='780M' "$BUILD/db_cli" < test/queries/sql/q33.sql
```

Пустое имя выбирает первое перечисленное GPU; отсутствующий или неоднозначный
match — ошибка, не молчаливый CPU fallback.

Основные параметры [`crystal.conf.example`](../crystal.conf.example):

| Ключ | Значение/смысл |
| --- | --- |
| `jit_jobs` | 1–4 compile/link-цепочки; default 4 |
| `jit_keep_artifacts` | Сохранять `.cpp`, `.so`, команды и native timing logs; default off |
| `jit_artifact_dir` | Родитель приватного каталога сессии; пусто — системный temporary directory |
| `jit_traces` | Clang/LLD traces для диагностики; default off |
| `pcuda_manifest` | Явный PCH/toolchain manifest; env `CRYSTAL_PCUDA_MANIFEST` приоритетнее |
| `dump_generated_code` | Генерировать код вместо запроса; для SQL-проверок нужен off |
| `extended_timing` | Подробные таймеры; эквивалент настройки `\timing` |
| `output_limit` | Default 1000; `off`/`all` отключают ограничение вывода |
| `reuse_scratch_buffers` | Переиспользовать workspace текущего совместимого плана; default true |
| `memory_guard_enabled` | Защита памяти, default on; не отключайте для маскировки OOM |
| `memory_guard_reserve_fraction` / `memory_guard_reserve_bytes` | Default 0.10 / 512MiB |
| `jit_overlap_experiment` | Исследовательское перекрытие S1/исполнения; default off |

Boolean принимает `on/off`, `true/false`, `yes/no`, `1/0`. Неизвестный ключ
вызывает ошибку. Не копируйте CMake-параметры compiler flags в runtime config.

<a id="ssb"></a>

## 6. Быстро проверить SSB

Все команды ниже предполагают готовые данные и окружение раздела 5.1.
Не запускайте GPU-клиенты параллельно друг другу.

<a id="ssb-sql"></a>

### 6.1. Все 13 SQL, один процесс, сравнение с эталонами

Рекомендуемая быстрая проверка: каждый запрос два раза — первый запуск и reuse.
Данные загружаются один раз на всю сессию:

```bash
python3 test/pcuda_stage5/validate_cli.py \
  --db-cli "$BUILD/db_cli" \
  --mode ssb --runs 2 \
  --timing-model pcuda-modules-v1 \
  --output-root "$RUN/sql-ssb"
```

Минимальный smoke без проверки повторного запуска: замените `--runs 2` на
`--runs 1`. При `--runs 1` значение `warm_reuse_verified` не подтверждает reuse:
проверять было нечего. Это не серия независимых cold-процессов на каждый SQL:
общие service kernels могут переиспользоваться между запросами.

Runner печатает `Artifacts: .../ssb-plain-...`. В этом каталоге:

- `report.json` — каждый запрос, строки результата, статус, ошибки, таймеры и counters;
- `stdout.txt`, `stderr.txt`, `stdin.sql` — полная сессия;
- `process.time.json` — native время всего процесса, **не** SQL summary;
- сохранённые JIT-команды/модули и приватный device-cache.

Для совместимого SF10 ожидается особенность: **24/26 file matches** при двух
повторах либо **12/13** при одном. Только Q1.1 не совпадает с историческим файлом.
Из-за него runner возвращает **exit code 1** и `success:false`. Не заменяйте это
на общий success и не игнорируйте другие failures; выполните независимую проверку
ниже. При `set -e` учитывайте этот ненулевой статус отдельно.

### 6.2. Почему Q1.1 отличается и как проверить

[`q11.sql`](../test/queries/sql/q11.sql) включает `1994-01-01`, а ручная
реализация/`scale10/11.txt` — только 1993 год. Это известная граница, не повод
менять SQL или эталонный файл. Свежий независимый CPU oracle:

```bash
"$CXX" -std=c++17 -O3 ${HOST_FLAGS:+$HOST_FLAGS} \
  test/pcuda_stage0/q11_reference.cpp -o "$RUN/q11-reference"
"$RUN/q11-reference" "$ROOT/data/10" > "$RUN/q11-oracle.json"
cat "$RUN/q11-oracle.json"
```

Для проверенного набора: `sql_q11=4486422947100`, `year_1993=4474878955766`,
`january_1_1994=11543991334`. Сверьте результат Q1.1 в `report.json` с **новым**
`sql_q11` своего oracle, а не только с этой напечатанной константой. Должно
выполняться `sql_q11 = year_1993 + january_1_1994`. Остальные 12 запросов должны
совпадать с файлами, с сохранением повторяющихся строк.

<a id="ssb-manual"></a>

### 6.3. Ручные SSB: без SQL parser/codegen

Самый короткий отдельный пример:

```bash
cmake --build "$BUILD" --target ssb_q33 -j4
"$BUILD/ssb_q33" "$ROOT/data/10" 1
```

Формат: `ssb_qNN [data-directory [trials]]`; default trials 4, допустимо 1–1000.
Targets: `ssb_q11`, `ssb_q12`, `ssb_q13`, `ssb_q21`, `ssb_q22`, `ssb_q23`,
`ssb_q31`, `ssb_q32`, `ssb_q33`, `ssb_q34`, `ssb_q41`, `ssb_q42`, `ssb_q43`.

Для автоматической проверки всех 13 сначала соберите ручные targets:

```bash
cmake --build "$BUILD" --target \
  ssb_q11 ssb_q12 ssb_q13 ssb_q21 ssb_q22 ssb_q23 \
  ssb_q31 ssb_q32 ssb_q33 ssb_q34 ssb_q41 ssb_q42 ssb_q43 -j4
python3 test/pcuda_stage7/validate.py manual \
  --build "$BUILD" --data "$ROOT/data/10" --trials 1 \
  --timing-model pcuda-modules-v1 \
  --output-root "$RUN/manual-ssb"
```

Ожидается **13/13** независимых file matches. Для обычных четырёх повторов
поставьте `--trials 4`: получится 52 проверки. Ручной Q1.1 использует границу
эталона и совпадает с ним. Эти реализации требуют nonnullable input и уникальные
dimension keys; это не заменяет SQL-проверки NULL/MHT/OR.

Время `Query CPU-ready ms` относится к ручной реализации, **не** к SQL parse,
codegen и сборке SQL-модулей. Prebuilt GPU-код всё ещё может требовать первого
внутреннего device JIT.

### 6.4. Логи и CSV через общий benchmark runner

Если нужны удобные CSV/логи только SSB-класса:

```bash
python3 olap_bench/tools/run_olap_bench.py \
  --db-cli "$BUILD/db_cli" \
  --manifest olap_bench/manifest.json \
  --class 07_ssb_standard \
  --mode single-session --warmups 0 --runs 2 \
  --limit all --no-dump-code \
  --timing-model pcuda-modules-v1 \
  --out "$RUN/ssb-csv"
```

Будут `summary.csv` (raw records), `aggregate_summary.csv`, `logs/`, `outputs/`.
Этот runner проверяет ошибки исполнения, но **не сравнивает payload с SSB oracle**:
успех здесь не заменяет 6.1/6.2. `single-session` сохраняет процесс и кеши;
`isolated` создаёт отдельные процессы, повторно загружает БД и не измеряет warm
того же процесса. Старое смешанное mean/first агрегирование CSV не является
подтверждающей cold-median серией.

<a id="regression"></a>

### 6.5. Широкая функциональная регрессия — уже не быстрый smoke

Нужна полная GPU-сборка, NumPy в выбранном Python и достаточно памяти:

```bash
cmake --build "$BUILD" -j4
ctest --test-dir "$BUILD" --output-on-failure
python3 test/pcuda_stage8/validate.py \
  --build "$BUILD" --timing-model pcuda-modules-v1 \
  --output-root "$RUN/full-regression"
```

Набор включает GPU/resource/ABI/NULL, Text, все SSB, manual и current/current_heavy
manifest. Родительский отчёт отдельно допускает прежний Q1.1 mismatch, но сохраняет
`full_file_parity:false` и исходный failed SSB child report; CPU oracle запускайте
отдельно. Категории `future` автоматически не становятся поддерживаемыми.
Исторический `test/pcuda_stage7/validate.py storage` проверял прежний numeric-only
этап с Text rejection; не используйте его как текущую общую Text-регрессию.

<a id="timing"></a>

## 7. Диагностика и измерения

### 7.1. Как читать времена

- Первый `Total query time` — **summary**: от SQL в уже загруженной БД до готового
  CPU-результата. Startup, загрузка base data и печать результата сюда не входят.
- `Generated execute time` — host/runtime execution interval с необходимыми
  ожиданиями и возможным device JIT, не чистое время GPU kernel.
- `Module build elapsed` — параллельная подготовка отсутствующих модулей;
  `Native compile/link ... sum` — суммы отдельных процессов. Они не складываются
  с elapsed для восстановления latency.
- Warm/совместимые значения проверяются нулевыми codegen/compiler/link/load/symbol
  counters, а не только коротким временем.

Точные границы — [`timing.md`](../timing.md). Runtime trace-флаги диагностические,
не ускоряющие. Для обычного smoke оставляйте `jit_traces=off`.

### 7.2. Перекрытие S1/исполнения — только явный эксперимент

```bash
CRYSTAL_JIT_OVERLAP_EXPERIMENT=on \
  "$BUILD/db_cli" < test/queries/sql/q42.sql
```

Это меняет runtime-путь на `pcuda-modules-overlap-exp-v1`, но не включает replay,
parallel S2 или jobs>4. Default остаётся off. Для stage-5 runner выбирайте
`--timing-model pcuda-modules-overlap-exp-v1`: он сам выставит соответствующий
env override. Общему CSV runner нужны **и env override, и matching timing model**;
его `--timing-model` сам по себе не включает оптимизацию. В эксперименте
build/load/execution перекрываются, component sums/residuals — null.
Результаты и границы — [8-ANALYSIS](../test/pcuda_stage8_analysis/REPORT.md).

### 7.3. Настоящая cold/warm-серия — не quick test

Нужен отдельный native first-result launcher. Для штатного пути:

```bash
cmake --build "$BUILD" --target db_cli pcuda_acceptance_first_result -j4
CRYSTAL_JIT_OVERLAP_EXPERIMENT=off \
python3 test/pcuda_stage8/measure.py measure \
  --build "$BUILD" --output-root "$RUN/performance" \
  --sessions 7 --runs 9 --timing-model pcuda-modules-v1
```

Это 7 cold-процессов на SQL/режим, cold +8 warm в каждом, plain/traced отдельно;
Q3.3 дополнительно меняет параметры. **Не запускайте** рядом сборки, другие GPU
клиенты, CPU oracles или online-анализ. После завершения **всей** серии запустите
oracle из 6.2 и отдельный анализ, подставив напечатанный каталог `ssb-series-*`:

```bash
python3 test/pcuda_stage8/measure.py analyze \
  --series "$RUN/performance/ssb-series-ID" \
  --q11-oracle "$RUN/q11-oracle.json" \
  --output "$RUN/performance-analysis"
```

Проверку oracle после серии выполните заново; не запускайте измеритель поверх
исторических raw. Для небольшой диагностической серии можно выбрать `--query q33`,
но её нельзя выдавать за проверку всех 13. Методика и ограничения —
[отчёт этапа 8](../test/pcuda_stage8/REPORT.md). Числа прототипов и единичные smoke
timings не являются SLA полного SQL.

<a id="install"></a>

## 8. Установка

После GPU-сборки, в **конечный абсолютный prefix**:

```bash
PREFIX="$ROOT/build/pcuda-install"
cmake --install "$BUILD" --prefix "$PREFIX"
```

Устанавливаются `bin/db_cli`, `lib/libsqlparser.so`, заголовки и заново
подготовленный installed PCH/manifest. Ручные `ssb_qNN` и tests этим install
recipe не устанавливаются. AdaptiveCpp/LLVM/GPU runtime по-прежнему нужны:
это не standalone-бинарник без compiler/SDK.

Clang PCH хранит абсолютные пути. Нельзя просто перенести prefix или применить
`DESTDIR`: установленный PCH нужно подготовить в конечном расположении.
При выборе manifest приоритет следующий:

1. `CRYSTAL_PCUDA_MANIFEST`;
2. runtime `pcuda_manifest`;
3. `../lib/crystal/pcuda-jit/toolchain.manifest` относительно бинарника;
4. build-tree manifest, встроенный при сборке.

Чтобы предыдущий smoke env не выбрал build-tree PCH, укажите installed manifest
явно. Рабочий каталог по-прежнему корень проекта с `data/10/`:

```bash
CRYSTAL_PCUDA_MANIFEST="$PREFIX/lib/crystal/pcuda-jit/toolchain.manifest" \
  "$PREFIX/bin/db_cli" < test/queries/sql/q33.sql
```

<a id="troubleshooting"></a>

## 9. Типичные проблемы

| Симптом | Что проверить |
| --- | --- |
| CMake не нашёл AdaptiveCpp/acpp | `AdaptiveCpp_DIR`, `CRYSTAL_PCUDA_COMPILER`, поддержку PCUDA установленным compiler; CPU-only не требует AdaptiveCpp |
| Ошибки `<string>`/`<type_traits>` при Clang 17 | Совместимые GCC headers; передать `--gcc-install-dir` в оба набора flags и создать новый build-каталог |
| Не найден `SQLParser.h`, `linenoise.h`, `-lsqlparser` | Полноту `deps/include` и `deps/lib`; `install-deps.sh` не устанавливает GPU SDK |
| `libsqlparser.so`/`libacpp-rt.so` not found | `ldd`, build/install RPATH, окружение установленного SDK; не смешивать версии runtime |
| Manifest/PCH несовместим | Выбранный manifest, прежние env overrides, compiler/flags/headers; пересобрать `cmake --build "$BUILD" --target pcuda_jit_pch`, при смене toolchain — новая конфигурация |
| После build PCH всё ещё отклонён | Посмотреть конкретный fingerprint mismatch; не удалять manifest и не отключать проверки, runtime сам PCH не пересоздаёт |
| Нет `data/10/...` | Рабочий каталог, готовые бинарные колонки, правильные имена; не запускать через `cd build` |
| Неверный размер `CUSTOMER7` | Нужен Text 10 bytes/row, не старое Int32-кодирование |
| Ожидали SF1/SF20 | Обычный CLI использует фиксированные SF10 размеры; runtime scale-флага нет |
| Memory guard/OOM | Свободную GPU/UMA и host memory, лишние клиенты, корректность размеров данных; сохранять guard включённым |
| Есть код, но нет результата | `\dump off`; `\dump on` генерирует исходники вместо исполнения |
| Результат обрезан | `\limit all` либо достаточный `output_limit` перед сравнением |
| SSB runner завершился с 1 | Смотреть `report.json`: допустимо только отдельно проверенное Q1.1 расхождение, не любые ошибки |
| Ошибка каталога `/tmp/opencode` | Создать доступный своему пользователю каталог; не менять права чужих файлов |

Не используйте старый [`test_db_cli.sh`](../test_db_cli.sh) как основной quick
runner: он меняет cwd на `build`, пишет только логи и не сравнивает результаты.
Предпочитайте команды раздела 6. Сохраняйте `test/pcuda_*` в репозитории: среди
них исходники действующих CMake targets, runners и независимых oracles.
Сборки, PCH, локальные конфиги и результаты прогонов храните в игнорируемом `build/`;
постоянный кеш скомпилированных SQL `.so` инструкцией не вводится.

<a id="verified"></a>

## 10. Проверка команд — 2026-11-15

На стенде AMD Radeon 780M, Clang/LLVM 17 из ROCm SDK 6.1.2 и установленном
AdaptiveCpp проверены новые, отдельные build-каталоги:

- Полная GPU Release-сборка и host CTest **19/19**; CPU-only сборка и **17/17**.
- Отдельная Debug-конфигурация и сборка `db_cli`; отдельный запуск Debug не повторялся.
- Интерактивные команды через stdin, один Q3.3, его повтор в одной сессии,
  установка в конечный prefix и Q3.3 из установленного CLI.
- SQL SSB: **24 file matches +2 прежних Q1.1**, отдельно проверенных свежим CPU
  oracle; исходный `success:false` сохранён. Manual SSB: **13/13**.
- CSV runner: **26 выполнений** без execution errors; это не отдельный payload oracle.
- Bash-синтаксис примеров и локальные Markdown-ссылки.

Локальные [команды и результаты](../build/docs-build-run/run-v9qyoz3y/validation-with-debug.json)
и [журнал действий](../build/docs-build-run/run-v9qyoz3y/actions.jsonl) находятся в
игнорируемом `build/` и не входят в clone. Два замечания к разовому скрипту сверки
вывода (ANSI/advisory и формат CTest summary) сохранены в артефактах; сборки и
клиенты из-за них не перезапускались, их raw не исправлялись.
Получение зависимостей из сети, подготовка нового SF10, полная GPU-регрессия и
performance-серии в этом документальном обновлении не повторялись.
