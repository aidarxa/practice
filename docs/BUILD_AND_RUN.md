# Сборка, запуск и быстрая проверка SSB

Инструкция для текущего **Crystal-PCUDA**, Linux и Bash. Все команды, если явно
не сказано иначе, выполняются **из корня репозитория**. Движок использует C++17,
AdaptiveCpp generic PCUDA и GPU; CPU-only вариант ниже собирает тесты, а не
CPU-версию SQL-движка. Подготовка данных собирается отдельно и GPU не требует.

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

| Задача | Вариант | Нужны GPU / данные |
| --- | --- | --- |
| Проверить host-код без GPU-стека | `CRYSTAL_BUILD_GPU=OFF`, затем CTest | Нет / нет |
| Создать готовые данные любого поддерживаемого SF | `bash bench/prepare_ssb.sh --scale SF` | Нет / создаёт сама |
| Запускать SQL | GPU Release, target `db_cli` | Да / выбранный native dataset |
| Получить все обычные targets и тесты | GPU Release, сборка без `--target` | Для сборки нужен SDK; для SQL — GPU и данные |
| Отлаживать host-код | Отдельная GPU Debug-сборка | Для SQL — да / да |
| Проверить Q3.3 без SQL-компилятора | Target `ssb_q33` | Да / да |
| Проверить все 13 SQL с эталонами | [SSB runner](#ssb-sql) | Да / да |
| Проверить все 13 ручных реализаций | [Manual runner](#ssb-manual) | Да / да |
| Прогнать широкую регрессию | [Functional acceptance](#regression) | Да / да; дополнительно NumPy |

**Короткий маршрут:** подготовить зависимости → собрать GPU Release → разместить
данные командой `bash bench/prepare_ssb.sh` → выполнить один `q33.sql` → запустить SSB runner.
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

### 4.1. Один запуск: dbgen → готовые колонки

Из корня проекта, даже до сборки движка:

```bash
bash bench/prepare_ssb.sh
```

Default — **SF10 в `data/ssb/sf10`**. Скрипт сам:

1. Собирает host-only C++17 конвертер и запускает его тесты.
2. При необходимости получает только submodule `bench/ssb-dbgen` на закреплённом
   revision. Уже изменённый submodule не перезаписывает, а выдаёт ошибку.
3. Собирает dbgen, генерирует все пять `.tbl` в приватном каталоге.
4. Потоково проверяет поля и создаёт 58 колонок без словаря.
5. Проверяет manifest/размеры и атомарно публикует **целый** новый набор.

Нужны Bash, Git, CMake и C/C++17 compiler, для первого получения submodule — сеть.
AdaptiveCpp, GPU, `deps/` и SQL parser для подготовки **не нужны**. Toolchain можно
выбрать обычными `CC`/`CXX`; пути к SDK не зашиты в конвертер. Например:

```bash
bash bench/prepare_ssb.sh --scale 1
bash bench/prepare_ssb.sh --scale 1 --output /absolute/path/to/new-sf1 --keep-tbl
bash bench/prepare_ssb.sh --scale 1 --from-tbl /path/to/original-tables \
  --output /absolute/path/to/imported-sf1
```

Все `--output` должны быть **новыми**. Скрипт не имеет `--force`: повторный запуск
не заменяет существующий набор или чужие файлы. Не удаляйте исходные данные ради
повторения команды. Ошибка конвертации не публикует частичный набор; приватные
staging-файлы при обработанной ошибке удаляются. Атомарность публикации не является
обещанием crash durability при отключении питания.

`--jobs N` задаёт параллельность host-сборки; default 4. Артефакты и CMake-команды
остаются в напечатанном `build/ssb-tools/prepare-*`. Обычно исходные `.tbl` после
успеха удаляются, `--keep-tbl` сохраняет их там для независимой побайтной проверки.
Это потребует дополнительного диска. `--from-tbl` никогда не удаляет входной каталог.
Файлы должны называться `lineorder.tbl`, `supplier.tbl`, `customer.tbl`, `part.tbl`,
**`date.tbl`**, с pipe-разделителем и завершающим `|`, без заголовков.

SF ≥1 должен быть целым: upstream молча усекал бы `1.5`, поэтому скрипт отклоняет
его. Десятичный SF <1 пригоден для smoke test, но dbgen масштабирует также DDATE:
например, SF0.01 дал лишь 25 дат. Для обычного SSB используйте целый SF ≥1.
Размеры таблиц всегда фактические, не вычисляются по SF. Очень большой SF может
превысить Int32-ключи генератора, диск или память устройства — скрипт не обещает
произвольный масштаб; переполнение полей отклоняется, а runtime сохраняет memory guard.

### 4.2. Формат и выбор набора

Набор содержит `dataset.manifest` с версией `CRYSTAL_SSB_NATIVE_1`, SF,
идентификатором генератора, реальными row counts, именами/типами/ширинами колонок.
Загрузчик проверяет manifest и все размеры **до GPU allocations**. Пример SF10:

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
- Все исходные строки, в том числе city/nation/region и manufacturer/category/brand,
  остаются **настоящим Text**. Никакого словаря или скрытых Int32-категорий нет.
- День недели занимает 9 байт: `Wednesday` сохраняется целиком; brand — до 10 байт.

Быстрая проверка трёх характерных размеров:

```bash
stat -c '%n: %s bytes' \
  data/ssb/sf10/LINEORDER5 data/ssb/sf10/CUSTOMER7 data/ssb/sf10/DDATE16
```

Для указанного SF10 ожидаются **239944868**, **3000000**, **2557** байт
соответственно. Это только предварительная проверка, не валидация всех данных.
Остальные файлы, размеры и Text также проверяются загрузчиком.

Выбор SF не требует пересборки. Передайте каталог через env или config:

```bash
CRYSTAL_DATA_DIR="$ROOT/data/ssb/sf1" "$BUILD/db_cli" < test/queries/sql/q33.sql
```

Default config: `data_dir = ./data/ssb/sf10`; `CRYSTAL_DATA_DIR` приоритетнее.
Относительный путь считается от cwd, абсолютный не зависит от cwd. Статистика
числовых колонок кешируется отдельно внутри выбранного набора
(`.crystal-numeric-stats-v1`), а не в общем SF10-кеше; read-only набор работает
без записи кеша. Статистика не является постоянным кешем SQL-модулей.

Прежний `data/10/`, старый `bench/loader/load_modified.c`/`convert.py` и числовые
`test/queries/result/scale10` сохранены **как исторические**. Они не совместимы
с native manifest/Text-схемой и не выбираются автоматически. Перекодировать их
поверх существующих файлов нельзя — создайте новый набор. CLI по-прежнему не
разбирает `--data`; используйте `CRYSTAL_DATA_DIR`.

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
export CRYSTAL_DATA_DIR="$ROOT/data/ssb/sf10"
export CRYSTAL_PCUDA_MANIFEST="$BUILD/pcuda-jit/toolchain.manifest"
export CRYSTAL_JIT_OVERLAP_EXPERIMENT=off
export ACPP_APPDB_DIR="$RUN/device-cache"
export LC_ALL=C
```

`RUN` нужен для примеров, которые явно используют эту переменную; сохраните
напечатанные runners пути. В новой оболочке переменные `ROOT`/`BUILD`/`RUN` нужно
задать снова. Не передавайте `"$RUN/…"`, если `RUN` не задан: получится путь `/…`
в корне файловой системы. Самостоятельный quick test без этих переменных — в 6.1.
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
| `data_dir` | Native dataset; default `./data/ssb/sf10`, env `CRYSTAL_DATA_DIR` приоритетнее |
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

Нужны готовые native данные и собранный движок. Пример 6.1 самодостаточен и не
требует переменных раздела 5.1; остальные примеры с `BUILD`/`RUN`/`CRYSTAL_DATA_DIR`
предполагают их предварительную настройку. Все команды выполняются **из корня
репозитория**, не из `test/ssb_native_data/`. Не запускайте GPU-клиенты параллельно.

<a id="ssb-sql"></a>

### 6.1. Все 13 SQL и manual с независимыми Text-oracles

Рекомендуемая быстрая проверка: каждый запрос два раза — первый запуск и reuse.
Данные загружаются один раз на всю сессию. Ниже используется build-каталог
`build/pcuda-release` из раздела 3; если вы собирали в `build/`, замените этот путь
в обеих командах на `build/`:

```bash
cmake --build build/pcuda-release -j4
python3 test/ssb_native_data/validate.py \
  --build build/pcuda-release --data data/ssb/sf10 --trials 2 \
  --timing-model pcuda-modules-v1
```

Первая команда обязательна, если ранее собирался только target `db_cli`: default
runner проверяет **SQL и manual**, поэтому нужны все `ssb_qNN` и CPU oracle
`bench/loader/ssb_reference`. Перед запуском SQL runner проверяет их наличие и
выдаёт команду сборки, если чего-то не хватает; сам он сборку не запускает.

Без `--output-root` runner сам создаёт результаты в
`build/ssb-native-validation/native-*`, каждый запуск — в отдельном каталоге.
Можно явно добавить `--output-root build/sql-ssb`; для этого `RUN` не нужен.
Не используйте `sudo` для обхода ошибки `/sql-ssb`: исправьте пустую переменную
или уберите `--output-root`, как в примере выше.

Минимальный smoke: `--trials 1`; только SQL: `--sql-only`. Для SQL-only ручные
`ssb_qNN` не нужны, но `db_cli` **и** CPU reference target `ssb_reference` обязательны.
Они входят в полную сборку. Нужен только стандартный Python;
NumPy требуется при добавлении `--manifest` и для широкой Text-регрессии.
SQL исполняются в одной сессии, manual — отдельным процессом на запрос.
Второй SQL проверяется нулевыми codegen/compiler/link/load/symbol counters.
Один trial не доказывает reuse. Это не серия независимых cold-процессов.

Runner печатает `Artifacts: .../native-...`. В этом каталоге:

- `report.json` — каждый запрос, строки результата, статус, ошибки, таймеры и counters;
- `sql/stdout.txt`, `sql/stderr.txt`, `sql/stdin.sql` — SQL-сессия;
- `qNN/` — команды и результаты отдельных manual-клиентов;
- `reference.json` — свежий независимый CPU oracle; `oracle.*` — его команда/вывод;
- `*/native.time` — native время всего процесса, **не** SQL summary;
- сохранённые JIT-команды/модули и приватный device-cache.

Ожидается **26/26 SQL payload checks и 26 manual результатов** при двух trials.
Города/страны/brands сравниваются как строки, не как числовые коды. Числовые
`scale10/*.txt` не являются oracle нового формата. Любое новое несовпадение даёт
ненулевой статус; нет исключения, скрывающего ошибку Q1.1.

Если подготовка выполнялась с `--keep-tbl`, можно дополнительно сверить **каждое
поле всех 58 колонок** с оригиналом: добавьте `--tables /path/to/prepare-ID/tables`.
Это потоковая CPU-проверка после завершения GPU-процессов, вне SQL summary.
Для current/current_heavy manifest и целевых dimension-проверок добавьте `--manifest`.
Oracle строится заново для выбранного набора; готовый чужой reference не подставляется.

### 6.2. Почему Q1.1 отличается и как проверить

[`q11.sql`](../test/queries/sql/q11.sql) включает `1994-01-01`, а ручная
реализация/`scale10/11.txt` — только 1993 год. Это известная граница, не повод
менять SQL или эталонный файл. Новый runner из 6.1 проверяет **обе границы**
раздельно: `reference.json` содержит `sql.q11` и `manual.q11`. Чтобы получить
CPU-результаты отдельно от GPU:

```bash
cmake --build "$BUILD" --target ssb_reference -j4
"$BUILD/bench/loader/ssb_reference" "$CRYSTAL_DATA_DIR" > "$RUN/ssb-reference.json"
```

Разница включает вклад `1994-01-01` и зависит от выбранного набора/SF. Исторические
failed file comparisons сохранены в старых отчётах; новый независимый oracle не
делает их успешными задним числом.

<a id="ssb-manual"></a>

### 6.3. Ручные SSB: без SQL parser/codegen

Самый короткий отдельный пример:

```bash
cmake --build "$BUILD" --target ssb_q33 -j4
"$BUILD/ssb_q33" "$CRYSTAL_DATA_DIR" 1
```

Формат: `ssb_qNN [data-directory [trials]]`; default trials 4, допустимо 1–1000.
Targets: `ssb_q11`, `ssb_q12`, `ssb_q13`, `ssb_q21`, `ssb_q22`, `ssb_q23`,
`ssb_q31`, `ssb_q32`, `ssb_q33`, `ssb_q34`, `ssb_q41`, `ssb_q42`, `ssb_q43`.

Для автоматической проверки всех 13 сначала соберите ручные targets:

```bash
cmake --build "$BUILD" --target \
  ssb_q11 ssb_q12 ssb_q13 ssb_q21 ssb_q22 ssb_q23 \
  ssb_q31 ssb_q32 ssb_q33 ssb_q34 ssb_q41 ssb_q42 ssb_q43 -j4
python3 test/ssb_native_data/validate.py \
  --build "$BUILD" --data "$CRYSTAL_DATA_DIR" --trials 1 \
  --timing-model pcuda-modules-v1 \
  --output-root "$RUN/manual-ssb"
```

Runner проверит также SQL. У manual все 13 результатов должны совпасть со свежим
CPU oracle. Для четырёх повторов поставьте `--trials 4`. Ручной вывод — JSON-массивы
между `__SSB_RESULT_BEGIN__`/`__SSB_RESULT_END__`; Text остаётся строками. Без явного
аргумента каталог выбирается так же, как в CLI, через config/env. Эти реализации
требуют nonnullable input и уникальные dimension keys; это не заменяет SQL-проверки
NULL/MHT/OR. Ссылки на строки в PHT/group table — не словарное кодирование Text.

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
manifest. Он использует новый native SSB validator, самостоятельно проверяющий обе
границы Q1.1; старые numeric file reports не изменяются. `full_file_parity:false`
по-прежнему не обещает oracle каждого поля произвольного manifest-запроса.
`--data PATH` выбирает набор для всей серии. Категории `future` автоматически не
становятся поддерживаемыми.
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
parallel S2 или jobs>4. Default остаётся off. Для native SSB runner выбирайте
`--timing-model pcuda-modules-overlap-exp-v1`: он сам выставит соответствующий
env override. Общему CSV runner нужны **и env override, и matching timing model**;
его `--timing-model` сам по себе не включает оптимизацию. В эксперименте
build/load/execution перекрываются, component sums/residuals — null.
Результаты и границы — [8-ANALYSIS](../test/pcuda_stage8_analysis/REPORT.md).

### 7.3. Производительность после смены формата

Прежние `test/pcuda_stage8/measure.py`/stage-5 numeric file validators и измерения
относятся к старой кодированной схеме. Их нельзя применять как Text oracle или
сопоставимый baseline нового алгоритма. Новая correctness-серия не утверждает SLA,
ускорение Text или закрытие этапа 8. Для диагностики native runner поддерживает
`--traces` с сохранением Clang/LLD traces, но такие времена не являются plain baseline.

Для новой performance-приёмки отдельно согласуются нагрузка и цели. Обязательны
непрерывный summary, native process timing, раздельные cold/warm, минимум семь
независимых cold-процессов и восемь warm на каждый, plain/traced серии с ротацией,
состояние всех кешей и offline oracle после **всей** серии. Не запускайте рядом
сборки, GPU-клиенты или online-анализ. Историческая методика и ограничения —
[отчёт этапа 8](../test/pcuda_stage8/REPORT.md) и [timing.md](../timing.md).

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
явно. Для относительного `data_dir` рабочий каталог — корень проекта;
`CRYSTAL_DATA_DIR` с абсолютным путём позволяет выбрать набор из любого cwd:

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
| Нет `dataset.manifest` или неверная схема | Выбранный `data_dir`/`CRYSTAL_DATA_DIR`; создать новый native набор через `bench/prepare_ssb.sh`, не подсовывать старый `data/10` |
| Неверный размер `CUSTOMER7` | Нужен Text 10 bytes/row, не старое Int32-кодирование |
| Ожидали другой SF | Создать новый набор `--scale SF`, выбрать его каталог; row counts берутся из manifest, runtime-флаг SF не нужен |
| Memory guard/OOM | Свободную GPU/UMA и host memory, лишние клиенты, корректность размеров данных; сохранять guard включённым |
| Есть код, но нет результата | `\dump off`; `\dump on` генерирует исходники вместо исполнения |
| Результат обрезан | `\limit all` либо достаточный `output_limit` перед сравнением |
| Native SSB runner завершился с 1 | Смотреть `report.json`: это ошибка исполнения/oracle/reuse. Обе границы Q1.1 проверяются отдельно, исключения для mismatch нет |
| `PermissionError: '/sql-ssb'` | `RUN` пуст: `"$RUN/sql-ssb"` стал `/sql-ssb`. Убрать `--output-root` или указать `--output-root build/sql-ssb`; `sudo` не нужен |
| Не найден `ssb_q11` / `ssb_reference` | Собран только CLI или выбран другой build-каталог. Выполнить `cmake --build build -j4` для своего `--build`; default проверяет SQL и manual, а SQL-only всё равно требует CPU oracle |
| Ошибка каталога `/tmp/opencode` | Создать доступный своему пользователю каталог; не менять права чужих файлов |

Не используйте старый [`test_db_cli.sh`](../test_db_cli.sh) как основной quick
runner: он меняет cwd на `build`, пишет только логи и не сравнивает результаты.
Предпочитайте команды раздела 6. Сохраняйте `test/pcuda_*` в репозитории: среди
них исходники действующих CMake targets, runners и независимых oracles.
Сборки, PCH, локальные конфиги и результаты прогонов храните в игнорируемом `build/`;
постоянный кеш скомпилированных SQL `.so` инструкцией не вводится.

<a id="verified"></a>

## 10. Проверка команд

### Native datasets — 2026-11-15–16

Сборка/запуск после перехода проверены на AMD Radeon 780M: Release **20/20 CTest**,
CPU-only **18/18**, host-only preparation и тесты публичного скрипта, малый набор,
SF1 и SF10, все 13 SQL/manual с независимыми Text-oracles, широкая GPU-регрессия
и plain/traced Text-сессии. Все 58 колонок новых наборов сверены с исходными `.tbl`.
Подробные scope, hashes, raw, quality gates и ограничения —
[один основной отчёт интеграции](../test/ssb_native_data/REPORT.md).
Это не performance-серия и не закрытие этапа 8.

### Историческая проверка инструкции до перехода формата — 2026-11-15

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
