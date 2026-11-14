# Этап 1: структура SQL отдельно от параметров

## 1. Статус и границы

**Этап 1 выполнен 2026-11-04.** Реализован и подключён к `db_cli` параметризованный
frontend. Это **не перенос GPU backend**: выполнение и генерация ядер остаются
SYCL; этапы 2–8 не объявляются реализованными.

По явному решению пользователя применяется временная граница:

```text
SQL / AST → разрешённые ссылки → QueryTemplate + ParameterValues
          → оптимизация без значений → bindLegacyPlan → прежний SYCL codegen
```

Ключ нового логического шаблона не зависит от обычных значений. Старый кеш
**скомпилированных библиотек** намеренно остаётся зависимым от SQL/значений:
legacy codegen ещё специализирует CASE-домены, LIMIT и алгоритмы. Подставлять
`QueryTemplate::cacheKey()` вместо старого ключа сейчас небезопасно. Отсутствие
codegen/acpp/link при смене значений — последующая интеграция этапов 5–6, не
достижение этого этапа. Модульный ABI и политика «8 typed + fallback» здесь не
реализованы и не переутверждены.

Сохранены исходные пользовательские правки PCH, сборки, конфигурации и исследований.
**Commit/push не выполнялись по решению пользователя.** Блокирующих задач для
критерия этапа 1 не осталось; ограничения legacy backend перечислены ниже.

## 2. Окружение и конфигурация

- Ветка `dev`, исходный HEAD `ac5e7b9c086b2dfde8f793e4a151dd3541f8fde8`.
- AdaptiveCpp `25.10.0+git.5989e1cf.20260821.branch.HEAD`, Clang/LLVM 17.0.0,
  LLVM revision `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`, LLD 17.0.0.
- AMD Radeon 780M / HIP, SF10: 59 986 217 fact-строк. NVIDIA/Intel не проверялись.
- C++17; основной host и CPU-тесты — Release (`-O3`), тесты дополнительно
  `-UNDEBUG`. Флаги и механизм старого SYCL JIT/PCH оставлены пользовательскими.
- Отдельные диагностические host-сборки: LLVM coverage `-O0 -g
  -fprofile-instr-generate -fcoverage-mapping`; ASan/UBSan `-O1 -g
  -fsanitize=address,undefined`. Это не изменение Release-конфигурации движка.
- CMake 4.4.5, Doxygen 1.18.0. Новые зависимости не устанавливались.

Основной каталог артефактов —
[`run-7fnzlcXC`](../../build/pcuda-stage1/run-7fnzlcXC/):
[версии и ldd](../../build/pcuda-stage1/run-7fnzlcXC/versions.json),
[окружение](../../build/pcuda-stage1/run-7fnzlcXC/environment.json),
[SHA-256 исходников](../../build/pcuda-stage1/run-7fnzlcXC/source_sha256-final.json),
[SHA-256 бинарников](../../build/pcuda-stage1/run-7fnzlcXC/binary_sha256-final.json),
[CMakeCache](../../build/pcuda-stage1/run-7fnzlcXC/engine/CMakeCache.txt),
[полные команды сборки](../../build/pcuda-stage1/run-7fnzlcXC/engine/compile_commands.json).
Каждая CLI-сессия хранит `stdin.sql`, конфигурацию, environment, stdout/stderr,
native wall/user/system, исходные/effective argv компилятора и собственный PCH.
Диагностические сессии дополнительно сохраняют исходные Clang/LLD traces.
Дополнительно сохранены [GPU inventory](../../build/pcuda-stage1/run-7fnzlcXC/device-info.txt),
[hashes toolchain/runtime](../../build/pcuda-stage1/run-7fnzlcXC/toolchain_sha256.json),
[PCH/report hashes всех 14 сессий](../../build/pcuda-stage1/run-7fnzlcXC/session-artifact-sha256.json),
[effective acpp dryrun](../../build/pcuda-stage1/run-7fnzlcXC/legacy-effective-dryrun.txt) и
[полные команды host-диагностики](../../build/pcuda-stage1/run-7fnzlcXC/validation-commands.json).
Сгенерированные `.cpp`/`.so` удерживаются в `generated/` каждой сессии;
`generated-manifest.json` содержит исходный путь и SHA-256. Это артефакты проверки,
не постоянный SQL-кеш. После удержания и проверки hashes удалены только собственные
660 + 240 временных файлов, когда ни один `db_cli` уже не работал.

### Модель и словарь

| Термин | Реализация и смысл |
| --- | --- |
| Шаблон | `QueryTemplate`: дерево операторов и упорядоченная схема, без значений |
| Слот | `ParameterRefExpr::slot`: идентичность параметра, сохраняемая при clone/pushdown |
| Значение | `ParameterValue`: Int64, Float64, typed NULL либо NULL-only; целое не проходит через double |
| Origin | Одна исходная AST-позиция литерала; нормализация simple CASE сохраняет её, равные независимые литералы — нет |
| Binding | Проверенный снимок значений одной операции; не меняет шаблон и соседний снимок |
| Ключ | Версионированные length-delimited токены структуры/схемы, а не SQL-текст, адрес AST или lossy hash |

Основные компоненты:

- [`query_template.h`](../../include/core/query_template.h),
  [`query_template.cpp`](../../src/core/query_template.cpp): типы, извлечение,
  ключи, проверки binding и временный адаптер; Doxygen описывает назначение и границы.
- [`expressions.h`](../../include/core/expressions.h): ссылки на параметры и
  provenance литералов; [`operators.h`](../../include/core/operators.h): LIMIT
  как выражение и `select_output_slots` для сохранения SELECT-идентичности.
- [`translator.cpp`](../../src/core/translator.cpp): разрешение колонок по каталогу,
  aliases, ORDER BY-позиций и HAVING до извлечения значений. Несуществующие и
  неоднозначные ссылки диагностируются на входе.
- [`optimizer_rules.cpp`](../../src/core/optimizer_rules.cpp): pushdown получает
  только структуру; исправлены parent-связи при вставке/удалении Filter.
- [`execution.cpp`](../../src/core/execution.cpp): оптимизирует параметризованное
  дерево, затем получает отдельную legacy-копию. В `visitor.cpp` заменены обращения
  к старому LIMIT на проверенный accessor; новый GPU-код не добавлялся.

## 3. Методика

Работа шла проверяемыми единицами: сначала падающая проверка, затем реализация,
проверка результата и отдельный рефакторинг. Сохранены первоначальные неуспешные
логи, включая ошибки конфигурации стенда и разборщика CLI.

1. Поведенческие CPU-тесты вызывают parser → prepare → optimizer → binding.
   Проверяются ключи и значения, независимые равные слоты, clone, изменённые
   привязки, NULL-контексты, LIMIT 0/ошибки и идентичность выходных ссылок.
2. Все 13 SSB выполняются по три раза в **одном** `db_cli` процессе. Сравнение —
   с ручными SF10-файлами, с multiplicity и нормализацией колонок, без требования
   порядка там, где SQL его не задаёт.
3. Manifest: все 62 `current` и 2 `current_heavy` файла, 66 SQL-операторов, затем
   19 целевых проверок в той же сессии. Memory guard включён, вывод ограничен 32
   строками. `future` не включён. Manifest-прогон — функциональный smoke, не
   независимый oracle каждого его результата.
4. Из 19 целевых проверок 18 имеют offline NumPy-oracle по исходным int32-колонкам
   SUPPLIER: фильтры, арифметика, CASE, HAVING, сортировка/позиции, LIMIT 0,
   Int64 выше 2^53, Float64 и повторные запросы. Одна — ожидаемый отказ legacy
   для прямого сравнения `column > NULL`, **не успешное SQL-выполнение**.
5. Пять запросов `test/nullable_data` проверяются на CPU через реальное подготовленное
   и bound-дерево, независимый тестовый интерпретатор и TSV. Это проверка frontend
   bindings, **не GPU-проверка null bitmaps**. Дополнительно использован SQLite.

### Измерения и кеши

Performance-приёмка в этап 1 не входит. Сохранённые интервалы — диагностические
метрики существующего CLI, а не новые SLA и не замер SQL→PCUDA. `summary` в смысле
итоговой performance-серии этапа 0 здесь не оценивался. Поля `total_query_ms`,
`compile_ms` и остальные не переименовывались; старый `compile_ms` включает
compile/link. Startup/data/PCH не включаются в пользовательскую задержку запроса.

Процесс и вызовы acpp наблюдаются нативным Bash `time`, `LC_ALL=C`, секунды в raw
таймере, миллисекунды в JSON-экспорте. Python организует запуск и анализирует
результаты **после** его завершения; Python-таймера/прокси внутри команды нет.
Тестовые observer-обёртки не меняют оптимизацию и не используют replay.

На процесс выделяется новый `ACPP_APPDB_DIR`, внутри сессии он сохраняется.
OS/HIP/driver caches не очищались. SQL-файлы изолируются случайными пробелами после
SELECT из-за старых общих `/tmp/query_*.{cpp,so}`; PCH резервируется через exclusive
create и сохраняется после процесса. Чужой PCH не перезаписывается runner-ом.

Есть отдельные traced/plain процессы с одинаковой последовательностью 19 checks;
это диагностические контроли корректности, не повторная performance-серия.
По одному процессу недостаточно для оценки overhead/медиан: влияние traces на
скорость не объявляется измеренным и из прежних результатов ничего не вычитается.
Пороги 600/200/10/9 мс и бюджет SQL→PCUDA ≤10 мс не изменены и здесь не проверялись.

## 4. Результаты

### Основные проверки

| Проверка | Результат | `summary`, мс | Артефакт |
| --- | --- | --- | --- |
| CTest Release | 5/5 executable, assert активны | Не измерялся | [ctest-storage-final.log](../../build/pcuda-stage1/run-7fnzlcXC/ctest-storage-final.log) |
| Новый frontend | 14 поведенческих групп, включая пять nullable SQL | Не измерялся | [лог](../../build/pcuda-stage1/run-7fnzlcXC/coverage/storage-tests.log) |
| ASan + UBSan | 14/14 групп, без ошибок/leaks | Не измерялся | [лог](../../build/pcuda-stage1/run-7fnzlcXC/sanitized-storage-tests.log) |
| SSB, итоговый бинарник | 36/39 с файлами; Q1.1 ×3 подтверждён CPU oracle | Не оценивался | [raw report](../../build/pcuda-stage1/run-7fnzlcXC/sql-baseline-f_plbv3l/report.json) |
| Manifest + параметры | 85/85 checks: 84 SQL-результата и 1 ожидаемый отказ | Не оценивался | [raw report](../../build/pcuda-stage1/run-7fnzlcXC/cli-plain-hbh165n0/report.json) |
| Целевые checks с traces | 19/19: 18 oracle + 1 ожидаемый отказ | Не оценивался | [raw report и traces](../../build/pcuda-stage1/run-7fnzlcXC/cli-traced-evkam5_3/report.json) |
| Те же checks без traces | 19/19 на том же hash бинарника | Не оценивался | [raw report](../../build/pcuda-stage1/run-7fnzlcXC/cli-plain-kd9ag_yx/report.json) |
| Python-инструменты | 10/10 unit tests | Неприменимо | [лог](../../build/pcuda-stage1/run-7fnzlcXC/python-verified-final-tests.log) |
| Doxygen | Генерация с `WARN_AS_ERROR=YES` успешна | Неприменимо | [лог](../../build/pcuda-stage1/run-7fnzlcXC/doxygen-final.log), [HTML](../../build/pcuda-stage1/run-7fnzlcXC/doxygen-final/html/index.html) |

### Аналоги инструментов ub-mode

`crapper`, `dryer`, `mutator` отсутствовали; пользователь разрешил доступные
аналоги без установки зависимостей. Python bindings libclang оказались несовместимы
с установленной библиотекой, поэтому анализ использует **реальный Clang AST JSON**.

- Новый `query_template.cpp`: 97,41% строк, 96,61% code regions, 92,86% branches,
  все 40 функций выполнены — [LLVM coverage](../../build/pcuda-stage1/run-7fnzlcXC/coverage/storage-report.txt).
- Аналог CRAP: `CC² × (1 − region_coverage)³ + CC`; это **region**, не statement
  coverage crapper. Все 42 AST-определения/инстанцирования нового компонента <10,
  максимум **9,061**. Четыре новых resolver-метода также <10, максимум **8**.
  Старые большие translator/visitor-функции этим утверждением не охвачены.
- Полные тела новых функций длиной ≥50 токенов сравнивались lexical similarity:
  пар с рангом ≥0,8 нет. Это ограниченный аналог dryer, не поиск всех дубликатов
  всего проекта. [Модель](../../build/pcuda-stage1/run-7fnzlcXC/quality-storage.json),
  [resolver](../../build/pcuda-stage1/run-7fnzlcXC/quality-resolver-storage.json).
- Первая мутационная серия: **11 killed, 1 survived, 0 build-failed**. Пропуск
  slot ID в ключе выявил недостаток тестов. После отдельного поведенческого теста
  перестроенных привязок: **12 killed, 0 survived, 0 build-failed**.
  [Историческая серия](../../build/pcuda-stage1/run-7fnzlcXC/mutations-v1/report.json),
  [итоговая серия](../../build/pcuda-stage1/run-7fnzlcXC/mutations-v2/report.json).

Проверка dependency-файла нового компонента не обнаружила SYCL headers:
[исходные зависимости](../../build/pcuda-stage1/run-7fnzlcXC/query-template.dependencies.d).

### Сохранённая история неуспешных проверок

- Первый manifest raw — **80/85**, `success:false`: четыре ошибки нового parser-а
  CLI на нулевом результате (`Columns: a | b`, затем placeholder `0`), один
  неподдерживаемый прямой NULL-предикат. Этот raw не исправлялся задним числом:
  [3mq965gt](../../build/pcuda-stage1/run-7fnzlcXC/cli-plain-3mq965gt/report.json).
  Исправлена тестовая утилита, добавлены unit tests; отказ backend учитывается
  отдельно и явно. Следующий прогон:
  [0_uwrnr3](../../build/pcuda-stage1/run-7fnzlcXC/cli-plain-0_uwrnr3/report.json).
- Все четыре SSB-серии сохранили исходный `success:false` из-за Q1.1, а не стали
  успешными после объяснения oracle: [первая](../../build/pcuda-stage1/run-7fnzlcXC/sql-baseline-k0qgpcmj/report.json),
  [вторая](../../build/pcuda-stage1/run-7fnzlcXC/sql-baseline-xp7rut29/report.json),
  [третья](../../build/pcuda-stage1/run-7fnzlcXC/sql-baseline-2go6ccc7/report.json),
  итоговая четвёртая — в таблице выше.
- Финальный clang-tidy нашёл лишний `std::move` для trivially-copyable Storage.
  После удаления [CPU frontend-бинарник побитно не изменился](../../build/pcuda-stage1/run-7fnzlcXC/trivial-storage-binary-equivalence.json),
  но hash ACPP-linked CLI изменился. Поэтому SSB, manifest, plain/traced controls,
  coverage и sanitizer checks повторены на окончательной сборке, а не приписаны
  ей по старым данным. В [итоговом clang-tidy](../../build/pcuda-stage1/run-7fnzlcXC/clang-tidy-final.log)
  нет user-code diagnostics; 26363 предупреждения из non-user headers подавлены
  стандартным фильтром инструмента, не изменениями исходников/компилятора.
- Падающие TDD-логи: [ORDER BY ordinal](../../build/pcuda-stage1/run-7fnzlcXC/tdd-reference-failure.log),
  [отсутствующий API](../../build/pcuda-stage1/run-7fnzlcXC/tdd-template-red-build-configured.log),
  [parent links](../../build/pcuda-stage1/run-7fnzlcXC/optimizer-parent-red.log),
  [duplicate alias](../../build/pcuda-stage1/run-7fnzlcXC/duplicate-alias-red.log),
  [nullable q4](../../build/pcuda-stage1/run-7fnzlcXC/nullable-fixture-red-diagnostic.log).
  Ошибки quoting BASE_PATH, первоначальный запуск новой CMake-цели до reconfigure
  и временного dump-helper без runtime/rpath также сохранены в каталоге артефактов.

## 5. Корректность и ограничения

**Подтверждено:**

- `a > 10` / `a > 15` имеют один template/schema key и разные значения;
  `a < 10` отличается. WHERE/HAVING/арифметика/CASE/LIMIT извлекаются без
  value-dependent folding. ORDER BY ordinal, направление, функции, колонки,
  SELECT-идентичность и topology ссылок на слоты остаются структурными.
- Равные независимые значения не объединяются. Simple CASE сохраняет один
  исходный базовый параметр через нормализацию; clone и pushdown сохраняют slot ID.
- Int64 `9007199254740993` не округляется; Float64 остаётся отдельной альтернативой.
  NULL выводится из однозначного числового контекста, иначе остаётся NULL-only.
  Untyped NULL при binding существующей схемы сохраняет её тип; смена числового
  типа требует новой схемы, а не тихого преобразования.
- Проверяются число/типы параметров и LIMIT: NULL, отрицательное значение и
  непредставимый размер не проходят legacy binding; LIMIT 0 допустим.
- Исправлены ORDER BY ordinal агрегатного SELECT, сравнение canonical qualifier
  с alias и parent-связи оптимизированного дерева. SELECT output mapping хранится
  явно; legacy materialization всё ещё group-first.

**Независимые oracle и обнаруженные расхождения:**

1. Q1.1: SQL даёт **4486422947100**, `11.txt` — **4474878955766**. Разность
   **11543991334** — строки `1994-01-01` включительно. [Новый offline NumPy oracle](../../build/pcuda-stage1/run-7fnzlcXC/q11-independent-oracle.json)
   это подтвердил. SQL и SF10-эталон не менялись.
2. Nullable CSV содержит `flag=1` у id **1, 2, 6, 8, 11**: **5**, не прежние 4.
   [SQLite 3.54.0 и прямой подсчёт CSV](../../build/pcuda-stage1/run-7fnzlcXC/nullable-independent-oracle.json)
   подтвердили исправление только строки q4 в `expected.tsv`. Первоначальная
   неуспешная проверка сохранена; это исправление ошибочного oracle, не GPU-результата.

**Не заявляется:** полный GPU NULL/3VL-паритет на nullable storage, аппаратная
проверка MHT/nonunique fixtures из `future`, PCUDA-примитивы, перенос ручных SSB
примеров, модульное кеширование и отсутствие device JIT. Они относятся к следующим
этапам. Прямой `column > NULL` и прежний float fast-filter всё ещё диагностируются
старым backend как неподдерживаемые; frontend-модель типов не обещает их исполнения.
Self joins отвергаются, а не склеиваются по одинаковым именам таблиц. Новые SQL-типы
и возможности `future` не добавлялись.

## 6. Выводы и решения

- **Факт:** frontend больше не передаёт обычные значения оптимизатору; данные
  исполнения отделены от структуры и сохраняются в точных типах.
- **Факт:** новый ключ различает не только операторы, но и исходную идентичность
  параметров/SELECT-выходов. Равенство текущих значений не меняет программу.
- **Согласовано пользователем:** адаптер к legacy backend; без commit/push;
  доступные аналоги трёх инструментов ub-mode. Существующие эксперименты не отменяются.
- **Ограничение:** новый ключ — ключ логического шаблона, **не окончательный ключ
  PCUDA-модуля**. ABI/toolchain/layout и bounded typed/fallback добавляются при
  реализации соответствующих этапов. Новый ABI здесь не фиксируется.
- **Неизмеренное:** ускорение SQL, выполнение бюджета ≤10 мс и GPU-performance
  эффект. Числа прототипа этапа 0 не переносятся на этот этап или полный SQL.

## 7. Воспроизведение, журнал и оставшаяся работа

Пример команд окружения этого прогона; пути GCC/data/toolchain нужно выбирать
для своего стенда, в новый frontend они не зашиты:

```bash
cmake -S . -B build/pcuda-stage1/reproduce/engine \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  '-DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1 -DBASE_PATH=\"/home/aidar/Проекты/practice/data/\"' \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DACPP_TARGETS=generic -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/pcuda-stage1/reproduce/engine -j4
ctest --test-dir build/pcuda-stage1/reproduce/engine --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test/pcuda_stage1 -p 'test_*.py' -v

python3 test/pcuda_stage0/sql_baseline.py \
  --db-cli build/pcuda-stage1/reproduce/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp --runs 3 \
  --output-root build/pcuda-stage1/reproduce
python3 test/pcuda_stage1/validate_cli.py \
  --db-cli build/pcuda-stage1/reproduce/engine/db_cli \
  --acpp-include /opt/AdaptiveCpp/include/AdaptiveCpp \
  --output-root build/pcuda-stage1/reproduce --include-heavy
# Отдельные процессы, не одновременно: --skip-manifest, затем --skip-manifest --traces.
python3 test/pcuda_stage1/mutate.py \
  --build build/pcuda-stage1/reproduce/engine \
  --output build/pcuda-stage1/reproduce/mutations \
  --cxx /opt/rocm_sdk_612/bin/clang++ --acpp-lib /opt/AdaptiveCpp/lib \
  --flag=--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
```

SSB runner возвращает nonzero при известном расхождении Q1.1 с файлом — это
ожидаемая сохранённая неуспешная проверка эталона, не повод скрывать её exit status.
Не запускать несколько legacy CLI одновременно: их глобальный PCH ещё не перенесён.

### Журнал действий

Подробный журнал действий и причин —
[`actions.jsonl`](../../build/pcuda-stage1/run-7fnzlcXC/actions.jsonl).

| Единица работы | Причина / проверенный результат |
| --- | --- |
| Изучение roadmap, AGENTS, PERF-COLD и пользовательского diff | Установлены границы этапа, сохранён [исходный snapshot](../../build/pcuda-stage1/run-7fnzlcXC/initial/) |
| Согласование адаптера, инструментов и Git | Не расширять этап 1 до backend/cache; не включать чужую работу в коммиты |
| TDD ссылок и параметров | Воспроизведены ошибки ORDER BY/aliases/parent; введены модель, provenance и ключи |
| Подключение legacy binding и LIMIT-expression | Сохранён рабочий CLI без опасного переиспользования value-specialized библиотеки |
| Release assertions и обновление старых codegen checks | Проверки теперь реально исполняются; устаревшие имена kernel/atomic/load приведены к фактической генерации |
| SSB, manifest, CPU/NumPy/SQLite oracle | Сохранены все неуспехи; исправлен только доказанно ошибочный nullable q4 oracle |
| Coverage, AST complexity, duplication, mutations, sanitizers | Новый компонент проверен; выживший mutant закрыт отдельным тестом |
| Doxygen, документация, hash/diff/link проверки | Статусы отражают только этап 1; [82 незатронутых пользовательских пути сохранены](../../build/pcuda-stage1/run-7fnzlcXC/user-preservation.json) |

Оставшаяся работа: этап 2 — явный физический план и утверждение ABI; этапы 3–4 —
PCUDA runtime/примитивы; этапы 5–6 — модульный кеш и runtime-зависимые алгоритмы;
этапы 7–8 — полный функциональный/GPU-паритет и performance-приёмка. Отсутствие
повторных codegen/compiler/link вызовов доказывается тогда счётчиками, не временем
этого адаптера. Численные цели, политика typed/fallback и необязательный replay
этапа 9 не изменены. Следующий этап автоматически не выполнялся.
