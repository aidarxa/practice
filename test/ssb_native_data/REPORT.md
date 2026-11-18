# Native SSB datasets: dbgen, колонковый импорт и выбираемый SF

## 1. Статус и границы

**Выполнено 2026-11-15–16 по согласованию пользователя:** исходный Text вместо девяти
категориальных кодов и выбираемый SF не только в генераторе, но и в SQL/manual runtime.
Один запуск `bash bench/prepare_ssb.sh` создаёт готовый SF10 в `data/ssb/sf10`.
Нужны Git, Bash, CMake и C/C++17 toolchain; GPU, AdaptiveCpp и SQL parser для
подготовки не нужны. При первом получении submodule нужна сеть.

- Новые `ssb_storage`/`ssb_importer` и `ssb_convert` — host-only, с отдельным
  `bench/loader/CMakeLists.txt`; основной Makefile loader теперь ведёт в native target.
- Один production schema: `include/storage/ssb_schema.h`. Все исходные строки
  сохранены как Text; `Wednesday` занимает полные 9 байт, brand — до 10 байт.
- Обязательный `dataset.manifest` (`CRYSTAL_SSB_NATIVE_1`) хранит реальное число
  строк каждой таблицы, SF, generator identity и точную физическую схему.
  Каталог/все размеры проверяются **до** создания GPU runtime/аллокаций базы.
- `data_dir` / `CRYSTAL_DATA_DIR` выбирают набор. Numeric/Text/validity loading и
  статистики используют именно его; fixed SF10 row-count macros удалены из runtime/manual.
- Все 13 SQL и независимых manual SSB используют native категории. Manual PHT
  хранит ссылки на dimension rows, grouping хеширует/сравнивает настоящие значения;
  ссылки на immutable representative rows **не являются словарём**.
- Существующие `data/10/`, numeric references и прежние raw/отчёты не перезаписаны.
  Native набор несовместим с прежним encoded loader; автоматического fallback нет.

Это **не performance-приёмка** и не Text speedup. ABI v2, default timing model,
stock acpp, optimizer levels и default-off overlap не заменены. Warm backlog,
replay, parallel S2 и jobs>4 не реализовывались. Этап 8/SQL SLA остаются открытыми.

## 2. Окружение и конфигурация

Исходная база: `b7d3f1609e7aa5456b4e1290de9d76c2055a1f6b`, ветка `pcuda`.
Исходное состояние и SHA-256 файлов —
[`initial.json`](../../build/ssb-native-data/run-884q4wbo/initial.json).
Submodule инициализирован на существующем gitlink
`ae1e254aa4d603d8ef1f44078e5abed011634b23`, без upgrade/force и изменения upstream.

| Компонент | Фактическая конфигурация |
| --- | --- |
| GPU | AMD Radeon 780M Graphics, `gfx1103`, HIP driver `60140093`; NVIDIA/Intel не проверены |
| AdaptiveCpp | 25.10.0+git.5989e1cf.20260821.branch.HEAD; HIP/OpenMP доступны, SQL выполнялся на GPU |
| Host engine / LLVM / LLD | Clang 17.0.0, ROCm LLVM `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`, LLD 17.0.0 |
| HIP | `6.1.40093-ea60ae9e7` из установленного ROCm SDK |
| Standalone importer / dbgen | Системные GCC/G++ 16.2.1, Release; GPU SDK не подключался |
| CMake | 4.4.5; root Release и CPU-only — отдельные build-каталоги |
| SQL kernels | C++17, generic PCUDA, `-O0 -g0`; host Release, assertions тестов активны (`-UNDEBUG`) |

Сырые версии — [`environment/`](../../build/ssb-native-data/run-884q4wbo/environment/),
устройство — [`inventory-device/stdout`](../../build/ssb-native-data/run-884q4wbo/inventory-device/stdout).
`acpp --version` выводит версию нижележащего Clang; версия AdaptiveCpp проверена
по установленному `AdaptiveCpp/common/config.hpp`, а не выведена из этого вывода.

Исходная root-конфигурация полностью записана в
[`release-configure/command.json`](../../build/ssb-native-data/run-884q4wbo/release-configure/command.json):
`CMAKE_BUILD_TYPE=Release`, `ACPP_TARGETS=generic`,
`CMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++`,
`AdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp`,
`CRYSTAL_PCUDA_COMPILER=/opt/AdaptiveCpp/bin/acpp`,
host/PCUDA `--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1`,
link flags `-fuse-ld=lld;-Wl,-O0`. Это параметры стенда, не машинный хардкод исходников.
Effective host команды — `release/compile_commands.json`, PCUDA/PCH —
`release/pcuda-jit/toolchain.manifest` и сохранённые query artifacts.
SHA-256 source/PCH/binary/manifest и финальный diff находятся в
[`audit/`](../../build/ssb-native-data/run-884q4wbo/audit/).

## 3. Методика и контракт импорта

### Подготовка и публикация

Каждый запуск получает приватные tool-build/source-table каталоги. Скрипт не
меняет и не force-checkout'ит существующий изменённый submodule. Фиксируются
`WORKLOAD=SSB`, pipe output, trailing delimiter, даты без дефисов; `-b` указывает
на pinned `dists.dss`, а `DSS_PATH` принудительно направлен в приватный каталог.
Выбранный SF передаётся upstream, row counts **считаются при чтении**, не вычисляются
по формуле. Nonintegral SF ≥1 отклонён: upstream усекал бы его до целого.

Конвертер держит ограниченную строку и буферы записи по колонкам, а для dimension
PK — множество встреченных Int32 ключей. Память под fact-таблицу целиком не нужна;
проверка уникальности требует O(dimension keys), не O(fact rows). Проверяются полное
число полей, strict Int32/overflow, ASCII flags, длины, UTF-8/NUL и dimension PK.
Text не нормализуется, пробелы не отрезаются, никакие числовые коды не создаются.

Сначала создаются все колонки и manifest в приватном sibling staging. После проверки
публикуется целая директория через Linux `renameat2(RENAME_NOREPLACE)`. Ни обычный
rename поверх пустой директории, ни `--force` не применяются. На обработанной ошибке
RAII удаляет только собственный staging; существующий output не меняется. Это atomic
visibility/no-overwrite, **не** гарантия crash durability при отключении питания.
При аварийном завершении процесса может остаться приватный незавершённый staging.

По умолчанию generated `.tbl` удаляются, `--keep-tbl` сохраняет их для независимой
проверки. Вход `--from-tbl` никогда не очищается. Числовой stats-cache — отдельный
optional snapshot внутри выбранного набора, с file extent/mtime invalidation и
atomic replacement; read-only набор работает без сохранения кеша. Менеджер владеет
копией dataset identity, а не dangling reference на временный manifest object.

### Исполнение и измерительные границы

Это correctness-серии: SQL Q1.1→Q4.3 в одном процессе, immediate repeats внутри него;
manual — отдельный процесс на query, несколько trials внутри. Новые process/module
и device-cache каталоги, device-cache сохраняется внутри сессии. OS/HIP/driver caches
не очищались. CPU oracles/анализ выполняются после завершения GPU-клиентов.

Native Bash `time` сохранён для процессов; engine intervals — существующий
`steady_clock`. `total_query_ms`/summary остаётся прямым data-ready→CPU-result
интервалом, не суммой стадий. Whole-session native wall **не** является временем
до первого результата. Hashing, oracle, печать и анализ не добавлялись в summary.
Clang/LLD traces сохранены в отдельной traced Text-сессии с теми же 41 checks;
plain/traced correctness одинаков, latency overhead здесь не оценивался/не вычитался.
Во время функциональной работы выполнялись и host build/quality jobs: её timings
не предназначены для performance-сравнения. Медианы/SLA из них не заявляются.

## 4. Результаты

Новые наборы, без записи в прежний `data/10/`:

| SF | LINEORDER | SUPPLIER | CUSTOMER | PART | DDATE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 60 176 | 20 | 300 | 2 000 | 25 |
| 1 | 6 001 173 | 2 000 | 30 000 | 200 000 | 2 557 |
| 10 | 59 986 217 | 20 000 | 300 000 | 800 000 | 2 557 |

Fractional dbgen масштабирует также DDATE; SF0.01 — лишь smoke fixture, не обычный
полный SSB calendar. Рекомендуемый benchmark scale — целый SF ≥1. Очень большой
SF ограничивается Int32 source keys, диском и памятью runtime; произвольный масштаб
на Radeon 780M не обещан.

| Проверка | Результат | Raw/основной артефакт |
| --- | --- | --- |
| Root Release CTest | **20/20** | `post-dry-release-ctest/` |
| CPU-only CTest | **18/18** | `post-dry-cpu-ctest/` |
| Public entry/client tests | **8/8**, включая реальную генерацию, гонку двух importers, protected `DSS_PATH`, early manifest rejection | `public-entry-final/` |
| SF0.01 SQL/manual, по два trials | **26/26 SQL +26/26 manual результатов** | `validation-small/native-4z7u1b68/report.json` |
| SF1 SQL/manual + manifest/targeted | **111/111 SQL checks +26 manual результатов**; тогда ещё без трёх позднее добавленных Top-K cases | `validation-sf1/native-hbmdtgan/report.json` |
| SF10 initial native suite | **111/111 SQL checks +26 manual результатов** | `validation-sf10/native-9n8eehlo/report.json` |
| Финальный SF10 после typed-parser DRY | **114/114 SQL** = 26 SSB +66 current/heavy statements +22 targeted; **26/26 manual** | `pinned-native-results/native-48de7wfn/report.json` |
| Полная GPU-регрессия | **8/8 hardware clients**, Text GPU **131/131**, Text CLI plain/traced **41/41 +41/41**; native child **127/127 SQL +39 manual результатов** | `final-functional/functional-6xf4w_4u/report.json` |
| Пять широких Text/star manifest форм | **5/5**: независимый полный count и каждый видимый payload | `sf10-wide-text-oracle-results/report.json` |
| Установка и Q3.3 из installed CLI | Успешно, installed PCH/manifest; payload сравнен с CPU oracle | `native-install/`, `installed-native-q33/` |

Все пути таблицы относительны
[`build/ssb-native-data/run-884q4wbo/`](../../build/ssb-native-data/run-884q4wbo/).
Ранние native sessions не экспортировали binary hash в собственном record;
финальная pinned-серия сохраняет его для SQL и всех manual процессов.
Итоговая сводка и exact paths — [`audit/validation.json`](../../build/ssb-native-data/run-884q4wbo/audit/validation.json).
Не все manifest payloads имеют независимый oracle: 66 statement checks означают
исполнение, а не полную SQL-correctness всех возвращённых ячеек. Targeted/SSB/Text
fixtures и пять wide forms имеют отдельные независимые проверки.

## 5. Корректность и quality gates

`oracle.cpp` не включает project/GPU/SQL/converter headers. Он независимо описывает
physical widths/LE/flags, потоково вычисляет все 13 SSB и при `--tables` проверяет
**каждый байт всех 58 колонок** против исходных dbgen полей, включая zero padding.
Это выполнено для SF0.01/SF1/SF10. Dimension keys независимо проверены на уникальность.
Manual выдаёт typed JSON arrays, SQL parser сохраняет строки/точные integers/NULL;
multiset сравнение не теряет повторы. Numeric-first mixed Text rows также проверены.

Q1.1 не менялся семантически: на новом SF10 `sql.q11=[[4486422947100]]`,
`manual.q11=[[4474878955766]]`; SQL дополнительно включает `1994-01-01`.
Обе формы сверены отдельно, а исторические file mismatches не стали успешными
задним числом. Старые numeric result files не использованы как Text oracle.

Использованы существующие `tools/ub`, не их stage-specific копии:

| Объект | Измеренный gate |
| --- | --- |
| `ssb_dataset.cpp` / `ssb_import.cpp` | 26 definitions, max CRAP-region **9.012 <10** |
| Dataset-scoped stats | 7 definitions, max **9.0 <10** |
| Manual host specification / Filter | 6 +4 definitions, max **6.009 / 5.116 <10** |
| Typed numeric parser | Int32/UInt64 instantiations: **100% regions**, CRAP **3 /3** |
| Whole-function cross-file dryer | **45 definitions**, нет пар rank ≥0.8 при ≥50 tokens |
| Targeted semantic mutants | **12/12 killed**, 0 survivors, 0 build failures/timeouts/unresolved |
| Самопроверка общего инструментария | **25/25** |

Финальные артефакты: `quality-post-dry-coverage/`, `quality-post-dry-crapper-*`,
`quality-post-dry-dryer/`, `quality-post-dry-mutants/`, `ub-self-tests/`.
Это выбранные host-компоненты, не whole-engine/device coverage. Не все регионы
покрыты: некоторые I/O failure paths, platform bounds и Filter branches не исполнялись;
constexpr helpers `supplier`/`customer` измерены как **0% runtime regions** при
compile-time подготовке рецептов. Низкий CRAP не назван 100% покрытием. Manual
dataset/kernel path проверен аппаратно и oracles, но его device coverage не измерено.

История не переписана: API initially red из-за отсутствующего header, typed-row
parser red и duplicate-PK red сохранены. Первый coverage export был **отклонён**
из-за LLVM multi-main mismatch; исправлен test harness с одним executable, не
отключена диагностика. Один AST вызов с общим `--tu` не видел `.cpp` definitions —
перезапущены отдельные полные scopes. Dryer обнаружил rank 0.935 у двух integer
parsers; они объединены в типизированный helper, затем baseline/coverage/12 mutants
и gates пересобраны. Ошибки initial build/target selection и все предыдущие raw
остались в собственных каталогах; не объявлены успешными задним числом.

## 6. Выводы и решения

Подтверждено: один preparation entry point работает без GPU, datasets реально
переключаются одним и тем же `db_cli` binary, Text нигде не подменён category IDs,
все 13 SQL/manual совпадают с независимыми native oracles. Compatible immediate
SQL repeats имеют нулевые engine codegen/compiler/link/load/symbol counters.
Это не доказывает отсутствие внутреннего device JIT AdaptiveCpp.

Сохранены границы: nonnullable/unique-PK manual input; strict Int32 raw numeric
domain; no-overwrite, mandatory native manifest; старый encoded формат — только
история. Новая latency/memory performance цель не выбрана. Не утверждаются Text
speedup, полные SQL SLA, hardware parity NVIDIA/Intel или закрытие этапа 8.

## 7. Воспроизведение и оставшаяся работа

Сначала выполните [быстрый старт](../../docs/BUILD_AND_RUN.md#quick-start).
Для нового набора в отдельном каталоге после полной GPU-сборки:

```bash
bash bench/prepare_ssb.sh --scale 1 --output data/ssb/reproduce-sf1 --keep-tbl
CRYSTAL_DATA_DIR=data/ssb/reproduce-sf1 ./build/pcuda-release/db_cli < test/queries/sql/q33.sql
python3 test/ssb_native_data/validate.py --build build/pcuda-release --data data/ssb/reproduce-sf1 \
  --manifest --timing-model pcuda-modules-v1
```

Каталог `data/ssb/reproduce-sf1` должен быть новым. Для проверки исходных файлов
добавьте `--tables` с напечатанным `prepare-ID/tables`. Standalone без SQL/GPU deps:

```bash
cmake -S bench/loader -B build/ssb-loader -DCMAKE_BUILD_TYPE=Release
cmake --build build/ssb-loader -j4
ctest --test-dir build/ssb-loader --output-on-failure
```

Для quality используйте `test/ssb_native_data/host-recipe.json` с
`tools/ub/coverage.py`/`mutator.py run`, затем `crapper.py` с настоящим coverage и
cross-file `dryer.py`; точные argv, compiler flags и TU choices сохранены в
[`actions.jsonl`](../../build/ssb-native-data/run-884q4wbo/actions.jsonl) и command JSON.
Каждый output должен быть новым. Никакие HTML/Doxygen outputs не генерировались.

Основной согласованный функциональный объём закрыт. Отдельными будущими решениями
остаются performance-приёмка native Text, поведение очень больших SF и переносимость
на другие устройства. Новые SQL-возможности, словарь, replay или default-on overlap
автоматически не поручены. Commit/push и очистка чужих worktrees/артефактов не выполнялись.

### Уточнение quick start — 2026-11-17

Пользователь воспроизвёл `PermissionError: '/sql-ssb'`: пустая shell-переменная
`RUN` превратила `"$RUN/sql-ssb"` в абсолютный путь у корня файловой системы.
В той редакции руководства (§6.1) исправили команду, убрав `RUN`, и явно указали
запуск из корня репозитория. Использовался уже существующий default output root
`build/ssb-native-validation`; Python runner, loader и GPU-код не менялись.
Синтаксис и shell argv проверены с незаданными переменными, без нового GPU-прогона.
Журнал и diff: [`quickstart-docs-1q_c2kb3/`](../../build/ssb-native-data/quickstart-docs-1q_c2kb3/).

### Preflight неполной сборки — 2026-11-17

Вторая пользовательская ошибка выявила недостаток runner: после SQL-сессии он
пытался запустить отсутствующий `build/ssb_q11`. В выбранном `build/` был `db_cli`,
но отсутствовали manual targets и `bench/loader/ssb_reference`; предыдущая краткая
команда ответа не включала обязательную полную сборку.

Добавлена проверка всех выбранных executable clients, native manifest и, при
`--tables`, исходных файлов **до** создания output и запуска SQL/GPU. Для неполной
сборки — exit 2, полный список недостающего и команда `cmake --build … -j4`, без
traceback/автосборки/молчаливого пропуска manual. `--sql-only` исключает только
manual clients: `db_cli` и независимый CPU oracle обязательны в обоих режимах.

Восемь новых availability/preflight тестов сначала воспроизвели ошибочный порядок,
затем прошли; вся Python client suite — 16 tests, 3 прежних optional integration
cases пропущены без включающих env flags. Реальная неполная корневая сборка теперь
отклоняется до SQL и без создания output. На существующей полной GPU-сборке малый
native набор снова прошёл **26/26 SQL +26/26 manual** с независимым CPU oracle.
C++ engine, генерация, измерительные интервалы и прежние raw не менялись; новой
performance-приёмки нет. Команды, red/green и smoke:
[`validator-preflight-vudcrlgh/`](../../build/ssb-native-data/validator-preflight-vudcrlgh/).

<a id="quick-start-verification"></a>

### Переработка руководства — 2026-11-18

По запросу пользователя [быстрый старт](../../docs/BUILD_AND_RUN.md#quick-start)
перенесён в начало руководства и README: одна конфигурация, одна полная сборка,
однократная подготовка данных и SQL. Разработческие проверки/диагностика вынесены
после пользовательского сценария. Для обычного запуска не задаются переменные
Crystal/AdaptiveCpp или config overrides; остаётся установленный SDK и его
обычное системное окружение. Код, CMake и старые наборы не менялись.

На стенде из раздела 2 проверены:

| Проверка | Результат |
| --- | --- |
| Конфигурация без явного GCC | Прошла, но полная сборка с Clang 17 и GCC 16 headers упала; нерабочая команда не вошла в быстрый старт |
| Конфигурация и полная сборка `build/pcuda-release` | Прошли в новом каталоге с GCC 14.3.1 для host и PCH/JIT; старый `build/` не реконфигурировался |
| `install-deps.sh` | Свежая загрузка и сборка зависимостей в отдельном каталоге; основные шаги запускались с существующим `deps/` |
| Подготовка данных | SF0.01 успешно сгенерирован в новом каталоге; для проверок SF10 переиспользован, не пересоздавался |
| Q3.3 прямым запуском, без runtime overrides | 24 строки, сверены с независимым CPU oracle нового прогона |
| Native SSB validator без `--data`, `--trials` и `--output-root` | `SQL 26/26; manual 13/13; success=True`, у каждого manual по два результата; default output в `build/ssb-native-validation/` |

Команды, окружение, исходный failure, stdout/stderr и сводка сохранены в
[`build/docs-build-run/quickstart-orllh0ja/`](../../build/docs-build-run/quickstart-orllh0ja/).
Широкая регрессия и performance-серии не повторялись. Историческая проверка
руководства до native Text сохранена отдельно:
[`validation-with-debug.json`](../../build/docs-build-run/run-v9qyoz3y/validation-with-debug.json).
Там Release/CPU-only CTest **19/19 и 17/17**, Debug compile, CLI/reuse/install,
24 SQL file matches +2 прежних Q1.1, отдельно проверенных CPU, manual **13/13**
и 26 CSV-исполнений без сравнения payload. Эти результаты не подменяют новые.
