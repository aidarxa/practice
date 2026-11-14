# Этап 7: функциональный PCUDA-перенос и ручные SSB

## 1. Статус и границы

**Основной числовой объём этапа 7 выполнен 2026-11-08 на AMD Radeon 780M.**
Заключительная проверка после очистки зависимостей линковки также завершена.
Полноценный текст был выделен пользователем 2026-11-08 в
отдельное расширение **7-TEXT: на момент числовой приёмки только план**.
По новому поручению 2026-11-10 выполнены исследование, согласование и внедрение
настоящего Text. Отдельные новые проверки/ограничения находятся в **§8**;
числовая история ниже сохраняется, а не пересчитывается задним числом.
Этап 8 и окончательная приёмка миграции не закрываются этой работой.

Согласованное решение от 2026-11-08:

- словарная подмена текста числовыми кодами **не принимается**;
- исправить физическую схему DDATE и декодировать ASCII byte-флаги в Int32 0/1;
- до 7-TEXT явно отклонять SQL-обращения к тексту, включая `SELECT *`, а не
  вычислять по случайным байтам и нулевому хвосту. `COUNT(*)` не читает текст;
- сохранить прежние числовые city/nation/region/mfgr/category/brand-коды: они уже
  находятся в Int32-файлах и не являются новым словарным кодированием этой работы.

Контракт расширения утверждён пользователем 2026-11-10:
[todo.md](../../todo.md#stage7-text), [исследование](../pcuda_string_research/REPORT.md).
Реализация использует единый UTF-8 Text и [ABI v2](TEXT_ABI.md), не словарь и не
CPU SQL fallback. Это новое состояние, не результат исторической числовой приёмки.

Артефакты всей работы:
[`build/pcuda-stage7/run-omwou2yw`](../../build/pcuda-stage7/run-omwou2yw/).
Начальное состояние сохранено в `initial/`, `initial.diff`, `initial-status.txt`,
`initial-sha256.json`; журнал действий и причин — `actions.jsonl`. Чужие
незакоммиченные изменения, `debug/`, старые raw/traces не удалялись. Commit/push
не выполнялись.

### Идея реализации простыми словами

1. **Не угадывать формат по имени или размеру запроса.** Единая физическая схема
   описывает все 58 колонок. У DDATE действительно 17 полей: после daynuminyear
   идут monthnuminyear, weeknuminyear, sellingseason и четыре флага. Старый список
   пропускал два поля и сдвигал последующие имена.
2. **Читать каждую числовую строку полностью.** Int32 декодируется как little
   endian, ASCII `'0'`/`'1'` — как 0/1. Недостаточный/лишний размер, неверная ширина,
   недопустимый флаг и переполнение отвергаются. Текст не загружается в числовую
   GPU-колонку. Новый namespace статистики не использует старые raw/сдвинутые stats.
3. **Ручной эталон остаётся ручным.** Три scalar-примера используют fused
   scan/filter и block SUM. Остальные выполняют отдельные dimension PHT builds и
   одно fused fact probe/aggregate; у Q3.3 — три build-ядра. Общими стали владение
   ресурсами, безопасный запуск и фиксированные ручные рецепты, не SQL-генератор.
4. **Нулевой результат — не отсутствие группы.** Суммы signed Int64 и presence
   хранятся отдельно. PHT имеет occupancy и поддерживает ключ ноль. GPU
   count/scan/write формирует компактный результат с checked ёмкостью; повтор и
   смена значения ручного dimension-фильтра используют те же буферы.

| Файлы | Ответственность |
| --- | --- |
| `include/storage/{column,ssb_schema,column_loader}.h` | Физические encodings, полная SSB-схема, строгая числовая загрузка |
| `include/core/memory.h`, `src/core/translator.cpp` | Storage-метаданные каталога и ранний запрет текста, включая projection star, но не COUNT(*) |
| `src/db/database_instance.cpp`, `include/crystal/utils.h` | Общая схема каталога/loader, пропуск текстовых буферов, правильные flags и DDATE |
| `src/core/execution.cpp` | Encoding/width входят в identity каталога для кеша физического плана |
| `test/queries/example/q*.cpp`, `support/{query_spec,runner}.*` | 13 ручных PCUDA entry points и независимые от SQL PHT/fused algorithms |
| `CMakeLists.txt` | Сборка всех примеров; отдельный `checked_sizes` без SQL-зависимостей runtime/ручных примеров |
| `test/pcuda_stage7/` | Host/GPU/storage/CLI oracles, recipes общих ub-tools и этот единственный основной отчёт |

Старый raw-prefix helper перенесён **только** в исторический fixture
`test/pcuda_stage3/legacy_storage_view.h`. Его прежние проверки и история не
выдаются за новый storage-контракт. Production engine и примеры его не используют.
Исторический SYCL ABI-negative adapter также не удаляется: он не линкуется в
движок/примеры. SDK-функция CMake `add_sycl_to_target` с PCUDA flags — название
API AdaptiveCpp, не второй frontend/backend движка.

## 2. Окружение и конфигурация

- AMD Radeon 780M Graphics, runtime подтвердил FP64=1 и atomic64=1. NVIDIA/Intel
  аппаратно не проверялись.
- Установленный AdaptiveCpp из прежнего стенда
  `25.10.0+git.5989e1cf.20260821.branch.HEAD`; compiler/runtime не заменялись.
  `acpp --version` выводит версию нижнего Clang, а не отдельную версию AdaptiveCpp.
- Clang/LLVM и LLD 17.0.0, ROCm LLVM revision
  `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`; C++17.
- Host и заранее собранные manual/services — Release `-O3 -DNDEBUG`;
  тесты дополнительно `-UNDEBUG`. Динамические SQL-модули/PCH —
  `--acpp-targets=generic --acpp-pcuda -std=c++17 -O0 -g0 -fPIC`,
  `-Wno-narrowing -Wno-deprecated-declarations`; отдельная линковка
  `acpp ... -fuse-ld=lld -Wl,-O0 -shared`.
- На этом стенде явно выбран совместимый GCC headers/runtime:
  `--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1`.
  Машинные пути не добавлены в production-код; default data path теперь `data/`.
- CLI-серии: `LC_ALL=C`, `TMPDIR=/tmp/opencode`, `ACPP_ADAPTIVITY_LEVEL=1`,
  `ACPP_ALLOCATION_TRACKING=1`, отдельный `ACPP_APPDB_DIR` на процесс.

Исходные tool/env outputs — `environment/`. Хеши исходников, headers, SDK,
бинарников и PCH до финальной очистки link dependencies —
`environment/final-inventory.json` (**147 записей**). Полные команды:
`engine/CMakeCache.txt`, `engine/compile_commands.json`, `CMakeFiles/*/link.txt`,
сессионные `.compile.command` / `.link.command`; effective acpp/Clang/LLD dry-runs
сохранены в `module-inspection/0/`. Это offline-диагностика, не replay.

**Итог после очистки link graph:**
[`post-link/final-inventory.json`](../../build/pcuda-stage7/run-omwou2yw/post-link/final-inventory.json)
содержит **142 hashes** актуальных исходников/headers/tests, CMake, CLI/manual
бинарников, PCH/manifest и SF10 reference-файлов. SDK hashes/версии остаются в
предыдущем inventory; SDK не изменялся.

| Итоговый артефакт | SHA-256 |
| --- | --- |
| `engine/db_cli`, 1 514 336 B | `8ba3f3fc9c7c9f0cd7f2fe0312fd7ffebe9b474a0d56d1b0e1cca5480f339e5e` |
| `engine/ssb_q33`, 234 496 B | `ec54e548d4f1e3e9a4930849470f4ff8b8a78eb957c00fac8c7e78c5e53234e5` |
| Toolchain manifest | `2c9e50df366b3e37c902ae449dbb281a183a699918977e9e79cfda62706ffc2a` |

PCH не менялся: **4 682 364 B**, SHA-256
`485b21459e0d4eb158745f72687839de3acf3f2c009767e22c5303aea9f9a531`.
ABI v1 не менялся: физические ASCII-флаги декодируются в прежний Int32 buffer ABI;
новая storage-метаинформация — только host-каталог, не изменение layout ABI.

## 3. Методика

Сначала поведенческий red: старый lookup не находил `d_monthnuminyear`; затем
тест выявил проходящий `SELECT *` над текстом. Raw failures сохранены. Новые
host-тесты проверяют physical schema, точное Int32/ASCII-декодирование, ошибки
границ, копирование storage-метаданных и frontend diagnostics. GPU tiny-fixture
ручных алгоритмов использует независимые ожидаемые числа, zero/negative keys,
нулевой/отрицательный SUM, пустую/частичную плитку, повтор и смену селективности.

SF10-результаты manual сравниваются как мультимножества с существующими файлами:
это сохраняет multiplicity, но не требует порядка без ORDER BY. Каждый из
13 примеров запускается в отдельном процессе, по четыре исполнения. Oracle и
разбор stdout работают **после** процесса. Числовой SQL/NULL/MHT/cache паритет
дополнительно проверяется реальным QueryEngine и `db_cli`, не строками исходника.
Печатный manual `Query CPU-ready ms` охватывает execute/fetch после подготовки
Dataset/ManualQuery workspace, не его конструкцию и не SQL frontend; он не
используется как V2 summary или end-to-end SQL latency в этом отчёте.

Storage CLI: сначала шесть процессов по 17 запросов, три plain/traced пары с
ротацией plain→traced /traced→plain /plain→traced. Затем добавлены явные проверки
всех пяти отклоняемых manifest-форм: по 21 запросу plain/traced. Эти наборы не
переписывают друг друга. После разделения link dependencies заново выполнены
все manual/native/SSB/manifest проверки и два storage-процесса по 21 запросу.
Отказ текста считается **ожидаемым неподдерживаемым
случаем**, а не успешно выполненным текстовым SQL.

Timing model — `pcuda-modules-v1`. `summary = total_query_ms`: непрерывный C++
`steady_clock` от SQL в загруженной БД до CPU QueryResult; внутри frontend,
необходимые compile/link/load, allocations, execution и fetch. Загрузка данных,
проверка, форматирование и печать снаружи. Native Bash `time` сохраняет
wall/user/system процессов и compile/link команд; Python не таймер и не compiler
proxy. Для связи CLI-ошибки с её marker используется native Bash `exec ... 2>&1`.

Device cache новый на процесс и сохраняется внутри; SQL cache только process-local.
OS/HIP/driver caches не очищались. Частоты/температуры и desktop-нагрузка не
фиксировались. Это correctness/diagnostic работа, не приёмка производительности:
n=3 недостаточно для заявления ускорения. Первое execution/fetch может содержать
device JIT, его чистое время не выделено. Native wall многозапросного процесса —
не время до первого результата. Отдельные PCH preparation/startup→first result
здесь не измерялись; SQL→PCUDA ≤10 мс не проверялся чистым прямым интервалом.

## 4. Результаты

### 4.1. Функциональные и quality-проверки

| Проверка | Подтверждённый результат до заключительной перелинковки | Raw относительно каталога работы |
| --- | --- | --- |
| Release CTest / CPU-only | **16/16 /14/14**, assertions включены | `ctest-final.stdout`, `cpu-ctest.stdout` |
| Все ручные SSB | **52/52** результатов, 13×4 | `manual-v2/manual-plain-5fzx4haq/report.json` |
| Storage CLI, восемь процессов | **104** числовых результата +**40** ожидаемых отказов текста | `analysis.json`, `storage-series/`, `storage-v2/` |
| Совместимые значения/NULL | **32** ready-запроса без plan/module codegen, compiler/linker/load/symbols | `analysis.json` и исходные counters |
| Current/current_heavy manifest + targeted | **83/88**, пять явно неподдерживаемых text/star форм | `sql-regression/manifest-plain-0f6qh9p0/report.json` |
| SQL SSB 13×3 | **36/39 с файлами**, только прежний Q1.1×3 подтверждён CPU oracle | `sql-regression/ssb-plain-lbpwvgme/report.json`, `q11-reference.stdout` |
| Native numeric/NULL/runtime SQL | **202 попытки: 200 результатов +2 ожидаемых отказа**, включая 5 nullable CSV | `numeric-final.*`, `numeric-fixture/` |
| MHT/OR multiplicity, PHT, module reuse, ABI/memory errors | Успех | `mht-final.*`, `pht-final.*`, `modules-final.*` |
| Primitives /resource runtime | **17 323 905** assertions /21 sparse fixtures с повторами и async error | `primitives-final.*`, `runtime-final.*` |
| ASan/UBSan и LLVM host coverage | Успех; не device sanitizer/coverage | `ub/host-sanitized-v1/report.json`, `ub/host-coverage-v2/report.json` |
| Локальные ub-tools self-tests /runner tests | **25/25 /3/3** | `ub-selftests.stderr`, `runner-tests.stderr` |
| Targeted mutants | **13/13 host +8/8 GPU killed**, survivors/unresolved=0 | `ub/{host,gpu}-mutants-v1/report.json` |
| CRAP-region | **21** выбранное host definition/instantiation, max **8,297 <10** | `ub/*-crapper-v1/report.json`, `analysis.json` |
| Dryer | Whole-body cross-file rank ≥0,8 при ≥50 tokens не найден | `ub/dryer-v1/`, `ub/manual-dryer-v1/` |
| ELF/HCF/entries | **168 модулей /8 процессов**, один native kernel на `.so`, конфликтов identities нет | `module-inspection/`, `module-inspection-v2/` |

Coverage неполное: шесть из 21 выбранных definitions/instantiations покрыты не
полностью. Весь `visitor.cpp`, большой `translate`, `DatabaseInstance` и manual
device closures не объявляются покрытыми. CRAP — для выбранных host-компонентов,
не для всех device kernels; GPU-путь проверен аппаратно и targeted mutations.
Dryer — whole-function heuristic, не доказательство отсутствия всех partial clones.
Мутации выполняются только в независимых копиях после свежей baseline-сборки;
build failure/timeout не считаются kills. Готовые baseline/copies/logs сохранены.

### 4.2. Историческая диагностика до заключительной очистки link dependencies

Первый запрос каждого из шести процессов: COUNT/SUM/MIN/MAX/AVG по
`d_lastdayinweekfl`, DDATE SF10, 2557 строк. Это **не Q3.3**, не число стенда этапа
0 и не SQL SLA. CLI SHA-256 этой серии:
`165d1100629b70c2c9deed57c32ac54c9608233c072601a6f743c1710a85dd54`.
Все значения — **мс**, median [min;max], n=3 на режим.

| Режим | **summary** | Build elapsed | First execution | Fetch | Load |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | **539,655 [535,577;548,366]** | 156,690 [153,427;157,364] | 378,937 [378,428;388,305] | 1,421 [1,392;1,538] | 0,316 [0,314;0,486] |
| Traced | **544,167 [539,847;544,666]** | 161,148 [159,319;161,706] | 378,102 [376,770;379,980] | 1,445 [1,304;1,454] | 0,521 [0,319;0,559] |

| Режим | Compile Σ wall/user/system | Link Σ wall/user/system | **summary** |
| --- | ---: | ---: | ---: |
| Plain | 308,156 /232,949 /69,169 | 264,791 /201,751 /90,171 | **539,655** |
| Traced | 317,034 /248,759 /63,407 | 266,337 /195,327 /95,018 | **544,167** |

Каждая ячейка — самостоятельная медиана; суммы параллельных команд не elapsed,
стадии/медианы не складываются для восстановления summary. Наблюдаемая разница
traced−plain summary **+4,512 мс**, build **+4,458 мс** — не выделенное чистое время
overhead traces и не доказанный выигрыш/проигрыш. Ничего не вычиталось из raw.
Сохранены 63 Clang +63 LLD traces этой серии; ещё 21+21 — в `storage-v2/`.
Все samples/разброс — `analysis.json`. Ready-повторы фильтра имеют другую SQL-форму,
поэтому их времена не выдаются за warm-исполнение этого первого запроса.

### 4.3. Итог после отделения `checked_sizes`

Новые результаты находятся в **`post-link/`**, старые raw и diagnostic samples
выше сохранены. Последний link graph ручных примеров содержит только
`manual_ssb`, `pcuda_scan`, `pcuda_runtime`, `manual_ssb_spec`, `checked_sizes` и
SDK/system libraries. Архивов `physical_execution`, `query_values`, `pcuda_modules`
и SQL parser в нём нет; соответствующие symbols не попали в бинарники. CLI не
содержит legacy launch adapter. Проверены также `.o.d` зависимости без full
`sycl/sycl.hpp`. Доказательство — `post-link/dependency-audit/report.json`.

| Итоговая проверка | Результат | Raw относительно каталога работы |
| --- | --- | --- |
| Сборка /Release CTest /CPU-only | Успех /**16/16 /14/14** | `post-link/{build,ctest,cpu-build,cpu-ctest}.*` |
| Все 13 manual SSB, четыре исполнения | **52/52** совпадения с SF10 | `post-link/manual/manual-plain-blsrum10/report.json` |
| Manual tiny, изменение predicate/reuse | Успех; zero/negative keys, zero/negative sums, пустые/partial tiles, ошибки ввода | `post-link/manual-tiny.*` |
| Storage CLI plain/traced | **26** числовых результатов +**16** ожидаемых text-отказов; **8** ready без новой работы | `post-link/storage/` |
| SQL SSB 13×3 | **36/39 с файлами**, только прежний Q1.1×3; warm counters=0 | `post-link/sql/ssb-plain-n2_mez5g/report.json` |
| Manifest/current_heavy +targeted | **83/88**, те же пять text/star ограничений | `post-link/sql/manifest-plain-1omjyuqa/report.json` |
| Native numeric/nullable/runtime | **202 попытки: 200 результатов +2 ожидаемых отказа**, включая 5 nullable CSV | `post-link/numeric.*`, `post-link/numeric-fixture/` |
| MHT/OR/PHT/module/ABI/memory | Успех | `post-link/{mht,pht,modules}.*` |
| Primitive /resource runtime hardware | **17 323 905** assertions /21 sparse fixtures с повторами и async error | `post-link/{primitives,runtime}.*` |
| GPU mutation campaign после нового link graph | **8/8 killed**, survivors/unresolved=0; fresh baseline | `ub/gpu-mutants-v2/report.json` |
| Nullable CSV/TSV независимым SQLite | **5/5** | `nullable-oracle.json` |
| ELF/HCF/entries новых SQL-модулей | **42 модуля /2 процесса**, без конфликтов identities | `post-link/module-inspection/report.json` |

Совокупно новые storage-процессы этой работы дали **130 числовых результатов и
56 ожидаемых text-отказов**, из них **40** ready-запросов с нулевыми счётчиками
codegen/compiler/linker/load/symbols. Это **не 186 успешных SQL-вычислений**: отказы
выделены отдельно. Всего в трёх offline module-inspection наборах — **210 модулей
/10 process scopes**, Clang/LLD traces — по **105** файлов; финальный traced-процесс
добавил ещё 21+21, исторические 84+84 сохранены.

Host-код после разделения CMake target не менялся: его свежие LLVM/ASan/UBSan,
CRAP/dryer и 13 host mutations из §4.1 относятся к тем же source hashes.
Offline-проверка финальных statuses, counters и boundaries —
[`final-analysis.json`](../../build/pcuda-stage7/run-omwou2yw/final-analysis.json),
исполняемый анализ — `analyze-final.py`. Для финального бинарника не заявляется
новый performance/SLA результат: таблица §4.2 явно остаётся историей предыдущей
линковки, а не переписанным измерением.

## 5. Корректность и ограничения

**Историческая граница числовой приёмки 2026-11-08; новые Text-проверки — §8.**
Пять действующих manifest-форм тогда были отложены до настоящего текста, **не переназначены
в `future` и не объявлены успешными**:

- `proj_05_select_star_small_dimension`;
- `proj_join_02_star_customer`;
- `custom2` — числовые сравнения с физически текстовым `c_mktsegment`;
- `stress_01_wide_star_customer`;
- `stress_02_wide_star_customer_supplier`.

Для каждой тогда отдельно проверялся явный `7-TEXT` error до числового codegen и
отсутствие фиктивного результата. Последующий числовой запрос в той же сессии
проходит без новой компиляции. Следовательно, это не **88/88 полного manifest**.
Projection/star по настоящим числовым tiny-каталогам, late materialization,
count/scan/write, MHT размножение и OR duplicate suppression проходят native tests.

Q1.1 не подгонялся под файл. Независимый CPU oracle без project/device helpers:
SQL **4486422947100**, 1993-only/file/manual **4474878955766**, дополнительный
1994-01-01 **11543991334**, 59 986 217 строк. SQL raw сохраняет `success:false`.

Ручные примеры — nonnullable SSB с уникальными dimension keys. Они явно отвергают
null sidecars и duplicate keys, а не выдают PHT за MHT. Общая SQL NULL/3VL и
неуникальные joins проверяются отдельно реальным engine: пустые/all-NULL,
COUNT/SUM/MIN/MAX/AVG, validity, join/filter/group/sort, typed results, CASE/HAVING,
LIMIT 0, сортировка и границы ёмкостей. Новые строковые операции, общий унарный
оператор, новая overflow/division-by-zero политика и Float64 GROUP BY CASE не
добавлялись. Согласованная числовая семантика этапа 6 сохранена.

### Сохранённые неуспехи и исправления методики

- `storage-red-*` — отсутствующий DDATE monthnum; `storage-check-3.*` — projection
  star первоначально обходил text gate. Это настоящие исходные red tests.
- `build-v2.stderr` — ошибка самого нового теста: binding type находится в
  `argument.schema.type`, а не `argument.type`. Исправлен тест, не ABI.
- `ub/host-coverage-v1/` — LLVM обнаружил **15 mismatched functions**, когда три
  разных `main()` экспортировались из трёх executables. Диагностика не подавлялась:
  создан один combined executable и выполнена новая честная коллекция v2.
- `manual-regression.stderr` — parser ошибочно считал `Result:` числом; использован
  прежний проверенный parser независимых reference-файлов.
- `manual-final/.../report.json` сохраняет failure из-за чрезмерно строгого запрета
  любого stderr. Все его числа совпали, но warning о первом device JIT принимался
  за ошибку. Новый runner разрешает только конкретный advisory, сохраняет stderr
  и отвергает остальные diagnostics; результаты проверены новым процессом в v2.
- `dependency-audit/` сохранил неубедительный пустой список библиотек из-за locale.
  В `dependency-audit-v2/` используется `LC_ALL=C`, стабильный NEEDED tag и
  обязательная проверка непустого списка. Старый audit не используется как доказательство.
- Последующий настоящий link audit показал лишний `physical_execution` archive
  в транзитивных CMake dependencies manual runtime, хотя его SQL-symbols linker
  не подтягивал. Выделен `checked_sizes`; **новые** сборка/полная функциональная
  серия/GPU mutation baseline подтверждают отсутствие этой зависимости (§4.3).

## 6. Выводы и решения

Подтверждены native PCUDA-исполнение всех 13 независимых ручных SSB, корректный
числовой storage-контракт и регрессия существующих числовых SQL-путей. У движка
и примеров нет full `sycl/sycl.hpp` frontend dependencies; legacy adapter остаётся
только историческим negative fixture. Единственный production lower-runtime seam
`pcuda_compat.cpp` сохранён в прежних согласованных границах.

Отдельный `checked_sizes` устраняет архитектурную лишнюю зависимость runtime и
ручных примеров от архива SQL-планировщика. Это очистка link graph, не новый
GPU-алгоритм/ABI. Replay, фоновые tiering, persistent SQL module cache и другой
toolchain не внедрялись. Полный текст на ту дату оставался открытым; его последующее
внедрение описано в §8. Окончательная приёмка этапа 8 этой работой не закрывается.

## 7. Воспроизведение и оставшаяся работа

Из корня проекта, с новым каталогом `RUN`:

```bash
RUN="$PWD/build/pcuda-stage7/reproduce"
GCC=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake -S . -B "$RUN/engine" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC -DBASE_PATH=\\\"$PWD/data/\\\"" \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DCRYSTAL_PCUDA_COMPILER=/opt/AdaptiveCpp/bin/acpp \
  "-DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=$GCC" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUN/engine" -j4
ctest --test-dir "$RUN/engine" --output-on-failure
"$RUN/engine/test_manual_ssb_gpu"
python3 test/pcuda_stage7/validate.py manual --build "$RUN/engine" \
  --output-root "$RUN" --timing-model pcuda-modules-v1
python3 test/pcuda_stage7/validate.py storage --build "$RUN/engine" \
  --output-root "$RUN" --timing-model pcuda-modules-v1
python3 test/pcuda_stage7/validate.py storage --build "$RUN/engine" \
  --output-root "$RUN" --timing-model pcuda-modules-v1 --traces
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" \
  --mode ssb --runs 3 --output-root "$RUN"
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" \
  --mode manifest --runs 1 --output-root "$RUN"
```

Две последние команды честно возвращают failure для описанных Q1.1 и пяти text
форм; остальные ошибки игнорировать нельзя. Полные использованные команды,
ротация и native fixtures — `run-functional.sh`, `run-final-checks.sh`,
`run-post-link.sh`; offline aggregation — `analyze.py` и `analyze-final.py`.

Quality recipes: `test/pcuda_stage7/{host,gpu}-recipe.json`. Используются именно
общие `tools/ub/{coverage,crapper,dryer,mutator}.py`, не stage-specific аналоги;
полные argv и причины находятся в их `actions.jsonl` и `.command.json`. Для GPU
recipe требуются SDK PATH, `CXXFLAGS`, `CRYSTAL_PCUDA_FLAGS`, `AdaptiveCpp_DIR` из
окружения стенда. Не совмещать GPU mutation tests с другими GPU-сериями.

Doxygen-комментарии описывают реализацию в коде. Doxygen/HTML не генерировались.
На момент числовой приёмки оставалось согласовать и реализовать **7-TEXT**, затем отдельно выполнить
этап 8, согласование абсолютных SQL performance targets и прямой бюджет frontend.

### 7.1. Начало расширения 7-TEXT — 2026-11-10

**Исторический preflight до согласования/кода. Фактическое внедрение — §8.**
По новому поручению пользователя выполнены чтение плана/правил, архитектурный
аудит и самостоятельное исследование строк. Отдельный широкий отчёт необходим
для сравнения GPU/CPU-практик, семантики и физических представлений; он не
дублирует этот отчёт внедрения:
[Строки на GPU: устройство, риски и проектирование 7-TEXT](../pcuda_string_research/REPORT.md).

Новые артефакты:
[`build/pcuda-stage7-text/preflight-cwwf566k`](../../build/pcuda-stage7-text/preflight-cwwf566k/),
включая initial diff/status/hashes, 14 первичных источников с SHA-256,
независимый `profile_text.py`, `data-profile.json` и `actions.jsonl`.
Старые результаты/неуспехи и пользовательские незакоммиченные изменения сохранены.

Наблюдение: ненулевые bytes после первого NUL массово присутствуют в текстовых
SF10 files, одновременно есть значения на всю ширину без NUL и адреса с
завершающим пробелом. Проверены все 16 текстовых dimension files и sample
по 10 000 записей двух fact files, всего 4 492 785 записей колонок. Это проверка
формы данных, **не GPU-исполнение Text SQL** и не performance-результат.

Предложены единый UTF-8 Text с бинарным сравнением, bounded-first-NUL decoding
без rtrim, компактные bytes/offsets/validity и borrowed views, collision-safe
group/join и ABI v2. `custom2.sql` требует отдельного решения: его числа 1/2
сравниваются с физически текстовым market segment. Подробности/альтернативы —
в исследовательском отчёте. До согласования эти предложения не являются
контрактом движка. Нынешние text gates не сняты, C++ не изменён; новые SQL/GPU,
sanitizer и ub-quality campaigns пока не запускались. Чекбоксы реализации
7-TEXT остаются открытыми.

## 8. Внедрение 7-TEXT — 2026-11-10–11

### 8.1. Статус и границы

**Функциональное расширение 7-TEXT выполнено 2026-11-11 на AMD Radeon 780M.**
Финальная последовательная серия и независимая агрегация:
[`final-regression/series.json`](../../build/pcuda-stage7-text/validation-171sfux5/final-regression/series.json),
[`analysis.json`](../../build/pcuda-stage7-text/validation-171sfux5/analysis.json).

Реализован native PCUDA Text: projection/star и литералы, все шесть сравнений,
WHERE/NULL/3VL, Text CASE, Text/mixed GROUP BY, HAVING, ORDER BY/LIMIT,
существующие inner equijoins/MHT/OR, COUNT/MIN/MAX(Text). SUM/AVG(Text), неявное
Text/number/Boolean, logical NUL и invalid UTF-8 отклоняются. LIKE/regex, функции
строк, collation, новые join kinds и Float64 GROUP BY не добавлены.

Единица значения — валидный UTF-8 с unsigned-byte сравнением, без нормализации;
пробелы значимы, пустое отличается от NULL. FixedText читается по полному размеру,
декодируется до первого bounded NUL либо на всю ширину; хвост игнорируется,
nullable payload не интерпретируется. Лимит значения INT32_MAX bytes, суммарные
размеры checked 64-bit; memory guard не ослаблялся. Результат CPU — owning строки.

Основные новые артефакты:
[`validation-171sfux5`](../../build/pcuda-stage7-text/validation-171sfux5/).
Начальное состояние/хеши и `actions.jsonl` сохранены; старые результаты и чужие
изменения не отменялись. Doxygen-комментарии добавлены, HTML не генерировался.
Тематическое [приложение ABI v2](TEXT_ABI.md) необходимо для полного описания
layout, pointer/NULL и lifetime-контракта; это не дублирующий отчёт о прогонах.

Этап 8, абсолютные SQL cold/warm-цели, прямой бюджет SQL→PCUDA ≤10 мс и выбор
оптимального string layout по производительности остаются отдельными задачами.
Показатели прототипа 600/200/10/9 мс на Text не переносятся. NVIDIA/Intel не проверены.

### 8.2. Окружение и конфигурация

Стенд прежний: AMD Radeon 780M/gfx1103, AdaptiveCpp pinned
`5989e1cfa4e70d55c65e786a4c71acb124ce54ff`, Clang/LLVM/LLD 17, C++17.
Host/services — Release, тесты с `-UNDEBUG`; SQL modules/PCH — generic PCUDA,
`-O0 -g0 -fPIC`, отдельный `acpp -shared -fuse-ld=lld -Wl,-O0`, до четырёх цепочек.
Совместимые GCC headers задаются флагом стенда, не machine path в production-коде.

Команды CMake/версии/env/hashes находятся в `environment/`, `engine/CMakeCache.txt`,
`engine/compile_commands.json`, JIT manifest и module `.compile/.link.command`.
PCH — подготовленный build artifact; новый ABI и device headers входят в его
dependency/fingerprint tracking. Пересоздание PCH на старте CLI не добавлялось.

### 8.3. Идея реализации и методика

1. **Отделить значение от хранения.** Storage нормализует FixedText в compact
   bytes/offsets/validity. Narrow offsets выбираются при доказанном Int32 extent,
   иначе UInt64 offsets с неотрицательным Int64-range. GPU видит borrowed 16-byte
   pointer/length, не `std::string`, не NUL-terminated C string и не словарный ID.
2. **Не специализировать обычные значения.** Frontend разрешает Text по каталогу,
   выводит тип NULL и выделяет независимые слоты до optimizer. Runtime arena хранит
   parameter bytes; module source/key не содержат литерал или его длину.
3. **Исполнять SQL на GPU.** Join строит buckets с row IDs и проверяет полный
   предикат; OR подавляет повтор той же пары, не разные одинаковые строки. GROUP
   BY CAS-публикует immutable representative index и отдельно обновляет states.
   Sort — GPU permutation merge с full Text comparator; fetch — prebuilt GPU
   sizes/scan/gather и D2H. CPU/SQLite не вычисляют SQL вместо GPU.
4. **Не удерживать память всех предыдущих запросов.** Typed scratch lifetimes
   coalesce в явном плане; loop/branch — консервативный region. HWM-workspace
   принадлежит одному value-free плану, а не совпавшим именам буферов разных SQL.
   Повтор и совместимые параметры сохраняют workspace; другая структура освобождает
   прежние owners. План/module/entry cache остаётся process-local и не очищается.
5. **Однозначно выдать результат.** CPU получает owning Text и отдельную validity.
   CLI печатает JSON-quoted Text, сохраняет whitespace многострочных литералов и
   ищет `;` вне quotes/comments. ORDER BY разрешает expanded star positions/aliases.

| Компонент | Ответственность |
| --- | --- |
| `storage/text_column.*`, `core/text.h` | Input validation, normalization, view/full compare/hash |
| `core/module_abi_v2.h`, `module_contract_v2.h` | Версионированные C POD и fail-closed validation |
| `core/text_plan*.cpp` | Value-free typed physical lowering, join/group/sort и scratch intervals |
| `core/text_storage.cpp`, `pcuda_plan_runtime.cpp` | Runtime RAII owners, active workspace, parameter arena |
| `core/text_fetch.cpp`, `db/database_instance.cpp` | GPU materialization и owning typed CPU result/window |
| `cli/{text_format,sql_input}.h`, `terminal_app.cpp` | Точный вывод и framing, не SQL execution |

Red→green проверялся реальными frontend/native/CLI clients. Независимый SQLite
oracle задаёт BINARY comparison и явно согласованный NULL-last; сравнивает
упорядоченные строки либо мультимножества с multiplicity. Строгие Text/number
отказы проверяются отдельно: динамическая типизация SQLite не является их oracle.
SF10 oracles декодируют физические файлы независимо от C++ schema/helpers;
широкие результаты проверяются полным row count и каждой видимой ячейкой окна.
Не заявляется полная CPU-материализация 12 млн широких строк.

Таймеры — native Bash time и существующие C++ steady_clock границы
`pcuda-modules-v1`; ABI v2 не переименовывает timing model. Query summary — один
непрерывный `total_query_ms` после data-ready до CPU Text result. Форматирование,
печать, oracle и offline trace analysis снаружи. Native multiquery process wall
не называется startup→первый результат. Загрузка eager Text находится до summary.

Device cache новый на процесс и сохранён внутри сессии; OS/HIP/driver caches не
очищены, частоты/desktop load не фиксировались. Первое execution/fetch может
содержать device JIT, его чистое время не выделяется. Финальные plain/traced
процессы выполняются последовательно, без GPU mutation campaigns одновременно.

### 8.4. Подтверждённые проверки

| Проверка | Результат | Артефакт относительно `validation-171sfux5/` |
| --- | --- | --- |
| Full Release build / CTest / CPU-only CTest | Без warnings / **18/18 /16/16** | `gpu-build-v12.*`, `{gpu,cpu}-ctest-final.stdout` |
| Native Text/SQLite matrix | **131/131: 125 результатов +6 ожидаемых отказов**; narrow/wide offsets, типы/NULL/CASE/group/join/OR/sort/fallback | `final-regression/text-gpu/gpu-plain-cd3ptdes/report.json` |
| SF10 Text CLI, 3 plain/traced пары | **246/246: 216 результатов +30 ожидаемых отказов** | `final-regression/text-cli-*/`, `analysis.json` |
| Ready CLI /немедленные warm | **126 /48** запросов, все шесть generation/build/load/symbol counters нулевые | `analysis.json`, исходные transcripts/counters |
| Full current/current_heavy/targeted manifest | **88/88**, пять прежних Text/star форм исполняются | `final-regression/manifest/manifest-plain-qs_g_cih/report.json` |
| Независимый аудит этих пяти форм | Все полные counts + **17 559** видимых строк/всех ячеек | `manifest-text-audit-final/report.json` |
| SQL SSB 13×3 | **36/39** с файлами; только Q1.1×3 подтверждён CPU oracle | `final-regression/ssb/`, `q11-reference.stdout`, `analysis.json` |
| Независимые manual SSB 13×4 | **52/52** | `final-regression/manual/manual-plain-98jzeh15/report.json` |
| Native numeric/NULL/3VL/PHT/MHT/OR/fallback/resources | **8/8** real executable clients | `native-final-v2/report.json` |
| Host targeted mutations | **19/19 killed**, survivors/unresolved=0; предыдущие 17/17 сохранены | `ub/host-mutants-final/report.json`, `ub/host-mutants-v2/report.json` |
| GPU targeted mutations | **7/7 killed**, survivors/unresolved=0 | `ub/gpu-mutants-v1/report.json` |
| LLVM host coverage | Свежий проверенный export, не device coverage | `ub/coverage-v5/report.json` |
| CRAP-region | **38** выбранных definitions/instantiations, max **9,081 <10**; 12 покрыты частично | `ub/*-crapper-final/report.json`, `analysis.json` |
| ASan/UBSan /C11 ABI | Успех, без device sanitizer claim; v1 layout не изменён | `ub/sanitized-final/report.json`, `abi-c11-*`, `dependency-audit.json` |
| Whole-function dryer | Cross-file/header rank ≥0,8 при ≥50 tokens не найден | `ub/dryer-v1/report.json`, `ub/headers-dryer-final/report.json` |
| ub-tools self-tests /Text runner /старый runner | **25/25 /3/3 /5/5** | `ub-selftests.stderr`, `runner-tests-final.stderr`, `legacy-runner-tests-final.stderr` |
| Native modules /dependencies | **181** modules в двух процессах, по одному HCF/kernel на `.so`, entries v2; **633** generated sources без SYCL/whole-query entry | `module-inspection-final/`, `dependency-audit.json` |

17 559 — 2557 DDATE + три окна по 5000 + два результата custom2. Полные counts:
11 916 634 для customer/region stars и 2 397 138 для customer/supplier same-nation.
Custom2 использует согласованные BUILDING/FURNITURE; независимые signed totals:
1998 → **3 922 998 979 437**, 1997 → **6 642 543 098 411**.

Quality scopes не скрываются: measured host coverage/CRAP не являются device
coverage или покрытием всего visitor/DB/runtime/planner. GPU mutations — targeted
семантические проверки, не exhaustive mutation score всего движка. Baseline и
каждая мутация пересобраны в независимой копии; build failure/timeout не kill.
Wide-offset аппаратный fixture использует маленькие данные с большим declared
upper bound; это проверка UInt64 offset path, не выделение >2 GiB строки.

Дополнительно Text parameter присутствует рядом с четырьмя независимыми numeric
slots во всех 16 Int64/Float64 схемах. Первые восемь filter specializations и один
tagged fallback готовятся лениво; схемы после готового fallback не генерируют/
не строят/не загружают модуль. Первое lowering новой типовой схемы допустимо;
16 последующих повторов имеют нулевые все шесть counters. Четыре прочих
специализированных модуля первого плана не выдаются за лишние filter specializations.

### 8.4.1. Диагностическая plain/traced серия, не SLA

Первый запрос — **`SELECT * FROM supplier ORDER BY s_suppkey LIMIT 7`**, SF10,
готовый PCH, все Text data уже загружены. Затем **8 немедленных warm** и остальная
матрица в процессе. Порядок трёх пар: plain→traced, traced→plain, plain→traced.
Все значения — **мс**, median [min;max], n=3 cold/режим и 24 warm/режим.

| Режим | **summary** | Build elapsed | Load | First execution | Text CPU fetch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | **622,163 [613,259;631,010]** | 144,946 [144,631;145,901] | 0,364 [0,351;0,365] | 275,053 [266,791;280,796] | 200,373 [200,092;202,675] |
| Traced | **622,093 [613,585;622,979]** | 149,561 [144,597;153,581] | 0,554 [0,385;0,585] | 268,687 [264,621;269,812] | 201,829 [197,682;202,454] |

| Режим | Compile Σ wall/user/system | Link Σ wall/user/system | **summary** | Warm total | Warm execution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 299,784 /225,346 /71,425 | 267,167 /199,463 /89,740 | **622,163** | 3,246 [2,911;4,050] | 0,714 [0,611;0,791] |
| Traced | 306,118 /232,277 /72,172 | 268,545 /196,696 /96,915 | **622,093** | 3,268 [3,078;3,641] | 0,714 [0,688;0,794] |

Суммы native команд не elapsed, их нельзя прибавлять к build elapsed. Каждая
ячейка — самостоятельная медиана; стадии/медианы не складываются для получения
summary. Все samples, ranges компонентов и stdout/timers — `analysis.json` и
`final-regression/`. Сохранены **174 Clang +174 LLD traces**; effective команды —
`module-inspection-final/0/`. PCH preparation и startup→первый CPU-результат в этой
серии отдельно не измерены; native wall всего 41-query процесса — другая метрика.

Наблюдаемая разница traced−plain median summary **−0,070 мс**, build **+4,615 мс**.
Это не выделенный overhead и не статистическое доказательство отсутствия эффекта
traces: n=3, частоты/нижние caches не контролировались. Разность cold/warm не
выдаётся за чистое device JIT. **622 мс здесь не сравниваются с 600-мс целью
прототипа Q3.3**: другой SQL, алгоритм, layout, output и границы подготовки. Новых
SQL/Text SLA, ускорения layout или достижения SQL→PCUDA ≤10 мс не заявлено.

### 8.5. Сохранённые неуспехи и границы

- Первые host failures содержали старые ожидания Text rejection/ABI v1. Native
  fake compiler, C entries и активные tests обновлены на v2; v1 ABI header и его
  historical negative fixtures не переинтерпретируются.
- Матрица выявила неявный Text-as-Boolean, trimming табуляции внутри SQL literal и
  отсутствие expanded-star ORDER BY. Для них сохранены red logs и новые clients.
- `manifest-v1`/`v2` — 86/88 memory guard failures; после частичной очистки —
  87/88. Изолированный wide query проходил. Устранено ошибочное присваивание
  исторических HWM owners новому плану; guard не отключался и SQL не урезался.
- Реальная bounded numerical regression выявила попытку присвоить scalar union
  v2 в восьмибайтный v1 tagged payload. `tagged-v2-red-build.*` и JIT compiler
  diagnostics сохранены; теперь копируется активный точный числовой член.
- Первые варианты независимых tests ошибочно требовали HAVING aggregate вне SELECT
  и non-key numeric OR join, отсутствовавшие в прежнем SQL contract. Они не стали
  новыми SQL features; ограничения и исходные failures сохранены.
- Independent oracle первоначально ошибся шириной DDATE weekday и таблицей small
  star (это DDATE, не SUPPLIER). Исправлен oracle по физическим данным/SQL; прежние
  failures не превращены задним числом в successes. Первая медленная scalar
  searchsorted версия audit остановлена внешним deadline, не считается проверкой.
- Начальные CRAP violations сохранены; UTF-8/ABI/framing разделены по обязанностям.
  Coverage collection v4 отказалась публиковать export после изменения `.pyc`;
  recipe теперь задаёт реальные C++ clients, bytecode отключён в runners.
- Первый host-mutator campaign остановлен внешним 120s deadline; он не выдаётся
  за законченный. `host-mutants-v2` выполнен заново без такого ограничения.
- В native resource fixture ожидаемая injected async error проверяется отдельно,
  а не запрещается любой stderr. Обычные неожиданные diagnostics остаются failure.

Известный Q1.1 SQL/file mismatch из §5 не исправляется изменением SQL, эталона или
агрегации. Старые raw не переписываются; файл/manual и SQL имеют разные date bounds.

### 8.6. Выводы

Настоящие строки проходят GPU execution и typed CPU output без словаря/CPU
fallback; прежние быстрые числовые fused SSB-пути сохранены. Bounded cache,
stream/error/RAII ownership распространяются на v2. Выбор compact views обоснован
контрактом и источниками, **не доказан как самый быстрый layout на AMD**.
Новые compiler flags/replay/persistent cache и широкая SQL type system не добавлены.

### 8.7. Воспроизведение и оставшаяся работа

Для текущего кода используйте свежую сборку из §7 с актуальными headers/PCH;
старый storage runner §7 проверяет исторический text gate, не текущий Text.

```bash
cmake --build "$RUN/engine" -j4
ctest --test-dir "$RUN/engine" --output-on-failure
python3 test/pcuda_stage7_text/validate_gpu.py --build "$RUN/engine" --output-root "$RUN"
"$RUN/engine/test_pcuda_text_resources"
python3 test/pcuda_stage7_text/validate_cli.py --build "$RUN/engine" --output-root "$RUN"
python3 test/pcuda_stage7_text/validate_cli.py --build "$RUN/engine" --output-root "$RUN" --traces
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" --mode manifest --output-root "$RUN"
python3 test/pcuda_stage5/validate_cli.py --db-cli "$RUN/engine/db_cli" --mode ssb --runs 3 --output-root "$RUN"
python3 test/pcuda_stage7/validate.py manual --build "$RUN/engine" --output-root "$RUN" --timing-model pcuda-modules-v1
```

После manifest отдельно запустите `test/pcuda_stage7_text/validate_manifest.py`
с его `--report`, `--data` и новым `--output`. Quality recipes —
`test/pcuda_stage7_text/{host,gpu}-recipe.json`, исполняются общими `tools/ub/`.
Для GPU задайте SDK PATH/AdaptiveCpp_DIR, совместимые host/PCH flags и отдельный
device cache; не совмещайте кампанию с другой GPU-серией. Полные команды/outputs,
копии baseline/mutants и action logs сохранены в артефактах.

Остаются этап 8 и его согласование абсолютных SQL targets/прямого frontend budget,
перенос аппаратной проверки на другие устройства и отдельные обоснованные
performance-эксперименты. Закрытие 7-TEXT не поручает следующий этап автоматически.
