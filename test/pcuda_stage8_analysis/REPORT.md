# 8-ANALYSIS: холодный SQL-путь, JIT и варианты сокращения задержки

## 1. Статус и границы

**Аналитическое продолжение этапа 8, выделено по просьбе пользователя 2026-11-12.**
Функциональная приёмка и исходная подтверждающая серия остаются в
[отчёте этапа 8](../pcuda_stage8/REPORT.md). Сюда перенесены разбор cold
Q1.1/Q2.1/Q4.2 и отложенный warm-backlog, добавлена оценка следующих вариантов.
**Согласованный объём S2→overlap завершён 2026-11-12.** Это не завершение этапа 8.
Overlap реализован как **default-off исследовательский путь**, а не как
автоматически принятая замена штатного исполнения.

Выполнено: read-only анализ 13 планов, восстановление HIP kernel tracing,
разделение наблюдаемых cold-задержек, экспорт 12 фактически скомпилированных
модулей и повторная сверка выводов PERF-COLD/JIT-TIER с доступными raw.
Новые измерения, правки движка/AdaptiveCpp или смена уровней оптимизации
при выделении этого отчёта **не выполнялись**.

**Решение пользователя 2026-11-12:** warm пока не большая проблема; недостатки
сохранить, оптимизацию отложить. Текущий предмет — cold. Обсуждение вариантов
не разрешает автоматически возобновить закрытое исследование JIT-TIER,
внедрить фон/tiering, поднять лимит acpp jobs или заменить runtime.
**Последующее решение пользователя «Давай пойдем по твоему плану»:** разрешены
изолированные прямые S2-наблюдения Q1.1/Q2.1/Q4.2 с matched stock-контролем,
затем проверка overlap fact S1 с готовыми dimension kernels при четырёх цепочках.
Generic build services, parallel S2/runtime redesign и jobs>4 не выбраны.
SQL-цели не утверждены;
replay остаётся необязательным этапом 9, сейчас не выполняется.

**Короткий вывод:** прямой S2 observer измерил transformation Q1.1/Q2.1/Q4.2:
**346,324 /563,228 /749,154 мс** в instrumented plain-серии. Это не весь device-JIT
путь и не GPU execution. У Q4.2 dimension transformation занимает **460,481 мс**:
окно для перекрытия fact S1 действительно есть. Smoke подтвердил пересечение
S1/S2 по абсолютным clock intervals. Подтверждающая observer-free серия уменьшила
Q4.2 **summary 1082,910→934,905 мс**; все семь plain matched pairs быстрее.
Для Q1.1/Q2.1 устойчивого выигрыша не заявляется. S2 optimizer levels/passes,
установленный runtime, ABI и SQL-алгоритмы не заменялись; default off сохранён.

## 2. Окружение, конфигурация и источники

- AMD Radeon 780M, SF10, C++17, host Release; AdaptiveCpp commit
  `5989e1cfa4e70d55c65e786a4c71acb124ce54ff`, LLVM/Clang/LLD 17.0.0,
  ROCm SDK 6.1.2. Другие GPU аппаратно не проверялись.
- SQL modules: `--acpp-targets=generic --acpp-pcuda -std=c++17 -O0 -g0 -fPIC`,
  готовый совместимый PCH; link через `acpp -shared -fuse-ld=lld -Wl,-O0`.
  `jit_jobs=4`, `LC_ALL=C`, `ACPP_ADAPTIVITY_LEVEL=1`,
  `ACPP_ALLOCATION_TRACKING=1`. Полные include/GCC flags, исходные/effective argv,
  fingerprints PCH/toolchain и hashes — в разделе 2 отчёта этапа 8 и его raw.
- Общий корень артефактов анализа:
  [`build/pcuda-stage8/run-8xun1k4x/`](../../build/pcuda-stage8/run-8xun1k4x/).
  Относительные raw-пути ниже относятся к нему, если явно не указан другой корень.
  `optimization-analysis-q5akas53/` — планы/ISA; `rocprof-repair-x12zi8pl/` —
  изолированный profiler и 13 trace/control pairs;
  `cold-explanation-t2r1cp5p/` — offline-разложение и проверка дампа.
- При выделении 8-ANALYSIS исходный текст документов и matching AdaptiveCpp cache
  sources сохранены в `analysis-docs-4hyd324d/initial/` и `sources/`; там же hashes,
  проверка полноты переноса и журнал. Исходные raw/traces не перемещаются и не
  переписываются. Изолированный rocprof не заменяет системный wrapper или runtime.
- Исторические исследования:
  [PERF-COLD](../pcuda_compile_perf/REPORT.md), особенно разделы 2/4.9/6;
  [JIT-TIER](../pcuda_jit_tiering/REPORT.md), особенно разделы 4.1/4.2/4.4/6.2.
  Raw подтверждающей JIT-TIER серии доступен:
  [`q33-jgw97j6a/report.json`](../../build/pcuda-jit-tiering/q33-jgw97j6a/report.json).

Проверены `src/core/module_compiler.cpp`, `module_cache.cpp`, `execution.cpp`,
фактические `build.sh` и источники AdaptiveCpp из `/home/aidar/AdaptiveCpp`:
`include/hipSYCL/runtime/kernel_cache.hpp:312–350`,
`src/runtime/kernel_cache.cpp:599–608`. Commit совпадает с указанным выше;
источники runtime не изменялись.

Согласованное продолжение использует отдельный корень
[`build/pcuda-stage8-analysis/run-grgh_dvo/`](../../build/pcuda-stage8-analysis/run-grgh_dvo/).
`before-confirmation/inventory.json` содержит версии, env, hashes binary/runtime,
**184 проверенных manifest inputs** и 20 физических числовых колонок трёх SQL
(не hash всех eager Text/прочих колонок). `before-confirmation/sources/` сохраняет
724 текущих файла; исходное пользовательское дерево до правок — `initial/`, 707 файлов.

Первый overlap smoke использовал default JIT flags новой CMake-сборки без явного
GCC selection; внутри него barrier/overlap были сопоставимы. До подтверждения
профиль выровнен с исходным measured stock: явный
`--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1` передан отдельно в
`CRYSTAL_PCUDA_FLAGS`, не только host `CMAKE_CXX_FLAGS`. Старые binary/PCH/manifest
сохранены в `engine-before-toolchain-alignment/`, smoke/raw не пересчитаны.
`aligned-toolchain-check.json` доказывает точное совпадение compile/link/environment
профилей и **побайтно одинаковый PCH 4 736 312 B**, SHA-256
`737d42900695f7451e9ad5eff1f905637202cf6c60846feefac27d03c9faee67`.
Full hashes и фактические commands находятся в inventory, CMakeCache/compile DB и
session `.compile/.link.command`. PCH подготовлен при build, не в summary/startup;
его preparation time здесь заново не измерялся. Числа прежнего PCH experiment
не подставляются как новое измерение.

## 3. Методика и границы выводов

Разделы 4.1–4.3 — offline-анализ уже законченных серий. Процессы исходно измерены native Bash
`time`, внутренние интервалы — C++ `steady_clock`, GPU durations — rocprof
`BeginNs/EndNs/DurationNs`. Python используется только для готовых JSON.

`summary` — непрерывный интервал от передачи SQL в **уже загруженную БД** до
готового CPU-результата, без startup/data loading/oracle/печати. `build_elapsed_ms`
охватывает ожидание внешних compile/link chains; суммы native command wall не
равны elapsed. Первый execution включает runtime/device preparation, не только
работу GPU. Device JIT не вычисляется вычитанием warm из cold.

Подтверждающая SQL-серия: семь cold-процессов на запрос/режим и восемь warm в
каждом; новые module/appdb directories, сохранение внутри процесса, OS/HIP/driver
caches не очищались. Отдельная rocprof серия — один процесс на запрос/режим,
восемь warm, описательная диагностика, не новая SLA-серия. Полные методика,
ротация и trace overhead — разделы 3/4.9 отчёта этапа 8.

Во всех таблицах — **мс**, если явно не сказано иначе. Медианы компонентов и
вложенные интервалы не складываются в summary. Отдельные mean breakdown ниже
аддитивны только потому, что остаток рассчитан в каждом исходном sample.
Исторический Q3.3 prototype без SQL parse/plan/codegen не сравнивается с SQL
Q4.2 как одинаковая программа. Новые серии и их отдельные области описаны в
4.4–4.6; исходные варианты в разделе 6 не получают измеренный выигрыш автоматически.

## 4. Результаты и доказательства

### 4.1. Почему cold Q4.2 существенно дороже Q1.1 и Q2.1

Это offline-разбор **той же** подтверждающей серии без новых GPU-измерений.
Raw с распределениями, всеми семью samples и проверкой additive mean breakdown:
`cold-explanation-t2r1cp5p/cold-comparison.json`. Читаемый дамп фактически
скомпилированных модулей:
[`debug/ssb-cold-q11-q21-q42/`](../../debug/ssb-cold-q11-q21-q42/README.md).
Экспортированы **12 `.cpp`** (3+4+5), побайтно сверены с session 0/plain;
`manifest.json` хранит исходные пути и SHA-256. Нет новой генерации, изменения
ядра или переноса `.so` в постоянный SQL cache. Планы реконструированы read-only
и совпадают по identities; SQL, compile/link argv/timers и служебные sources
приложены отдельно.

Ниже **мс, медианы**; summary — непосредственный query interval в уже загруженной
БД, не полный старт процесса. Разброс summary приведён; все min/max остальных
полей и native wall/user/system — в исходной серии и offline JSON.

| SQL | **Cold summary [min;max]** | Build elapsed | Загрузка `.so` | Первое исполнение плана | Fetch | Новые `.so` / build waves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1.1 | **504,281 [498,073;515,264]** | 157,758 | 0,416 | 343,723 | 0,384 | 3 /1 |
| Q2.1 | **768,106 [765,455;780,811]** | 151,737 | 0,325 | 544,813 | 69,017 | 4 /1 |
| Q4.2 | **1090,962 [1085,539;1150,029]** | 307,982 | 0,634 | 712,595 | 69,148 | 5 /2 |

**Наблюдаемая разница summary:** Q4.2−Q1.1 = **586,681 мс**,
Q4.2−Q2.1 = **322,856 мс**. Разницы соответствующих медиан стадий:

- против Q1.1: build **+150,224**, первое исполнение **+368,872**,
  fetch **+68,764 мс**;
- против Q2.1: build **+156,245**, первое исполнение **+167,782**,
  fetch практически тот же (**+0,131 мс**).

Эти разницы медиан **не складываются в точную разницу summary**. Для проверки
разложения отдельно рассчитан остаток в каждом sample, затем средние; здесь
средние действительно аддитивны (округление — до 0,001 мс):

| Сравнение | **Разница средних summary** | Build | Первое исполнение | Fetch | `.so` load | Прочая подготовка/промежутки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4.2−Q1.1 | **596,994** | 152,073 | 374,008 | 69,942 | 0,216 | 0,755 |
| Q4.2−Q2.1 | **332,710** | 158,074 | 172,376 | 1,445 | 0,232 | 0,584 |

Это не новая серия, не замена медианных gates и не оценка доверительного
интервала. Остаток не называется чистым frontend, драйвером или device JIT.

#### Причина 1: пятый модуль попадает во вторую build wave

**Compile/link уже параллельны между модулями:** `jit_jobs=4` запускает до четырёх
независимых **процессных цепочек**. Внутри цепочки `acpp -c` заканчивается перед
`acpp -shared`; компиляция и линковка одного модуля не могут идти одновременно.
Линковки разных модулей могут перекрываться друг с другом и с другими компиляциями.
Это видно и в `src/core/module_compiler.cpp:177–204`, и в фактических `build.sh`.
Один вызов Clang не становится четырёхпоточным из-за `jit_jobs=4`; внутренние
потоки LLD — отдельная настройка, `--threads=N` в этих командах не задан.

Однако планировщик использует **жёсткий batch barrier**, не непрерывный пул:
он ждёт завершения всей четвёрки, затем начинает следующую группу. Фактический
порядок из сохранённых скриптов:

```text
Q1.1: [DenseCount | DenseWrite | Scan]                      → готово
Q2.1: [Build DDATE | Build SUPPLIER | Build PART | Scan]    → готово
Q4.2: [Build DDATE | Build PART | Build SUPPLIER | Build CUSTOMER]
                                                       → ждать ВСЕ четыре
                                                       → [Scan] → готово
```

Для Q4.2 медиана самой длинной compile→link chain первой группы **148,135 мс**,
отдельной chain Scan — **154,484 мс**; это интервалы команд, не две независимые
медианы, которыми заменяется direct build elapsed. Измеренное ожидание всей
сборки — **307,982 мс**. Q2.1 помещается в одну волну (**151,737 мс**), поэтому
добавление пятого модуля даёт почти ещё одну полную wave. Это конкретное
ограничение текущей реализации, **не неизбежная цена дополнительного JOIN**.
Улучшение порядка/pool при лимите четыре не обещает убрать всю вторую волну:
прежний longest-first counterfactual Q4.2 даёт лишь около **6,298 мс**, не 150 мс.
Повышение лимита/изменение scheduling без отдельного решения не выполнялось.

#### Причина 2: `.so` ещё не содержит готовый машинный код для GPU

Для `--acpp-targets=generic` есть **два этапа компиляции**:

```text
Generated C++ → acpp -O0 / .o → linker / .so
                 (первый этап, входит в build_elapsed_ms)
.so с host wrapper и промежуточным GPU IR → первый вызов PCUDA
                 → AdaptiveCpp device JIT для gfx1103 → GPU kernels
                 (внутри первого исполнения, либо fetch для его service kernel)
```

Второй этап тоже выполняется на **CPU**, хотя готовит код для GPU. `-O0` первого
acpp не отключает оптимизацию второго этапа. Установленный стек подробно разобран
в [JIT-TIER, раздел 2](../pcuda_jit_tiering/REPORT.md#22-что-находится-в-скомпилированном-файле):
наличие `.so` не означает завершение device compilation. `jit_jobs=4` управляет
внешними acpp/link chains, **не** четырьмя параллельными S2-компиляциями; внутренний
cache mutex AdaptiveCpp также описан в том исследовании. Новая S2-инструментация
или изменение runtime здесь не внедрялись.

Q2.1 — три dimension builds и fused scan/probe/aggregate; Q4.2 — **четыре** builds
и другой fused kernel: четыре JOIN вместо трёх, два fact-операнда для
`SUM(lo_revenue-lo_supplycost)`, три ключа группировки вместо двух. Это разные
программы/наборы kernels, а не одинаковый «запрос с JOIN». Q1.1 использует только
fact scan и scalar aggregate с вспомогательной материализацией.

При этом **warm Q4.2 быстрее Q2.1**: median total **40,320 против 53,692 мс**.
Поэтому приписывать cold-разницу более медленным GPU-вычислениям JOIN неверно.
Дополнительное подтверждение — отдельный n=1 rocprof diagnostic sample:

| SQL | **Diagnostic cold summary** | Первое исполнение плана, host interval | GPU durations kernels этого плана |
| --- | ---: | ---: | ---: |
| Q1.1 | **491,180** | 334,582 | 13,542 |
| Q2.1 | **771,492** | 549,153 | 53,721 |
| Q4.2 | **1094,993** | 712,736 | 40,267 |

Compaction при fetch здесь исключена из суммы kernels плана. Это отдельные
диагностические samples, не части семипроцессных медиан. Они подтверждают, что
сотни миллисекунд host-интервала первого исполнения **не являются временем
работы GPU kernels**. В него входят device JIT, создание/инициализация scratch,
submission, ожидания и другие runtime-операции. **Чистой доли S2 по этим SQL
мы пока не измерили**: всю разницу/остаток называть JIT нельзя.

#### Причина 3: первый fetch у групповых запросов включает служебное GPU-ядро

Q2.1/Q4.2 используют `compactSparseRows`, Q1.1 уже имеет columnar scalar result
и этим путём не идёт (`src/db/database_instance.cpp:293–325`). В первом fetch
могут находиться device JIT/подготовка общего service kernel, scratch и копии.
Это объясняет различие **границ работы**, но не разрешает объявить все 69 мс
чистым JIT или временем D2H. Сам compaction kernel на GPU занимает лишь
тысячные миллисекунды для этих двух запросов; после первого вызова fetch короткий.
Встроенная компактизация копирует фактические строки, не весь sparse domain.

**Вывод:** большая cold-разница реальна и локализована: внешняя build wave и
дорогое первое исполнение/device preparation; для сравнения с Q1.1 добавляется
первый service fetch. SQL→PCUDA и `dlopen/dlsym` не объясняют сотни миллисекунд.
Это объяснение текущего пути, **не доказательство оптимальности 1,09 с**.
Следующий предмет cold-диагностики — прямое разделение S2/прочего runtime внутри
первого исполнения и fetch; исправления warm пока не выполняются.

### 4.2. Что уже проверялось: почему повторный перебор O0/O1 не основной путь

Память пользователя верна: [JIT-TIER, разделы 4.1–4.2](../pcuda_jit_tiering/REPORT.md)
не подтвердил улучшения первого ответа относительно штатного пути. Но важно
разделять «часть компиляции удалось сократить» и «пользователь получил ответ раньше».
Ниже исторический **Q3.3 SF10 prototype**, семь cold/56 warm на вариант, plain;
это не замеры нового SQL-пути и не новые цели. Значения сверены с raw report.

| Вариант | **summary [min;max]** | S2 `fullTransformation`, сумма 4 kernels | Внутренний LLVM optimizer, внутри S2 | Warm execute |
| --- | ---: | ---: | ---: | ---: |
| stock | **468,364 [464,009;476,436]** | Не инструментирован | Не инструментирован | 8,079 |
| hiprtc33, контроль с инструментом | **505,914 [502,352;542,998]** | 298,161 | 17,682 | 8,055 |
| hiprtc00, два промежуточных O0 | **471,323 [463,476;508,595]** | 257,937 | 0,141 | 7,805 |
| clang00, другой завершающий compiler | **972,398 [966,191;1013,650]** | 572,137 | 0,136 | 143,085 |

В инструментированном контроле S2 уменьшился примерно на **40 мс**; но hiprtc00
не обогнал stock по summary. Накладные расходы инструмента из raw не вычитались.
У hiprtc00 COMGR всё ещё выполнял завершающую оптимизацию/сборку: это **не**
полностью O0 device compiler. У clang00 поменялся также маршрут получения
машинного кода; его ухудшение нельзя приписать только цифре оптимизации.
Внутренний LLVM optimizer в контроле занимал лишь часть S2. Парсинг/связывание IR,
обязательные преобразования, backend codegen, hipRTC/COMGR и загрузка остаются.
Вложенные интервалы не складываются; полные диапазоны/команды — в JIT-TIER raw.

PERF-COLD уже проверил PCH, параллельные acpp chains, linker/thread/ELF flags,
no-exceptions/RTTI/unwind и adaptivity variants. PCH/parallel4 используются;
многократного дополнительного выигрыша от остальных flags не подтверждено.
Replay помогал **внешней** сборке прототипа, не отменял S2; его ограничение этапом 9
не пересматривается этим обсуждением.

### 4.3. Два разных барьера исходной реализации

Номера строк ниже относятся к сохранённым исходникам до overlap-эксперимента.

1. `src/core/module_compiler.cpp:199–204`: после каждой четвёрки compile→link
   chains ожидается вся группа. При пяти близких по цене задачах и четырёх слотах
   хотя бы одна задача обязана ждать; простая перестановка не устраняет второй ход.
2. `src/core/module_cache.cpp:103–113`, `execution.cpp:402–428`: compiler возвращает
   **все** binaries, затем проверяются/публикуются все `.so`, и только после этого
   начинается выполнение плана. Готовые dimension kernels до этого не используются.

Первый барьер — лимит ресурсов/scheduling. Второй — API «весь набор готов».
**Компиляция fact-модуля не требует результатов компиляции dimension-модулей**:
каждый — отдельный TU/`.so`, общается через host ABI v2, межмодульных device calls
нет. Результаты dimension builds нужны fact-ядру **при выполнении**, а не acpp
при компиляции. Эти две зависимости нельзя путать.

Это архитектурное ограничение API подготовки, а не ошибка SQL-семантики.
Предложение его изменить требует предварительно оговорённого рефакторинга,
поэтапной готовности и сохранения транзакционной публикации cache/typed reservations.
На момент исходного разбора такой рефакторинг не выполнялся. Последующий opt-in
эксперимент ниже сохраняет отдельную транзакционную публикацию.

### 4.4. Прямое S2-наблюдение полного SQL-пути

Новая серия —
[`s2-series-2vnfhh8d`](../../build/pcuda-stage8-analysis/run-grgh_dvo/s2-series-2vnfhh8d/series.json),
offline validation —
[`s2-analysis/report.json`](../../build/pcuda-stage8-analysis/run-grgh_dvo/s2-analysis/report.json),
покомпиляторная атрибуция —
[`attribution.json`](../../build/pcuda-stage8-analysis/run-grgh_dvo/s2-analysis/attribution.json),
все дополнительные таблицы — `s2-analysis/tables.txt`.

**84 отдельных процесса:** 3 SQL ×7 сессий ×stock/probe ×plain/Clang+LLD traces,
в каждом cold +8 warm. Порядок SQL циклически сдвигается; trace/plain и stock/probe
чередуются. По 7 cold и 56 warm на каждую комбинацию. Полный сбор завершён до
oracle/statistics/trace analysis, без конкурентных builds/GPU jobs. Оба режима
запускают `db_cli` через один и тот же loader; только probe добавляет `--preload`.
Это не `LD_PRELOAD` в окружении: отдельный exec-fixture доказал отсутствие DSO и
переменной в дочернем executable. Compiler children не получают observer.

Сохраняются прежние `pcuda-modules-v1`, data-ready→CPU-ready summary и границы
startup/fetch. Process/module/appdb новые для каждого процесса; внутри warm
состояние сохраняется. OS/HIP/driver caches не очищены. Observer отдельно собран
host Release под matching LLVM17/AdaptiveCpp; установленные библиотеки не заменены.
Он вызывает оригинальные ELF функции, не меняя O3 S2, внешний opt или HIPRTC.
Для внешнего S2 opt применяется прежний Bash native timer из JIT-TIER; его
дополнительный exec/IO входит в instrumented summary и не вычитается.

Ранние smoke завершались abort/SIGSEGV; их stdout/stderr и GDB-диагностика
сохранены, не включены в успешную серию. `stock-elf-bindings/` подтвердил, что
stock HIPRTC provider — **`libamdhip64`**, не отдельный `libhiprtc`. Resolver
использует `RTLD_NOLOAD` для уже локально загруженных HIP symbols и вызывает
тот же оригинал, не смешивая opaque handles разных providers. Исправление
observer не заменяло установленные HIP/HSA библиотеки. Команды и причины —
`probe-local-symbol-red`, `probe-build-v3/v4`, `probe-gdb*`, `stock-elf-bindings`
в журнале артефактов; исходные неуспешные smoke сохранены отдельно.

Plain-медианы, **мс**; у summary дополнительно `[min;max]`:

| SQL / observer | **summary** | S1 build elapsed | Load `.so` | First execution | Fetch | Warm total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1.1 stock | **500,671 [499,267;503,758]** | 158,671 | 0,409 | 340,392 | 0,432 | 12,546 |
| Q1.1 probe | **513,183 [511,085;518,763]** | 157,877 | 0,285 | 353,391 | 0,379 | 12,705 |
| Q2.1 stock | **764,450 [761,050;775,612]** | 152,257 | 0,477 | 542,667 | 68,766 | 53,935 |
| Q2.1 probe | **777,967 [773,814;791,422]** | 150,891 | 0,489 | 554,366 | 71,159 | 54,013 |
| Q4.2 stock | **1097,638 [1083,646;1110,588]** | 308,297 | 0,562 | 716,548 | 68,557 | 40,457 |
| Q4.2 probe | **1111,295 [1099,470;1116,460]** | 310,961 | 0,648 | 726,872 | 71,214 | 40,493 |

Native суммы отдельных acpp, мс; это не build elapsed:

| SQL / observer | **summary** | Compile Σ wall/user/system | Link Σ wall/user/system |
| --- | ---: | ---: | ---: |
| Q1.1 stock | **500,671** | 239,483 /183,340 /52,973 | 198,282 /149,700 /68,188 |
| Q1.1 probe | **513,183** | 238,461 /186,157 /51,710 | 198,866 /151,906 /69,920 |
| Q2.1 stock | **764,450** | 324,601 /246,932 /72,374 | 265,759 /195,919 /96,532 |
| Q2.1 probe | **777,967** | 324,328 /249,477 /69,302 | 264,475 /201,006 /95,034 |
| Q4.2 stock | **1097,638** | 415,226 /324,268 /87,102 | 334,340 /245,850 /117,599 |
| Q4.2 probe | **1111,295** | 416,719 /315,744 /93,391 | 333,903 /242,289 /121,901 |

Observer увидел **210** полных transformations: 4/5/6 на процесс соответственно,
включая scan/compact services. Каждому соответствует optimizer, внешний opt и
HIPRTC Complete. Внутренние `prepareIR`/bitcode/translate hooks не достигаются
из-за inline/direct calls установленной сборки: их попытки сохранены в smoke
snapshot/audit, бесполезные wrappers удалены. Отсутствие события **не означает ноль**.

| SQL | **summary probe** | S2 full transformation | LLVM optimizer | External opt host span | HIPRTC Complete | Остальной S2 interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1.1 | **513,183** | 346,324 | 34,273 | 67,323 | 114,275 | 130,683 |
| Q2.1 | **777,967** | 563,228 | 71,452 | 96,168 | 226,768 | 170,329 |
| Q4.2 | **1111,295** | 749,154 | 103,111 | 126,651 | 314,565 | 206,426 |

Все дочерние интервалы **вложены** в full transformation. Последняя колонка
считается для каждого процесса как parent minus **union** наблюдаемых children,
затем берётся медиана; она не объявляется исключительно driver/IR parsing/linking.
Медианы колонок не складываются. HIP module-load host spans наблюдены отдельно:
1,738 /2,184 /4,179 мс; эти process observations не прибавляются к S2 parent как
часть измеренного им интервала. Runtime malloc/copy/fill/wait totals имеют scope
всего процесса до экспорта observer, включая loading/warm/teardown, **не query-only**.
Отдельных query/execution/fetch boundaries для этих HIP totals нет; они не
подменяют прямые SQL execution/fetch timers или прежние GPU kernel timestamps.

Каждый generated модуль сопоставлен по SHA-256 с независимым dump. Покомпиляторные
медианы full transformation, мс (общий summary указан для соответствующего SQL):

| SQL / **summary probe** | Модули / service: S2 full, мс |
| --- | --- |
| Q1.1 / **513,183** | Scan_LINEORDER 148,351; AggregateDenseCount 59,395; scanLevel 75,584; AggregateDenseWrite 63,765 |
| Q2.1 / **777,967** | Build DDATE 99,781; SUPPLIER 102,306; PART 113,710; fact 179,512; compactSparseRows 69,903 |
| Q4.2 / **1111,295** | Build DDATE 120,242; PART 123,433; SUPPLIER 112,500; CUSTOMER 103,019; fact 219,607; compactSparseRows 68,510 |

Сумма dimensions внутри каждого sample, затем медиана: **314,101 мс** Q2.1 и
**460,481 мс** Q4.2. S2 full не включает все cache/dispatch/load costs и не является
полным «чистым device JIT». Тем не менее это прямое наблюдение дорогостоящей
compiler transformation, а не прежнее вычитание cold−warm или GPU kernel sum.

**Цена диагностики:** probe−stock по разнице summary medians = +12,512 /+13,517 /
+13,657 мс. Медианы парных разностей = +12,294 /+15,810 /+15,127 мс; диапазоны
[7,468;18,092] /[5,454;21,545] /[−11,118;29,717]. Это наблюдаемое изменение вместе
с variability, не универсальная константа overhead. Traced−plain summary median
для stock = +0,305 /+1,651 /+3,635 мс, для probe = +3,597 /+7,988 /−0,683 мс.
Ни эти изменения, ни предполагаемый overhead не вычитаются из исходных результатов.
Все raw, native external-opt wall/user/system и **336 Clang/LLD traces** сохранены.

| SQL stock | **summary**, мс | Main startup | Data loading | Spawn→first CPU result [min;max] |
| --- | ---: | ---: | ---: | ---: |
| Q1.1 | **500,671** | 2347,809 | 5202,970 | 8056,525 [8033,248;8319,409] |
| Q2.1 | **764,450** | 2346,522 | 5198,476 | 8314,013 [8281,526;8338,016] |
| Q4.2 | **1097,638** | 2345,558 | 5212,446 | 8641,709 [8632,436;8744,892] |

Полный startup не равен сумме медиан и не входит в summary. PCH не пересобирался
на SQL/startup пути; его build-time подготовка и manifest остаются отдельными.

### 4.5. Opt-in overlap: контракт и предварительная проверка

`jit_overlap_experiment=on` / `CRYSTAL_JIT_OVERLAP_EXPERIMENT=on` выбирает только
исследовательский путь; по умолчанию **off**. `ModuleBuildSequence` владеет одним
конечным S1 worker. Он последовательно собирает волны максимум по четыре через
прежний `ICompiler::compile`, публикуя immutable readiness futures. Основной поток
остаётся единственным владельцем module cache/load/GPU, выполняет прежний план
без разбиения fused kernels и использует один ordered stream. Parallel S2 нет.

Параметры/источники/typed reservations выбираются до worker. Логические bindings
проверяются до выполнения плана, module ABI/capabilities/completion owner — до
соответствующего launch. **Все** binaries, включая неиспользованную ветку, должны
успешно собраться и пройти load/ABI до atomic cache publication. Ошибки compilation,
count/ABI, callback/execution и late-unlaunched modules не публикуют candidates и
не расходуют typed slots. Worker всегда joined; plan completion и module owners
дренируют GPU до unsafe free/unload. Retry повторяет подготовку, а готовые
values/typed NULL/warm не создают worker/build/load/symbol events.

S1 сборка, result preparation, lazy load и execution теперь перекрываются.
Поэтому используется отдельная **`pcuda-modules-overlap-exp-v1`**, не меняющая
default `pcuda-modules-v1` и старые raw/gates. Прямой summary прежний; compile
наблюдает весь worker span, execution включает readiness waits/lazy load, native
acpp суммы остаются отдельными. Обе component sums и residuals для experiment
равны **null**. Полный контракт — [timing.md](../../timing.md).

Первый smoke:
[`s2-series-kzd7kkyn`](../../build/pcuda-stage8-analysis/run-grgh_dvo/s2-series-kzd7kkyn/series.json),
[`clock audit`](../../build/pcuda-stage8-analysis/run-grgh_dvo/overlap-smoke-analysis-v2/report.json).
12 процессов, 24 результата, stock/probe ×barrier/overlap для всех трёх SQL,
по одному cold/комбинацию, один warm. **Не статистическая подтверждающая серия.**
Q4.2 фактически перекрыл **157,483 мс** второй S1 волны с full S2 intervals; для
одноволновых Q1.1/Q2.1 пересечение нулевое. Intersections считаются union по
целочисленным absolute steady-clock boundaries. Из одного smoke не утверждается
устойчивый выигрыш или SQL SLA; требовавшиеся matched n≥7 серии приведены ниже.

### 4.6. Подтверждение overlap — завершённая ограниченная серия

После выравнивания flags/PCH и повторной полной регрессии завершены
**112 независимых процессов, по cold +8 warm**: 84 без observer для трёх SQL
×barrier/overlap ×plain/traced ×7; ещё 28 Q4.2 probe процессов с теми же двумя
schedules/диагностическими режимами ×7. Query order циклически сдвигается,
порядок schedules и trace/plain меняется между сессиями. В каждом случае —
**7 cold и 56 warm**. Основная оценка summary — cohort **без observer**; probe
отдельный, в основную статистику не подмешивается. Обе коллекции закончены до
oracle/statistics/trace analysis, без конкурентных builds/GPU tests.

Process/module/appdb directories новые для каждого cold-процесса, внутри warm
состояние сохраняется; OS/HIP/driver caches не очищались. Таймеры — native Bash
`time` и C++ `steady_clock`; Python организует процессы и offline-анализ, не
подменяет таймер и не находится внутри измеряемой compile/launch цепочки.

- Основные raw:
  [`s2-series-z5959_rm`](../../build/pcuda-stage8-analysis/run-grgh_dvo/s2-series-z5959_rm/series.json),
  [проверка/распределения](../../build/pcuda-stage8-analysis/run-grgh_dvo/overlap-confirmation-analysis/report.json).
- Диагностические raw:
  [`s2-series-toffpkp7`](../../build/pcuda-stage8-analysis/run-grgh_dvo/s2-series-toffpkp7/series.json),
  [проверка/intersections](../../build/pcuda-stage8-analysis/run-grgh_dvo/overlap-probe-confirmation-analysis/report.json).
- Общие offline tables/paired differences:
  [`overlap-final/report.json`](../../build/pcuda-stage8-analysis/run-grgh_dvo/overlap-final/report.json),
  `overlap-final/tables.txt`, `overlap-final/invariants.json`.
- `after-confirmation-preservation.json`: все binary-affecting sources, binary,
  manifest/PCH/toolchain/runtime и 20 input columns **не менялись** во время серий.

Observer-free plain, медианы **мс**, summary `[min;max]`:

| SQL / schedule | **summary** | S1 build elapsed | `.so` load | Execution | Fetch | Warm total /execute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1.1 barrier | **491,351 [485,045;497,020]** | 152,475 | 0,400 | 336,355 | 0,363 | 12,514 /12,216 |
| Q1.1 overlap | **491,262 [487,931;500,017]** | 152,641 | 0,446 | 490,145 | 0,375 | 12,489 /12,232 |
| Q2.1 barrier | **753,728 [749,821;766,294]** | 146,277 | 0,359 | 537,212 | 67,842 | 53,868 /53,620 |
| Q2.1 overlap | **759,252 [752,325;763,779]** | 148,331 | 0,562 | 688,930 | 67,920 | 53,922 /53,657 |
| Q4.2 barrier | **1082,910 [1075,742;1141,171]** | 300,038 | 0,616 | 713,349 | 68,157 | 40,615 /40,207 |
| Q4.2 overlap | **934,905 [926,108;942,123]** | 302,920 | 0,767 | 864,945 | 68,461 | 40,558 /40,163 |

Execution overlap **включает ожидание первой S1 волны**, а barrier execution
начинается после всей сборки. Поэтому рост этого поля не означает замедления
GPU; складывать его с build/load нельзя. Прямой summary имеет одинаковые границы.

Native суммы отдельных acpp, мс; не восстановление parallel latency:

| SQL / schedule | **summary** | Compile Σ wall/user/system | Link Σ wall/user/system |
| --- | ---: | ---: | ---: |
| Q1.1 barrier | **491,351** | 228,782 /184,113 /45,310 | 195,570 /147,390 /67,679 |
| Q1.1 overlap | **491,262** | 228,348 /180,049 /45,513 | 194,096 /150,844 /63,093 |
| Q2.1 barrier | **753,728** | 313,594 /248,995 /63,246 | 256,179 /195,387 /88,538 |
| Q2.1 overlap | **759,252** | 314,932 /249,275 /61,429 | 257,134 /196,727 /88,930 |
| Q4.2 barrier | **1082,910** | 402,480 /325,578 /75,097 | 328,173 /245,543 /122,964 |
| Q4.2 overlap | **934,905** | 401,444 /323,926 /75,482 | 327,109 /242,342 /118,723 |

Для Q4.2 разница summary medians **−148,005 мс**; медиана **семи парных разностей**
overlap−barrier = **−151,606 мс [−201,414;−143,940]**. Все семь plain pairs лучше.
Это разные статистики, не суммирование медиан. Q1.1 — разница medians −0,089 мс,
парная median +0,207 [−3,099;5,848]; Q2.1 — +5,524 мс, парная median +5,818
[−11,319;9,350]. Для этих одноволновых запросов выигрыша не заявляем. Warm здесь
только regression control, не реализованная warm-оптимизация и не тест на
статистическую эквивалентность.

Matched Clang/LLD traced, **summary**, мс:

| SQL | **barrier summary [min;max]** | **overlap summary [min;max]** | Traced−plain median: barrier /overlap |
| --- | ---: | ---: | ---: |
| Q1.1 | **494,971 [490,035;503,422]** | **492,409 [489,237;511,923]** | +3,620 /+1,147 |
| Q2.1 | **760,228 [750,485;778,161]** | **758,611 [754,979;775,108]** | +6,500 /−0,641 |
| Q4.2 | **1082,839 [1078,763;1092,012]** | **937,932 [932,939;958,330]** | −0,071 /+3,027 |

Q4.2 traced тоже лучше во всех семи pairs: парная median −147,385 мс
[−154,080;−120,433]. Отрицательные/малые trace differences не объявляются
ускоряющим эффектом диагностики; overhead не вычитается из исторических raw.

Отдельный Q4.2 probe cohort, plain, медианы мс:

| Schedule | **summary [min;max]** | Build elapsed | S2 full transformation | Optimizer /external opt span /HIPRTC |
| --- | ---: | ---: | ---: | ---: |
| barrier | **1091,403 [1084,783;1122,157]** | 299,217 | 746,732 | 102,035 /128,481 /312,007 |
| overlap | **946,226 [937,413;949,224]** | 302,283 | 748,992 | 103,025 /127,775 /309,902 |

Наблюдаемое пересечение второй S1 волны с dimension full transformations —
**150,979 мс [147,496;153,609]** plain и **152,938 [151,002;156,088]** traced,
в каждом режиме n=7. Это union absolute intervals целого compiler-wave call,
включая file/timer work, **не** доказательство ровно стольких миллисекунд чистого
acpp CPU-work. Все 168 transformations диагностического cohort успешны.
Сам S2 total не стал меньше; native build тоже не ускорился. Подтверждённый
эффект — сокращение критического пути, а не уменьшение объёма компиляции.
Probe summary traced: barrier 1097,435 [1093,237;1125,109], overlap
946,747 [941,290;954,700]; оба остаются отдельными instrumented observations.

Полный старт observer-free plain, медианы мс, не сумма столбцов:

| SQL / schedule | **summary** | Main startup | Data loading | Spawn→first CPU result [min;max] |
| --- | ---: | ---: | ---: | ---: |
| Q1.1 barrier | **491,351** | 2304,856 | 5262,586 | 8059,157 [7998,442;8321,769] |
| Q1.1 overlap | **491,262** | 2302,637 | 5261,461 | 8050,041 [8010,962;8146,441] |
| Q2.1 barrier | **753,728** | 2300,842 | 5256,370 | 8308,906 [8281,927;8430,388] |
| Q2.1 overlap | **759,252** | 2300,540 | 5236,198 | 8291,054 [8279,237;8476,065] |
| Q4.2 barrier | **1082,910** | 2303,930 | 5268,896 | 8650,091 [8598,136;8982,828] |
| Q4.2 overlap | **934,905** | 2302,698 | 5253,780 | 8492,493 [8462,317;9066,108] |

Серия ограничена SF10/этим AMD стендом, незакреплёнными частотами/температурой,
семью процессами на case и указанными cache states. На других SQL/устройствах
или после произвольной смены плана latency/zero-allocation не обещаются.
Default-on adoption и абсолютные SQL SLA этим результатом **не утверждены**.

## 5. Корректность, перенос документов и ограничения

Исходные функциональные результаты, 1750 acceptance samples, 26 profiler/control
процессов/234 результата/585 GPU kernel events и известные Q1.1 mismatches
остаются в разделе 4 [основного отчёта](../pcuda_stage8/REPORT.md).
Файл Q1.1/SQL/oracle и прежние failure-статусы не менялись. Дамп 12 generated
sources сохранён без изменения содержимого; README теперь указывает сюда.

При переносе проверяются полный текст cold-разбора и warm-backlog, все строки
таблиц, ссылки, source/dump hashes и неизменность measured `db_cli`. Snapshot
до переноса хранится в `analysis-docs-4hyd324d/initial/`. Старые журналы/проверки
продолжают описывать свою дату, не переписываются под новый путь.

На момент выделения отчёта работа была только документальной: прежние прогоны
не выдаются за тесты новой оптимизации. Последующее согласованное продолжение
использует самостоятельный observer из этой папки, не переписывает JIT-TIER и
не меняет установленные compiler/HIP/HSA runtime. Новых SQL-функций, backend,
ABI или persistent SQL cache нет. Эффект overlap/generic builds/parallel S2
не считается измеренным только по наличию S2-инструментации.

Новая S2-серия независимо проверена после завершения всех процессов: **756 CPU
результатов**, **504 совпадения с файлами** и **252 прежних Q1.1 mismatch**,
каждый подтверждён свежим независимым `q11_reference.cpp` на 59 986 217 строках.
SQL = **4486422947100**, file/1993-only = **4474878955766**, 1994-01-01 =
**11543991334**. `success` offline-reader означает принятие независимых oracles,
не full file parity; per-record `file_matches:false` и число mismatches сохранены.
**672 warm** имеют нулевые codegen/compiler/link/load/symbol deltas; observer
не увидел дополнительных full transformations сверх 4/5/6 cold programs/process.

После реализации и после выравнивания PCH оба scheduling режима проходят один
и тот же полный набор независимых клиентов. Последний полный прогон —
[`default`](../../build/pcuda-stage8-analysis/run-grgh_dvo/regression-default-aligned/) и
[`overlap`](../../build/pcuda-stage8-analysis/run-grgh_dvo/regression-overlap-aligned/),
оба parent reports возвращают functional
success с явным `full_file_parity:false`; SSB child report остаётся `success:false`:

| Область | Default | Opt-in overlap |
| --- | ---: | ---: |
| Release CTest / CPU-only | 19/19 /17/17 | общий build, те же CPU checks |
| Manifest current/current_heavy + targeted | 88/88 | 88/88 |
| Independent manual SSB | 52/52 | 52/52 |
| Text GPU | 131/131 | 131/131 |
| Text CLI plain +traced | 41+41 | 41+41 |
| Все 13 SQL SSB ×3, file oracle | **36/39** | **36/39** |

Дополнительно реальные native SQL modules проверяют late S1 failure после первой
готовой волны и успешный retry, 16 типовых схем/8+1 fallback, typed NULL, структурное
переиспользование и zero-build повторы. Host gated compiler не может закончить
вторую волну до использования первой: partial readiness проверяется поведением,
а не sleep/timing heuristic. Есть late ABI/count/callback failures, неиспользованная
плохая ветка, hits+misses, duplicate indices, source-ready boundary и typed rollback.

Общие локальные `tools/ub` (25 selftests), без stage-specific копий:

- Observer recorder/resolver: CPU CTest 2/2 и отдельный parent-only preload/exec
  fixture; **8/8 semantic mutants killed**, CRAP-region <10. Для recorder минимум
  region coverage 2/3, max CRAP 3,333; для resolver минимум 5/6. Неисполненные
  allocation/export/overflow error branches не названы полным покрытием.
- Coordinator/cache: **39 definitions**, source-verified LLVM host coverage,
  CRAP-region max **8,013** (<10), минимум region coverage 2/3, unknown coverage нет.
  Полные тела пяти изменённых core TUs проверены cross-file dryer на rank ≥0,8,
  min 50 tokens. Старый дубль двух result-tree обходов устранён единым typed DFS
  после поведенческого pin и повторной SQL-регрессии; алгоритм обхода тот же.
- Финальная targeted mutation campaign — **12/12 killed**, без build failures,
  timeouts или survivors: `quality/overlap-mutants-v3/`. Исходная **11/12** в
  `overlap-mutants/` сохранена: удалённый rethrow в `finish()` маскировался вторым
  error channel через future. Прямой API test независимо проверяет repeated
  `finish()` до потребления futures; позднейшие 12/12 не переписывают survivor.
- ASan+UBSan host campaign проходит: `quality/overlap-sanitizers-v2/`. Первый
  запуск сохранил **stack-use-after-scope в новом test fixture**: completion counter
  объявлялся после retaining cache. Исправлен порядок его lifetime; production
  runtime completion владеет `shared_ptr`, этот dangling capture там не вводился.

`quality/*/{report.json,actions.jsonl}` и command/stdout/stderr сохраняют raw и
scope. Host coverage/sanitizers не являются device coverage, не охватывают весь
host-код GPU adapter/SQL engine и не заменяют указанные аппаратные oracles.
Doxygen-комментарии добавлены в код; HTML не генерировался. GPU regressions и
указанные overlap quality campaigns завершены до подтверждающей performance-серии.

После **обеих** коллекций восстановлена воспроизводимость существующих
`test/pcuda_stage{5,8}/host-recipe.json`: добавлены linkage `module_build.cpp`
и `-pthread`, stage-5 mutation нацелена на уже действующий ABI v2, stage-8
source-ready target сделан однозначным после появления второй API-ветки.
Первоначальный baseline link failure сохранён в `quality/prior-host-recipe-red/`
и **не считается kill**. Новые независимые кампании:
[`stage5-current-host-recipe`](../../build/pcuda-stage8-analysis/run-grgh_dvo/quality/stage5-current-host-recipe/report.json)
и [`stage8-current-host-recipe`](../../build/pcuda-stage8-analysis/run-grgh_dvo/quality/stage8-current-host-recipe/report.json)
дали **12/12 killed каждая**, без build failures/timeouts/survivors. Это новые
host-проверки текущих recipes, не пересчёт исторических raw этапов 5/8 и не
изменение измеренного runtime. После коллекций C++/binary inputs не менялись.

Confirmation дополнительно проверил **1008 CPU-результатов**: 756 file matches,
252 прежних Q1.1 отдельно CPU-verified, **896 warm** без новых шести S1 событий.
112 cold processes действительно подготовили планы; всего **476** module_codegen /
compiler /linker /load и **1428** symbols. Все 476 реально emitted `.cpp`
совпадают с исходными 12 программами после нормализации только module
identity/namespace SHA-256: алгоритм ядра не заменён. Сохранены **336+140**
Clang/LLD JSON. Максимальный прямой frontend interval — **1,316 мс** на этой
выборке, не новый SLA для произвольного SQL. `overlap-final/invariants.json`
проверяет также отсутствие ложных additive components у experimental model.

## 6. Выводы, варианты и решения

### 6.1. Исходные варианты cold-ускорения

Ниже сохранены оценки до согласованного эксперимента. Вариант **A** позднее
реализован только opt-in (4.5); **B–E** не выбраны и не внедрены. Наличие A
не разрешает автоматически остальные варианты или default-on adoption.

**A. Перекрыть внешнюю сборку и подготовку/исполнение готовых модулей.**
Для Q4.2 после готовности dimension `.so` запустить их обычные ordered-stream
операции и device JIT, пока отдельная цепочка собирает fact `.so`:

```text
Сейчас:      S1: [4 dimension .so] → [fact .so] → S2/build dimensions → S2/fact
Кандидат:    S1: [4 dimension .so] → [fact .so] ─────────────────────┐
                    готовность └→ S2/build dimensions ─────────────┴→ S2/fact
```

Это может сократить **критический путь**, не уменьшая работу компилятора и не
поднимая лимит acpp jobs. Простая новая сортировка batches не делает такого
перекрытия; прежние 6,298 мс counterfactual его не оценивают. Сколько времени
удастся скрыть, зависит от реальных S2 intervals и CPU/memory contention;
экономия 150 мс **не обещается**.

Нужен интерфейс module readiness/futures и runtime-координатор зависимостей:
compiler не владеет query memory, генератор не управляет исполнением. ABI проверяется
до запуска каждого модуля; cache reservations/публикация остаются атомарными либо
получают явно согласованный новый контракт. Если поздний compile/ABI check падает,
частично запущенный запрос должен безопасно завершить pending GPU work перед
RAII cleanup/unload и сохранить retry/typed-cache invariants. Нельзя просто
запустить `executor` на неполном `LoadedKernels`. GPU-зависимости можно сохранить
в одном явном ordered stream; дополнительный stream сам по себе не требуется.

**B. Уменьшить количество разных программ, требующих cold-компиляции.**
Вместо четырёх SQL-generated dimension build kernels изучить небольшой набор
заранее подготовленных PCUDA PHT-build services с runtime POD descriptor:
колонки, допустимые фильтры, параметры и key/payload layout. Fact scan/probe/aggregate
остаётся специализированным и fused; service kernels тоже сохраняют fused
filter+build. Это не whole-query `.so`, не по одному kernel на SQL-оператор и не
CPU SQL fallback. Подготовка универсальных services при сборке соответствует
архитектуре проекта, но новый service path **ещё не реализован**.

Выигрыш возможен в S1 (меньше отсутствующих SQL `.so`) и S2 (reuse одного service
code object между dimensions), но prebuilt generic `.so` **всё ещё требует первого
device JIT**. Разные runtime specialization/configuration keys могут породить
несколько S2 variants; «одна `.so`» не доказывает «одна device compilation».
Нужны прямые counters и проверка. Не переносить SQL-specific JIT скрыто в startup.
Сначала ограниченный PHT fast path с сохранением общего корректного fallback;
NULL/OR/MHT/Text, signed/zero keys и fixed ABI v2 не заменять предположениями SSB.
Универсальные ветвления могут замедлить warm или увеличить IR — эффект неизвестен.

**C. Параллелить независимые device-компиляции S2.**
Это уже требует изменения AdaptiveCpp, не `jit_jobs=4` и не ещё одного std::thread.
В проверенном commit `get_or_construct_jit_code_object` держит общий `_mutex`
во время `jit_compile` и создания code object; lookup готового кода тоже берёт
его. JIT-TIER подтвердил задержку готового LOW **0,125 →34,611 мс** при фоне.
Удалить lock нельзя: нужны короткая публикация, per-key in-flight/dedup,
независимые LLVM contexts/compiler state, проверка HIP/COMGR thread safety,
ограничения ресурсов и безопасные shutdown/unload/error paths.
Также нужен API подготовки без фиктивного GPU launch и без общей mutable `_config`.
Это потенциальное сокращение wall time, не гарантированное уменьшение CPU-work.
Только отдельный изолированный runtime experiment после явного согласования;
установленный compiler/runtime движка не заменять автоматически.

**D. Сократить обязательную повторную работу самого S2-компилятора.**
После прямой атрибуции проверить повторные parsing/linking общих bitcode/builtins,
запуски внешних инструментов и возможность process-only reuse неизменяемых
compiler inputs. Это не тот PCH, который ускоряет C++ S1. Нельзя без измерения
приписать весь неизмеренный остаток чтению IR или драйверу; reusable LLVM objects
имеют context/ownership/thread-safety ограничения. Новый LLVM/COMGR/toolchain
может менять цену codegen, но ускорение не проверено и upgrade требует отдельного
совместимого стенда. Удалять обязательные passes или менять SQL math semantics
ради времени нельзя. Это более глубокое исследование, не очередной перебор O0.

**E. Изолированно проверить пять S1 slots вместо четырёх.**
Для пяти независимых модулей нет compile/link dependency, требующей ждать
готовности четырёх. Пятый слот мог бы убрать очередь, но увеличить конкуренцию
за CPU/RAM; величина выигрыша неизвестна. В текущем engine диапазон jobs **1–4**,
и контрольный профиль согласован до четырёх. Менять его молча нельзя; вариант
допустим только как отдельно утверждённый эксперимент с парным baseline.
Это ускоряет внешнюю сборку, **не** устраняет дорогое S2.

**Что не выдаём за решение истинного cold:** скрытый прогрев до summary,
перенос произвольного SQL JIT в startup, кеш результатов, постоянный кеш SQL `.so`,
CPU fallback, смену PCUDA на vendor-only backend или объединение всех kernels
в query bundle. Штатный device JIT-cache AdaptiveCpp может помочь новому процессу
при уже готовом коде, но это другой cache state, не ускорение первого пустого cache.
Replay касается S1 и по-прежнему отложен. Tiering после ответа требует дешёвого
первого исполнителя и сам по себе не ускоряет первый результат.

### 6.2. Согласованная последовательность и критерии

Исходный, затем согласованный пользователем порядок: **сначала прямые S2
timers/counters для Q1.1/Q2.1/Q4.2**, используя опыт JIT-TIER,
но на полном `db_cli` пути и в изолированном диагностическом окружении. Разделить
S1 compile/link, S2 transformation/backend/load, runtime allocations/fills/waits,
GPU durations и service fetch; не складывать вложенные интервалы.
Сопоставимая plain-серия обязательна: прежний инструмент сам заметно менял summary.
Не объявлять `first execution − GPU kernel sum` чистым временем JIT.

Исходно после атрибуции приоритет имели **A (overlap)** либо **B (меньше разных
build programs)**; пользователь разрешил следовать плану S2→**A**, не реализовать
всё сразу. C/D затрагивают runtime/compiler и существенно рискованнее.
E — независимый ограниченный опыт, если пользователь разрешит отступить от
четырёх slots. Это предложение выбора, не поручение реализовать всё сразу.

**Согласованный S2→A объём выполнен:** S2 transformation прямо наблюдён,
default-off readiness protocol реализован, correctness/quality/повторные серии
проверены. Подтверждён выигрыш Q4.2 в этой серии; одноволновым Q1.1/Q2.1 ускорение
не приписывается. Включение A по умолчанию остаётся отдельным решением, не новым
обязательным gate. SQL-цели/закрытие этапа 8 требуют согласования пользователя.

Для любого согласованного изменения: сохранение ABI/cache/RAII/ordered stream,
failure/retry/partial-compile fixtures, отсутствие лишних JIT events при смене
совместимых values/typed NULL, регрессия 13 SSB/current manifest/manual/Text/NULL,
локальные C++ ub-tools для выбранных изменённых компонентов. Затем минимум
7 cold ×8 warm, прямой summary, native compile/link timers и сопоставимые
Clang/LLD trace/plain серии. Warm измеряется как контроль регрессии, а не новая
задача его оптимизации. Абсолютные SQL-цели согласовать отдельно.

### 6.3. Найденные warm-недостатки — сохранены, оптимизация отложена

**Решение пользователя 2026-11-12:** warm пока не представляет большой проблемы.
Ниже сохранён backlog, **не текущий план реализации и не новое условие закрытия
этапа 8**. Сейчас приоритет — cold-диагностика; предложение начать с warm отменено.
Абсолютные SQL-цели по-прежнему не утверждены.

| ID | Подтверждённый недостаток реализации | Пока не доказано / будущая проверка |
| --- | --- | --- |
| W1 | Fact `BlockLoad` не использует текущую selection mask | Фактическая экономия traffic/времени после S2; сохранить OR/NULL/MHT и padded lanes |
| W2 | Key-only PHT резервирует `2*slots`; occupancy — UInt32 на слот | Влияние меньшего layout/bitmap на cache, atomics и latency; отдельно проверить конкурентное build-поведение |
| W3 | PHT indexing содержит runtime modulo; в Q3.3 ISA видны соответствующие вычисления | Доля стоимости modulo; допустимость direct-domain lookup для каждого каталога, out-of-domain/alias rejection |
| W4 | Групповые kernels делают 2–3 `atomicExch` ключей на каждую подходящую строку | Доля contention и эффект декодирования ключей/локальной агрегации без потери presence/NULL/точности |
| W5 | Домены групп существенно шире результата: Q3.3 437500/24, Q3.4 437500/4, Q4.3 1750000/795 | Эффект runtime compression на весь запрос; не вшивать обычные значения в код, не ухудшать cold |

Замеры и ограничения — [разделы 4.7/4.9](../pcuda_stage8/REPORT.md); найденные свойства не объявляются
измеренными причинами замедления каждое по отдельности. Q1.* count/prefix/write
для одной строки тоже сохранены как кандидат, но прежде всего **cold/JIT**:
в warm их GPU kernels суммарно около 0,01 мс.

**Прежнее предложение для будущего согласования:** небольшие общие изменения
внутри fact kernel, а не
тринадцать SQL-specific реализаций. В `include/crystal/load.h`/`src/core/visitor.cpp`
сохранять selection mask для последующих fact loads; в
`include/crystal/hash.h`/`include/crystal/join.h` и PHT lowering отдельно проверить
checked direct-domain lookup и точную key-only ширину таблицы. Изменять по одному
фактору: эти кандидаты могут уменьшить чтения/адресную арифметику и workspace,
но **ускорение пока не измерено**. Нельзя считать их одним неразделимым patch или
обещать выигрыш всех 13 запросов заранее.

Ограничения этого предложения:

1. Mask относится к **входу текущего оператора**, не к промежуточному состоянию
   одной ветки OR; inactive/padded lanes детерминированы. Отдельно проверить
   NULL/3VL, OR/MHT, partial/empty tiles и отсутствие чтений неактивных lanes.
2. Direct lookup допускается только по доказанному диапазону и уникальности
   build keys. Сохранить signed/zero keys, NULL, rejection out-of-domain/aliases,
   проверку размеров и обычный путь там, где контракт не доказан. Простая замена
   `% length` на вычитание неверна: текущие ключи 1..N с minimum=0 используют
   wrap для N; нужен согласованный новый layout, не удаление проверки.
3. ABI v2/layout, explicit ordered stream, RAII, HWM plan ownership и value-free
   module cache не меняются. Обычные значения не становятся codegen constants;
   changed values/typed NULL не должны требовать новых модулей.

**Следующий, отдельный по риску шаг:** уменьшить повторные записи ключей групп
через atomics (декодирование ключей при finalization, локальное объединение
агрегатов там, где допустимо). Проверять presence независимо от ненулевой SUM,
nullable/нулевые/отрицательные группы и Float64; не добавлять фиксированный warp
size или незаконное reassociation. Runtime compact group domains для
Q3.2/Q3.3/Q3.4/Q4.3 рассмотреть после этого: они уменьшают workspace/инициализацию,
но текущая компактизация сама по себе не основной warm bottleneck. Не зашивать
текущие два города/годы/brand в исходник или статические массивы.

**Cold отдельно:** упростить scalar aggregate finalization Q1.* и оценить цену
device JIT/числа специализированных модулей; compiler batch scheduling оставить
низким приоритетом согласно counterfactual выше. Не вводить replay, eager SQL
compilation вне summary или постоянный SQL cache. Изменение shapes kernels может
повлиять на cold в обе стороны, поэтому каждый кандидат требует нового замера.

После каждого согласованного изменения: independent behavioral fixtures,
локальные C++ ub-tools для изменённых компонентов, 13 SQL SSB/manifest/manual/
Text/NULL регрессия и zero-build counters; затем парные серии не менее **7 cold ×
8 warm** с обычным и Clang/LLD diagnostic режимами. Profiler trials — отдельно.
Сначала установить корректность и измеренный эффект; только затем предлагать
новые абсолютные SQL-цели. Ни один пункт этого списка ещё не реализован.

## 7. Воспроизведение и оставшаяся работа

Проверки и подтверждающие SQL/PCH/prototype серии воспроизводятся командами
раздела 7 [отчёта этапа 8](../pcuda_stage8/REPORT.md). Точные profiler argv/env/stdin,
изолированные build/install и raw durations — в `rocprof-repair-x12zi8pl/`;
дамп SQL sources и исторические команды — в `debug/ssb-cold-q11-q21-q42/`.
Для повторов использовать новый каталог, не запускать исторический build script
поверх raw. Исторический JIT-TIER имеет собственные команды/артефакты в разделе 7;
его результаты здесь только сверены, не проведены заново.

Согласованное продолжение завершено: stock/probe S2-серия и ограниченный overlap.
Текущие артефакты —
[`run-grgh_dvo`](../../build/pcuda-stage8-analysis/run-grgh_dvo/): snapshot 707 файлов,
`initial-files.json`, начальные diff/status, `actions.jsonl` и полный сырой вывод
каждой проверки. Измеритель — `measure.py collect`; offline-reader — `analyze`.
Первоначальные неуспешные smoke и GDB/ELF-binding диагностика сохраняются отдельно
от подтверждающей серии. HIP runtime totals observer имеют scope всего процесса
(data loading, cold, warm, teardown), а не отдельного SQL. Отсутствие внутренних
hooks из-за inline/direct calls не означает нулевую стоимость этих стадий.
Не переносить runtime changes в установленный стек до независимых проверок.
Перенос текста проверяется по snapshot и всем строкам таблиц; результаты этой
документальной работы — `analysis-docs-4hyd324d/validation.json` и `actions.jsonl`
в общем корне артефактов. Commit/push и очистка чужих работ не выполнялись.

Минимальное воспроизведение текущего эксперимента (новый пустой `RUN`; SDK/GCC
paths — конфигурация стенда, не hardcode production-кода):

```bash
RUN="$PWD/build/pcuda-stage8-analysis/reproduce"
GCC=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake -S . -B "$RUN/engine" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC" \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DCRYSTAL_PCUDA_COMPILER=/opt/AdaptiveCpp/bin/acpp \
  "-DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=$GCC" \
  '-DCRYSTAL_PCUDA_LINK_FLAGS=-fuse-ld=lld;-Wl,-O0' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUN/engine" -j4
cmake -S test/pcuda_stage8_analysis -B "$RUN/probe" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC" \
  -DLLVM_DIR=/opt/rocm_sdk_612/lib/cmake/llvm \
  -DACPP_PREFIX=/opt/AdaptiveCpp -DROCM_PREFIX=/opt/rocm_sdk_612
cmake --build "$RUN/probe" -j4
ctest --test-dir "$RUN/engine" --output-on-failure
ctest --test-dir "$RUN/probe" --output-on-failure
python3 test/pcuda_stage8/validate.py --build "$RUN/engine" \
  --output-root "$RUN/default-regression" --timing-model pcuda-modules-v1
python3 test/pcuda_stage8/validate.py --build "$RUN/engine" \
  --output-root "$RUN/overlap-regression" --timing-model pcuda-modules-overlap-exp-v1
# Обе коллекции последовательно; никакой oracle/analysis между ними.
python3 test/pcuda_stage8_analysis/measure.py collect --build "$RUN/engine" \
  --probe "$RUN/probe/libs2_probe.so" --loader /lib64/ld-linux-x86-64.so.2 \
  --output-root "$RUN" --sessions 7 --runs 9 --diagnostics both --mode stock \
  --schedule barrier --schedule overlap --timing-model pcuda-modules-v1
python3 test/pcuda_stage8_analysis/measure.py collect --build "$RUN/engine" \
  --probe "$RUN/probe/libs2_probe.so" --loader /lib64/ld-linux-x86-64.so.2 \
  --output-root "$RUN" --sessions 7 --runs 9 --diagnostics both --mode probe \
  --query q42 --schedule barrier --schedule overlap --timing-model pcuda-modules-v1
# Только после ВСЕХ timed процессов:
/opt/rocm_sdk_612/bin/clang++ "--gcc-install-dir=$GCC" -std=c++17 -O3 \
  test/pcuda_stage0/q11_reference.cpp -o "$RUN/q11-reference"
"$RUN/q11-reference" data/10 >"$RUN/q11-oracle.json"
# Подставить отдельно напечатанные s2-series-* для двух коллекций.
python3 test/pcuda_stage8_analysis/measure.py analyze --series "$RUN/s2-series-ID" \
  --q11-oracle "$RUN/q11-oracle.json" --output "$RUN/analysis-ID"
```

Исходный S2 stock/probe cohort воспроизводится первым `collect` без `--mode stock`
и без `--schedule`: по умолчанию оба observer режима, штатный barrier.
Host quality recipes — `host-recipe.json` (observer) и `overlap-recipe.json`
(coordinator/cache). Их запускают общие `tools/ub/{coverage,crapper,dryer,mutator}.py`
в отдельных новых output directories, **не параллельно performance-серии**.
Полные реальные argv/flags/reasons, ранние red/неудачи, source/binary hashes и
сырой вывод находятся в `run-grgh_dvo/actions.jsonl` и command files.

Финальный read-only audit —
[`final-audit/`](../../build/pcuda-stage8-analysis/run-grgh_dvo/final-audit/):
diff относительно `initial/` и `before-confirmation/`, сохранность всех 707/724
snapshot-файлов, manifest/runtime/data hashes, ссылки и числовая сверка таблиц.
После серий менялись пять документов, два quality recipe и автоматически
перегенерированный Python bytecode; C++/binary inputs не менялись. Предсуществующие
`__pycache__` не удалялись, `sycl.csv` неизменён. Ошибка первого audit в трактовке
`git diff --no-index` exit 1 сохранена отдельно: это были различия без whitespace
ошибок, не failure исходников или tests. Журнал действий/причин —
[`actions.jsonl`](../../build/pcuda-stage8-analysis/run-grgh_dvo/actions.jsonl).

Оставшиеся решения — абсолютные SQL cold/warm-цели и приёмка этапа 8; возможное
default-on adoption требует отдельного разрешения. B–E, warm W1–W5 и replay
не начаты автоматически. Текущий ограниченный эксперимент новых замеров не требует.
