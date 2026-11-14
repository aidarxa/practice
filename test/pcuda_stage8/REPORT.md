# Этап 8: проверки и окончательная приёмка PCUDA

## 1. Статус и границы

**Реализация, функциональные проверки и измерения выполнены 2026-11-11 на AMD Radeon
780M. Этап открыт: после измерений пользователь выбрал сначала оптимизацию,
а перед изменением алгоритмов — общий анализ всех 13 SSB.** Предложенная ниже
таблица SQL-целей **не утверждена**. Анализ, восстановление профилирования и
согласованный default-off overlap-эксперимент с повторными измерениями завершены
2026-11-12 в отдельном 8-ANALYSIS. Абсолютные cold/warm-цели SQL требуют
отдельного утверждения; эксперимент не закрывает приёмку.
**Уточнение пользователя 2026-11-12:** warm пока не является большой проблемой;
найденные недостатки warm сохранить, но его оптимизацию **отложить**. Текущий
фокус — cold-разница Q1.1/Q2.1/Q4.2 и ограниченный overlap-эксперимент,
не реализация предложенных warm-изменений.
По просьбе пользователя аналитическое продолжение выделено как
[8-ANALYSIS](../pcuda_stage8_analysis/REPORT.md): cold-разбор, варианты ускорения
JIT и отложенный warm-backlog ведутся там. Здесь остаются проверки и исходные
подтверждающие/диагностические измерения; выделение отчёта не закрывает этап.
Прежние **600/200/10/9 мс** остаются требованиями отдельного прототипа Q3.3,
не автоматически требованиями полного движка.

Артефакты: [`build/pcuda-stage8/run-8xun1k4x`](../../build/pcuda-stage8/run-8xun1k4x/).
Начальные 344 файла сохранены в `initial/`, hashes — `initial-sha256.json`,
незакоммиченный diff/status — `initial.diff` / `initial-status.txt`.
Журнал действий и причин — `actions.jsonl`, команды и сырой вывод —
`*.command.json`, `*.stdout`, `*.stderr`. Чужие изменения и прежние raw не удаляются.

### Идея изменений простыми словами

Раньше нельзя было точно проверить «SQL→PCUDA ≤10 мс»: `prepare_ms` включал
подготовку GPU-памяти, а `codegen_ms` не охватывал выпуск обёрток модулей.
Теперь один таймер начинается перед разбором SQL и заканчивается, когда все
исходники готовы, **до** запуска компилятора. Компиляция, память и выполнение
по-прежнему входят в полный непрерывный `summary`, но не в этот frontend-бюджет.

Отдельный observer в начале `main` отмечает startup, загрузку данных и первый
успешный CPU-результат. Native parent получает marker через pipe **до печати**.
Память/планы, PCUDA-ядра, ABI v2 и политика кешей не заменялись. Replay,
persistent SQL cache, новые SQL-возможности и HTML/Doxygen-генерация не добавлены.

## 2. Окружение и конфигурация

Версии, env и команды сохраняются в `environment/`, `engine/CMakeCache.txt`,
`engine/compile_commands.json`, PCH manifest и module `.compile/.link.command`.
Установленный AdaptiveCpp: `25.10.0+git.5989e1cf.20260821.branch.HEAD`.
C++17, host Release, динамические модули generic PCUDA `-O0`, готовый PCH,
штатный acpp и до четырёх независимых compile/link chains.

- Устройство: AMD Radeon 780M Graphics (`gfx1103`), FP64/atomic64 подтверждены
  runtime. NVIDIA/Intel аппаратно не проверялись.
- AdaptiveCpp commit прежнего установленного стека:
  `5989e1cfa4e70d55c65e786a4c71acb124ce54ff`; Clang/LLVM/LLD 17.0.0,
  ROCm LLVM revision `f8542d25dc0d56558ea2f9fbb0a2aae9433c1e27`.
- Host Release `-O3 -DNDEBUG`, тесты дополнительно `-UNDEBUG`. SQL modules/PCH:
  `--acpp-targets=generic --acpp-pcuda -std=c++17 -O0 -g0 -fPIC`,
  `-Wno-narrowing -Wno-deprecated-declarations`, include проекта и явно
  конфигурируемый для этого стенда
  `--gcc-install-dir=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1`.
  Link: отдельный `acpp ... -fuse-ld=lld -Wl,-O0 -shared`.
- `LC_ALL=C`, `TMPDIR=/tmp/opencode`, `ACPP_ADAPTIVITY_LEVEL=1`,
  `ACPP_ALLOCATION_TRACKING=1`. Стек/драйвер не менялись. PCH — **4 736 312 B**;
  его fingerprint учитывает ABI v2 и device headers, не SQL values.
- Все данные SF10 загружаются до summary, включая eager Text, даже для
  числовых SSB. Прототип читает лишь необходимые десять числовых колонок;
  его startup/data-load нельзя выдавать за startup полного движка.
- Дополнительная GPU-диагностика использует исходники пользовательского SDK
  `/home/aidar/rocm_sdk_builder`: rocprofiler намеренно закреплён recipe на
  `rocm-5.3.3`, roctracer — компонент сборки ROCm 6.1.2. Исправленный profiler
  собран/установлен **только** в `rocprof-repair-x12zi8pl/{build,install}` внутри
  артефактов этапа; compiler, HIP/HSA runtime и `/opt/rocm_sdk_612/bin/rocprof`
  не заменялись. Имеющиеся tracer `.so` подключены в изолированный prefix
  ссылками; полная пересборка ROCm не запускалась.

Исходные и effective acpp/Clang/LLD argv сохранены в module command files и
`module-inspection/0/*-effective.*`. Это offline dry-run, не replay. Финальные
hashes, dependency/preservation audit и команды воспроизведения находятся в
`environment/`, `stage8.diff`, `stage8-changed-files.json` и финальном inventory.

## 3. Методика

Измерения процессов — native Bash `time`; внутренних интервалов — C++
`steady_clock`. Python только организует процессы и разбирает готовые файлы.
Performance-серия выполняется отдельно от сборок/quality/GPU-регрессии;
статистика, traces и независимые CPU-oracles — после **всей** серии.
Новый process/module/device cache на каждый SSB/session/режим; внутри warm
сохраняется. OS/HIP/driver caches не очищаются. Частоты/температуры не закрепляются.

`summary = total_query_ms`: от передачи SQL в уже загруженную БД до owning CPU
результата, без startup, base loading, oracle, форматирования/печати.
Frontend — отдельный прямой, перекрывающийся интервал; к component sum не
прибавляется. Суммы native compile/link при параллелизме не равны build elapsed.
Разница cold/warm не называется чистым device JIT.

Подтверждающая SQL-серия:
[`ssb-series-y58itcgc/series.json`](../../build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc/series.json).
**182 независимых cold-процесса:** 13 запросов ×7 сессий ×2 режима.
В каждом процессе один SQL первый, затем восемь немедленных warm; Q3.3 также
восемь compatible changed-value executions (города 235/235, годы 1993–1996).
Итого **1750 CPU-результатов**, включая **112** changed. Порядок SQL циклически
сдвигается между сессиями; plain/traced чередуются внутри каждой пары.
Ни сборки, ни mutation/GPU clients, ни oracle/trace analysis параллельно не шли.
`measure` сохранял только raw; отдельный `analyze` запущен после всей SQL и
контрольной prototype-серии. Первоначальный n=1 smoke не включён в статистику.

Все таблицы — **мс, median [min;max]**, если явно не написано иначе. Cold n=7,
warm n=56 на запрос/режим; каждый компонент имеет собственную медиану.
Медианы/вложенные trace-интервалы не складываются для восстановления summary.
Разброс — наблюдаемый диапазон, не доверительный интервал и не maximum SLA.
Новые численные цели не являются доказательством ускорения относительно SYCL.

Дополнительная серия rocprof — **диагностика, не замена подтверждающей серии**:
13 запросов × один независимый процесс plain/rocprof, один cold + восемь warm
в каждом; порядок режимов чередуется между запросами. Отдельные новые appdb и
query/module caches, OS/HIP/driver caches не очищаются. Сборки, quality tools,
другие GPU clients и offline analysis во время серии не запускались. `jit_traces`
выключен в обоих режимах; прежняя отдельная Clang/LLD trace-контрольная серия
сохранена. GPU tracing overhead показывается отдельно, не вычитается из baseline.
Raw пишется сразу, oracle/statistics/trace analysis выполняются после всех 26
процессов. Из `profile.json` берутся **BeginNs/EndNs/DurationNs**, а не округлённый
строковый `dur` в микросекундах; GPU lanes определяются по metadata process names.

## 4. Результаты

### 4.1. Функциональная и quality-приёмка

| Проверка | Результат | Артефакт |
| --- | --- | --- |
| Release CTest / CPU-only | **19/19 /17/17** | `ctest-final.stdout`, `cpu-ctest-final.stdout` |
| Новые runner checks | **5/5** | `runner-green.stderr` |
| Host ASan/UBSan | Без ошибок, не device sanitizer | `ub/sanitized-final/report.json` |
| Изменённые cache/startup host definitions | CRAP-region <10; coverage неполное | `ub/crapper-{cache,startup}-final/report.json` |
| Whole-body dryer | Пар rank ≥0,8 не найдено в выбранных двух sources | `ub/dryer-v1/report.json` |
| Host /GPU mutations | **12/12 /2/2 killed**, survivors/unresolved=0 | `ub/{host-mutants-v2,gpu-mutants-v1}/report.json` |
| Current/current_heavy manifest +targeted | **88/88** | `functional-a7th5n17/manifest/manifest-plain-heut02vc/report.json` |
| Независимые manual SSB | **52/52** | `functional-a7th5n17/manual/manual-plain-8amd0psu/report.json` |
| SQL SSB 13×3 | **36/39** file matches, только прежний Q1.1×3 отдельно CPU-oracle | `functional-a7th5n17/ssb/ssb-plain-utura0t1/report.json`, `q11-reference.stdout` |
| Native Text matrix | **131/131**, включая ожидаемые строгие отказы | `functional-a7th5n17/text-gpu/gpu-plain-wf_ov9vb/report.json` |
| Восемь прежних native resource/numeric/MHT/OR/module/primitive/manual clients | Все проходят | `functional-a7th5n17/report.json` |
| Один структурно изменённый kernel | Ровно **1 codegen/compile/link/load**, остальные hits, возврат без build | `gpu-modules-final.stdout` |
| Assertions в Release | 29 compilations с действующим `-UNDEBUG`, 4 negative builds при `NDEBUG` | `release-assertions.json` |
| Независимые Text/star counts/payload | Все пять форм, **17 559** видимых строк/всех ячеек | `manifest-text-audit/report.json` |
| SF10 Text CLI plain/traced | **82/82: 72 результата +10 ожидаемых отказов** | `functional-a7th5n17/text-cli-{plain,traced}/` |
| Локальные ub-tools self-tests | **25/25** | `ub-selftests.stderr` |
| Native ABI/registration | **64 .so /3 process scopes**, один HCF/kernel на модуль, entries v2 | `module-inspection/report.json` |

CRAP-region проверен для **всех 26 definitions** двух выбранных host sources:
`module_cache.cpp`, `startup_timing.cpp`; max **8,013 <10**, семь покрыты частично.
Это не покрытие всего `QueryEngine`, `visitor`, storage/runtime или device-кода.
Host coverage не выдаётся за device coverage; device correctness проверен
реальными аппаратными clients. Dryer — whole-body heuristic ≥50 tokens,
не доказательство отсутствия всех partial/renamed clones. Мутации targeted,
пересобраны в отдельных копиях после свежей baseline; timeout/build failure
не объявлялись kills. Device sanitizer не запускался.

### 4.2. Полный SQL: cold summary и компоненты без traces

| SQL | **summary [min;max]** | SQL→PCUDA median | Build elapsed | Load | First execution | Fetch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1.1 | **504,281 [498,073;515,264]** | 0,645 | 157,758 | 0,416 | 343,723 | 0,384 |
| Q1.2 | **510,866 [500,959;515,921]** | 0,663 | 157,192 | 0,276 | 349,616 | 0,638 |
| Q1.3 | **505,467 [503,254;511,128]** | 0,671 | 157,225 | 0,413 | 345,404 | 0,461 |
| Q2.1 | **768,106 [765,455;780,811]** | 0,788 | 151,737 | 0,325 | 544,813 | 69,017 |
| Q2.2 | **766,327 [759,450;776,483]** | 0,838 | 151,761 | 0,332 | 542,552 | 69,134 |
| Q2.3 | **752,557 [748,354;758,152]** | 0,798 | 150,641 | 0,323 | 530,257 | 69,107 |
| Q3.1 | **777,556 [767,934;810,056]** | 0,924 | 152,235 | 0,325 | 553,297 | 69,026 |
| Q3.2 | **767,986 [766,346;852,196]** | 0,915 | 153,346 | 0,326 | 543,226 | 69,161 |
| Q3.3 | **669,055 [661,411;691,137]** | 0,925 | 152,184 | 0,279 | 445,891 | 69,917 |
| Q3.4 | **652,686 [650,828;676,936]** | 0,892 | 150,406 | 0,405 | 432,115 | 69,870 |
| Q4.1 | **1047,404 [1043,247;1092,066]** | 1,016 | 305,219 | 0,607 | 669,214 | 69,460 |
| Q4.2 | **1090,962 [1085,539;1150,029]** | 1,147 | 307,982 | 0,634 | 712,595 | 69,148 |
| Q4.3 | **813,395 [807,995;823,288]** | 1,058 | 152,286 | 0,362 | 585,099 | 70,703 |

Q4.1/Q4.2 готовят пять отсутствующих модулей при четырёх доступных chains,
поэтому нужен второй build wave. Это не новая whole-query линковка. Q3.3
имеет четыре фактических fused launches, но три уникальных .so:
канонически одинаковые customer/supplier builds разделяют модуль.
Во всех 182 новых процессах действительно были native compiler/link/load events.

Бюджет **SQL→PCUDA ≤10 мс выполнен на всех 1750 samples**, не только в медиане:
общий максимум **1,310 мс**. Это host frontend до готовых исходников, не latency
всего SQL. У готовых планов этот путь короче; максимум plain changed Q3.3 — 0,430 мс.

### 4.3. Warm и изменённые значения без traces

| SQL | **Warm total [min;max]** | Warm execution [min;max] |
| --- | ---: | ---: |
| Q1.1 | **12,520 [11,300;16,667]** | 12,057 [11,126;15,837] |
| Q1.2 | **12,677 [11,433;19,075]** | 12,178 [11,257;18,470] |
| Q1.3 | **12,463 [11,464;14,446]** | 12,211 [11,280;13,796] |
| Q2.1 | **53,692 [53,051;56,468]** | 53,375 [52,805;55,967] |
| Q2.2 | **44,040 [43,396;54,242]** | 43,728 [43,167;53,694] |
| Q2.3 | **40,547 [40,047;42,182]** | 40,269 [39,816;41,827] |
| Q3.1 | **24,651 [24,220;32,985]** | 24,343 [23,946;32,334] |
| Q3.2 | **19,401 [19,024;23,386]** | 18,833 [18,571;22,640] |
| Q3.3 | **17,168 [16,670;21,925]** | 16,538 [16,237;21,221] |
| Q3.3 changed | **16,331 [16,087;19,242]** | 15,854 [15,637;18,688] |
| Q3.4 | **17,133 [16,671;17,811]** | 16,469 [16,191;17,111] |
| Q4.1 | **34,412 [34,059;39,646]** | 34,045 [33,799;39,281] |
| Q4.2 | **40,320 [39,863;46,637]** | 39,948 [39,538;46,137] |
| Q4.3 | **24,627 [24,191;27,108]** | 23,654 [23,257;26,080] |

**1568 ready executions**, включая первый changed query каждого процесса,
имеют нулевые plan/module codegen, compiler, linker, load и symbol counters.
Это доказательство отсутствия SQL JIT движка, не обещание отсутствия работы
внутреннего device JIT AdaptiveCpp. Immediate repeats переиспользуют active
plan workspace; zero-allocation после произвольной смены SQL не обещается.

### 4.4. Native compile/link и trace-контроль

Plain; каждая ячейка — своя медиана, **Σ wall/user/system**, мс:

| SQL | Compile Σ wall/user/system | Link Σ wall/user/system | **summary** |
| --- | ---: | ---: | ---: |
| Q1.1 | 234,640 /180,536 /53,150 | 198,806 /160,039 /61,711 | **504,281** |
| Q1.2 | 236,306 /183,782 /54,708 | 198,467 /147,113 /73,742 | **510,866** |
| Q1.3 | 233,707 /183,242 /48,853 | 199,482 /151,948 /70,010 | **505,467** |
| Q2.1 | 322,806 /253,710 /71,699 | 265,098 /200,510 /90,406 | **768,106** |
| Q2.2 | 324,733 /262,835 /62,842 | 263,952 /194,364 /92,975 | **766,327** |
| Q2.3 | 322,957 /247,117 /74,070 | 265,999 /196,476 /96,287 | **752,557** |
| Q3.1 | 327,543 /258,551 /65,374 | 267,208 /196,649 /95,545 | **777,556** |
| Q3.2 | 327,745 /255,569 /80,269 | 267,528 /197,712 /90,786 | **767,986** |
| Q3.3 | 247,359 /191,294 /53,405 | 199,215 /147,828 /74,431 | **669,055** |
| Q3.4 | 244,724 /192,881 /51,897 | 197,146 /145,585 /70,224 | **652,686** |
| Q4.1 | 411,319 /322,515 /83,612 | 334,036 /257,915 /114,819 | **1047,404** |
| Q4.2 | 413,668 /327,576 /86,658 | 334,202 /246,174 /124,622 | **1090,962** |
| Q4.3 | 329,724 /253,808 /73,687 | 264,019 /194,884 /94,768 | **813,395** |

Суммы команд **не прибавляются** к parallel build elapsed. Native acpp включает
собственный Python-driver; дополнительного Python proxy/timer в цепочке нет.
Сохранены **343 Clang +343 LLD traces** SQL-серии; compile/link wall не заменяются
`ExecuteCompiler`/`ExecuteLinker`, вложенные фазы не складываются.

| SQL | **Traced summary [min;max]** | Traced build | Δ summary traced−plain | Traced warm total |
| --- | ---: | ---: | ---: | ---: |
| Q1.1 | **502,599 [500,198;508,902]** | 159,147 | −1,682 | 12,523 |
| Q1.2 | **505,276 [502,650;530,104]** | 159,109 | −5,590 | 12,498 |
| Q1.3 | **510,946 [501,287;517,127]** | 160,925 | +5,479 | 12,495 |
| Q2.1 | **767,880 [763,541;772,637]** | 154,514 | −0,226 | 53,526 |
| Q2.2 | **765,367 [761,584;775,872]** | 153,021 | −0,960 | 43,936 |
| Q2.3 | **755,647 [753,512;762,764]** | 154,903 | +3,090 | 40,520 |
| Q3.1 | **776,685 [775,252;791,538]** | 155,149 | −0,871 | 24,767 |
| Q3.2 | **768,517 [767,976;791,372]** | 153,606 | +0,531 | 19,351 |
| Q3.3 | **669,179 [663,201;689,575]** | 153,836 | +0,124 | 17,126 |
| Q3.4 | **659,870 [658,004;673,957]** | 153,607 | +7,184 | 17,130 |
| Q4.1 | **1047,553 [1043,663;1095,264]** | 310,271 | +0,149 | 34,606 |
| Q4.2 | **1094,970 [1090,330;1168,202]** | 310,702 | +4,008 | 40,224 |
| Q4.3 | **816,312 [809,163;867,560]** | 158,489 | +2,917 | 24,508 |

Это наблюдаемая цена диагностического режима с шумом execution/scheduling,
не выделенное чистое время overhead. Отрицательные Δ не означают ускорение
от traces. Ничего не вычиталось из исторических или новых raw.
Все traced компоненты, min/max, native sums и отдельные trace-фазы:
`ssb-analysis/report.json`, `tables.txt` и исходные session directories.

### 4.5. Полный старт процесса и PCH отдельно

Plain медианы; независимые прямые интервалы, не сумма медиан:

| SQL | Startup | Data loading | Main→CPU-ready | Spawn→CPU-ready | **summary** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1.1 | 2342,176 | 5294,280 | 8146,934 | 8149,752 | **504,281** |
| Q1.2 | 2338,253 | 5288,020 | 8129,063 | 8131,769 | **510,866** |
| Q1.3 | 2339,917 | 5286,229 | 8132,695 | 8135,584 | **505,467** |
| Q2.1 | 2338,138 | 5292,114 | 8396,444 | 8398,982 | **768,106** |
| Q2.2 | 2339,271 | 5299,041 | 8401,444 | 8404,373 | **766,327** |
| Q2.3 | 2339,731 | 5287,622 | 8379,740 | 8381,646 | **752,557** |
| Q3.1 | 2342,139 | 5298,923 | 8411,171 | 8414,120 | **777,556** |
| Q3.2 | 2337,637 | 5301,145 | 8413,313 | 8416,261 | **767,986** |
| Q3.3 | 2340,293 | 5299,644 | 8303,538 | 8305,604 | **669,055** |
| Q3.4 | 2337,765 | 5298,350 | 8283,801 | 8286,256 | **652,686** |
| Q4.1 | 2337,213 | 5288,209 | 8669,798 | 8671,716 | **1047,404** |
| Q4.2 | 2340,863 | 5283,876 | 8734,340 | 8736,710 | **1090,962** |
| Q4.3 | 2338,752 | 5289,118 | 8446,392 | 8448,508 | **813,395** |

Startup включает конструктор БД, выбор GPU, каталог и проверку PCH manifest.
Загрузка включает все numeric/Text base columns, их normalization и H2D.
Полный старт здесь **около 8–8,7 с**, а не 0,5–1,1 с query summary; его стоимость
не спрятана. Отдельные constructor subcomponents не измерялись, поэтому весь
startup не приписывается только драйверу или проверке PCH.

Отдельно семь rotated native acpp PCH preparations на режим с теми же headers/flags:

| Режим | PCH wall [min;max] | PCH user | PCH system |
| --- | ---: | ---: | ---: |
| Plain | 323,505 [315,384;531,290] | 289,274 | 32,886 |
| Traced | 330,133 [323,873;369,953] | 297,449 | 30,198 |

Raw: `pch-series/`. Это **отдельные эквивалентные копии**, не подмена input PCH
основной CLI-серии; первый CMake PCH build и создание fingerprint отдельно не
измерялись. PCH cost не амортизируется/не добавляется к SQL summary.
Δ wall median **+6,628 мс** — наблюдение, не поправка к историческим цифрам.

### 4.6. Повтор ограниченного контрольного прототипа Q3.3/V2

Исходники/методика PERF-COLD не менялись. Raw:
[`series-qpcbjsh7/report.json`](../../build/pcuda-stage8/run-8xun1k4x/series-qpcbjsh7/report.json).
Семь cold и 56 warm на режим, все **126/126** результатов совпали с независимым
SF10 reference. Четыре модуля, вместе **61 696 B**. Нет SQL parse/plan/codegen;
query preparation находится внутри полного V2 summary.

| Режим | **summary [min;max]** | Build elapsed [min;max] | Warm total [min;max] | Warm execution [min;max] |
| --- | ---: | ---: | ---: | ---: |
| Plain | **471,499 [469,554;475,697]** | 135,866 [134,670;138,759] | 7,714 [6,207;9,148] | 7,682 [6,180;9,105] |
| Traced | **475,645 [466,179;478,476]** | 136,939 [136,263;140,356] | 7,752 [6,195;9,332] | 7,717 [6,168;9,289] |
| Утверждённый предел медианы plain | **≤600** | **≤200** | **≤10** | **≤9** |

**Все четыре прежних gate выполнены.** Отдельный максимум warm execute >9 мс
не скрывается: утверждённый gate относится к медиане, не к максимуму.
Plain: preparation **33,127**, load **0,328**, первое execution **302,193**,
fetch **0,043**, startup **238,836**, data loading **173,887**, spawn→CPU-ready
**887,397 мс**. Эти медианы не складываются в summary/полный старт.
У прототипа есть свои native timers, traces и отдельные PCH preparations;
они не переносятся в таблицы SQL или цену нового engine PCH.

### 4.7. Анализ всех 13 SSB до изменения алгоритмов

Raw: `optimization-analysis-q5akas53/{analysis.json,tables.txt}`, отдельные
`q*/plan.txt` и исходники модулей. Read-only inspector подготовил те же
value-free планы: **на всех 13 запросах identities модулей точно совпадают**
с `.so` подтверждающей performance-серии. Это не новый оптимизированный backend.

| SQL | Запуски модулей / уникальные `.so` | Порядок fact probes | Домен групп / строки результата | Result / обнуления, MiB |
| --- | ---: | --- | ---: | ---: |
| Q1.1 /Q1.2 /Q1.3 | 3 /3 | нет join | 1 /1 | <0,001 /<0,001 |
| Q2.1 | 4 /4 | PART→SUPPLIER→DDATE | 7000 /280 | 0,160 /10,246 |
| Q2.2 | 4 /4 | PART→SUPPLIER→DDATE | 7000 /56 | 0,160 /10,246 |
| Q2.3 | 4 /4 | PART→SUPPLIER→DDATE | 7000 /7 | 0,160 /10,246 |
| Q3.1 | 4 /4 | CUSTOMER→SUPPLIER→DDATE | 4375 /150 | 0,134 /4,497 |
| Q3.2 | 4 /4 | CUSTOMER→SUPPLIER→DDATE | 437500 /600 | 13,351 /17,922 |
| Q3.3 | 4 /3 | CUSTOMER→SUPPLIER→DDATE | 437500 /24 | 13,351 /17,922 |
| Q3.4 | 4 /3 | CUSTOMER→SUPPLIER→DDATE | 437500 /4 | 13,351 /17,922 |
| Q4.1 | 5 /5 | SUPPLIER→CUSTOMER→PART→DDATE | 175 /35 | 0,004 /13,521 |
| Q4.2 | 5 /5 | CUSTOMER→SUPPLIER→PART→DDATE | 4375 /100 | 0,134 /13,653 |
| Q4.3 | 5 /4 | SUPPLIER→CUSTOMER→PART→DDATE | 1750000 /795 | 53,406 /67,757 |

Здесь не посчитаны служебные prefix/compaction kernels; число модулей не равно
числу всех GPU-запусков. «Обнуления» — объявленные байты инициализации плана,
не измеренный DRAM traffic. Result — внутренний разреженный агрегат, не объём
D2H: существующая GPU-компактизация уже копирует только фактические строки.

Подтверждённые общие свойства:

- Join/filter/aggregate уже fused. Порядок probes различается между семействами;
  нельзя объявлять joins неслитыми или менять их порядок только по SQL spelling.
- `BlockLoad` в `include/crystal/load.h` учитывает границу tile, но **не текущую
  selection mask**. PHT probe в `include/crystal/join.h`, напротив, уже пропускает
  неактивные lanes. Наличие C++ loads само по себе не доказывает реальный traffic
  после оптимизации AdaptiveCpp S2.
- PHT build в `src/core/visitor.cpp` резервирует `2*slots` и для key-only варианта;
  occupancy — отдельный UInt32 на слот. Это проверяемый кандидат уменьшения
  workspace/инициализации, а не уже измеренное ускорение.
- Q1.* после scalar aggregate всё ещё используют `AggregateDenseCount →
  DevicePrefix → AggregateDenseWrite` для одной строки. Уже существующий small
  fused materializer допускается лишь при SORT/HAVING; его не нужно внедрять заново.
- Групповые fact kernels записывают 2–3 ключа через `atomicExch` на каждую
  подходящую строку, затем агрегат. Перенос записи ключей в finalization возможен
  только при сохранении отдельного presence и корректности нулевых/NULL групп.
- Q4.1/Q4.2 имеют пять отсутствующих модулей и жёсткие batches по четыре.
  Longest-first моделирование на наблюдённых длительностях даёт разницу лишь
  **9,920 /6,298 мс**. Это counterfactual, **не** измеренный выигрыш планировщика;
  replay, новые flags и постоянный SQL cache не применялись.

AMDGPU metadata всех запросов и ISA Q3.3/прототипа сохранены в
`optimization-analysis-q5akas53/device-notes/`, hashes/команды — `device-notes.json`.
Это статические сведения compiled device JIT, не аппаратные occupancy/stall counters.
Например, SQL Q3.3 fact имеет 36 VGPR, 39 SGPR, 0 private segment и wave32;
в ISA остаются global loads и последовательности вычисления runtime modulo.
Наличие этих инструкций не измеряет их долю в задержке и не разрешает убирать
проверки signed keys, aliasing или границ perfect-domain lookup.

### 4.8. Восстановление rocprof без замены стека движка

Первый `rocprof --hip-trace` в `optimization-analysis-q5akas53/rocprof-q33/`
завершился кодом 0, но tool libraries не загрузились, JSON был повреждён.
Эта диагностика **остаётся неуспешной**. Первоначальная интерпретация
«библиотеки отсутствуют» уточнена: отсутствовали именно пути через `lib`;
реальные библиотеки находятся в `lib64`, их `ldd` не показывает missing dependencies.

Причина — `bin/rpl_run.sh`: четыре пути заранее заданы через `lib`, а отдельная
функция выбирает первый существующий `lib`/`lib64`. В этом SDK существуют оба
каталога; CMake обоих компонентов устанавливает библиотеки в **lib64**.
Исправление в `/home/aidar/rocm_sdk_builder/src_projects/rocprofiler/`:

- `CMakeLists.txt` конфигурирует wrapper с `CMAKE_INSTALL_LIBDIR` и устанавливает
  полученный скрипт, а не исходный шаблон;
- `bin/rpl_run.sh` выводит profiler/tracer/tool/metrics paths из одной настройки;
  независимый выбор по существованию каталога удалён;
- `test/test_wrapper_paths.py` проверяет поведение вызова wrapper для lib, lib64
  при наличии lib, HSA/counter mode, multiarch, absolute libdir и relocation.

Воспроизводимый patch builder:
`patches/rocm-6.1.2/rocprofiler/0002-honor-configured-library-directory.patch`.
Чистое применение в независимой копии даёт те же три файла. Чужие изменения
builder (`rocSOLVER_lapack`, `install_deps.sh`, `error.txt`) сохранены по SHA-256.

Проверки: red — **6/7 failures** на неверных путях, green — **7/7**;
четыре независимые shell path mutations — **4/4 killed**, syntax проходит.
C++/GPU production sources в этом исправлении не менялись; C++ crapper/dryer/
coverage к Bash/CMake не применимы. Сборка и установка только profiler прошли
в отдельном prefix. Реальный Q3.3 smoke дал валидный trace с **45 GPU events**
за девять запросов. Первоначальная проверка trace в artifact-driver упала из-за
строкового `dur`; исправлена обработка формата, исходный trace не менялся.
Также сохранён неуспех первой проверки patch, где `git apply` пропустил пути
из-за родительского репозитория; повтор в независимом git tree проверил байты.

Все raw/configure/build/install/test/trace команды и журнал —
`rocprof-repair-x12zi8pl/`, `rocprof-*.command.json` в каталоге этапа.

### 4.9. Реальные kernel traces всех 13 SSB

Raw и полный offline-экспорт:
[`rocprof-repair-x12zi8pl/matrix-analysis.json`](../../build/pcuda-stage8/run-8xun1k4x/rocprof-repair-x12zi8pl/matrix-analysis.json),
`matrix.json`, `matrix-tables.txt`, `trace-q*-{plain,profile}/`.
**26 процессов, 234 CPU-результата, 585 GPU kernel events.** Все 13 trace JSON
валидны, events сопоставлены с identities/порядком запусков физических планов;
девять последовательностей kernels в каждом процессе совпадают по структуре
и не перекрываются на GPU. **208 warm** имеют нулевые codegen/compiler/link/
load/symbol counters. **216** file matches и **18** прежних Q1.1 file mismatches;
все 18 отдельно совпали с CPU-oracle. Q3.3 smoke из предыдущего раздела не включён
в эти числа/статистику.

Ниже **мс**. Summary — один cold sample на режим, **не медиана семи процессов**.
Warm и fact — median [min;max], восемь повторов внутри одного процесса; это
описательная диагностика, не статистически доказанный эффект или новый SLA.
Build/post — медианы сумм соответствующих kernels **в каждом выполнении**,
не сумма медиан. Их min/max и все индивидуальные durations — в JSON.

| SQL | **Cold summary plain /rocprof**, n=1 | Warm total plain /rocprof, n=8 | Fact GPU | Build GPU /post GPU |
| --- | ---: | ---: | ---: | ---: |
| Q1.1 | **492,305 /491,180** | 12,657 [11,705;14,316] /13,638 [12,066;14,745] | 12,079 [11,166;13,407] | 0 /0,011 |
| Q1.2 | **507,578 /517,340** | 12,627 [11,604;14,453] /13,509 [12,684;15,095] | 12,079 [11,381;13,727] | 0 /0,009 |
| Q1.3 | **507,217 /497,841** | 12,465 [11,675;14,182] /14,502 [12,570;15,722] | 13,029 [11,435;14,789] | 0 /0,009 |
| Q2.1 | **761,707 /771,492** | 54,006 [53,662;54,516] /54,105 [53,902;55,559] | 53,155 [52,957;54,616] | 0,234 /0,004 |
| Q2.2 | **780,579 /768,255** | 44,326 [44,091;45,147] /44,300 [44,026;44,972] | 43,551 [43,285;43,901] | 0,133 /0,003 |
| Q2.3 | **757,535 /754,468** | 41,014 [40,752;43,212] /40,925 [40,728;41,366] | 40,166 [40,008;40,422] | 0,107 /0,003 |
| Q3.1 | **790,224 /769,681** | 24,780 [24,667;27,175] /24,907 [24,762;27,594] | 24,074 [23,965;26,507] | 0,144 /0,003 |
| Q3.2 | **773,535 /761,297** | 19,462 [19,336;20,778] /19,650 [19,363;20,869] | 18,500 [18,333;19,567] | 0,111 /0,165 |
| Q3.3 | **663,173 /668,762** | 17,095 [16,939;17,572] /17,248 [17,067;17,872] | 16,166 [16,074;16,500] | 0,077 /0,163 |
| Q3.4 | **652,734 /661,181** | 17,136 [16,836;17,792] /17,116 [16,972;17,619] | 16,079 [15,988;16,401] | 0,078 /0,165 |
| Q4.1 | **1054,841 /1040,931** | 34,721 [34,572;35,678] /34,864 [34,663;35,576] | 33,651 [33,563;34,041] | 0,400 /0,002 |
| Q4.2 | **1084,319 /1094,993** | 40,588 [40,393;41,189] /40,650 [40,517;41,283] | 39,386 [39,294;39,714] | 0,463 /0,003 |
| Q4.3 | **810,767 /811,276** | 24,688 [24,488;25,767] /24,904 [24,744;25,566] | 22,296 [22,221;22,721] | 0,332 /0,654 |

**Основной подтверждённый вывод:** fused fact scan/filter/probe/aggregate
занимает **95,8–99,9% суммы длительностей GPU kernels** в warm (медиана доли
по восьми выполнениям для каждого SQL). Это доля именно
kernel time, не всего summary: в неё не входят host submission/waits, GPU fills,
копирования, аллокации и device JIT. Устранение только compaction не может дать
многократное warm-ускорение Q3.3: сам kernel здесь около **0,163 мс**, fact —
**16,166 мс**. Для Q4.3 compaction заметнее (**0,654 мс**), но fact всё ещё доминирует.
Q1.* post kernels суммарно около 0,01 мс: их сокращение — прежде всего cold/JIT
кандидат, не объяснение текущих 12–13 мс warm.

Цена диагностического режима не скрывается: у Q1.* warm total medians выше plain
на **0,883–2,037 мс**; у групповых запросов разница от **−0,089 до +0,217 мс**.
Отрицательные разницы не объявляются ускорением от profiler; n=1 process/mode
не выделяет чистый overhead и не контролирует частоты/thermal-состояния.
Обычные baseline/SLA числа остаются из семипроцессной серии разделов 4.2–4.6.

Аппаратные DRAM bytes/cache hits/VALU stalls/atomic contention **не измерены**:
HIP-trace даёт timings/API/kernel attribution, а не HW counters. Следовательно,
ещё нельзя выбрать между memory bandwidth, runtime modulo, register pressure и
atomics как доказанной единственной причиной. Текущий ремонт подтверждает
работу HIP kernel tracing; поддержку всех counters этой старой сборкой rocprofiler
не заявляем. Проверка default CMake configuration с compatibility symlinks также
проходит в отдельных `build-default/install-default`; generated wrapper совпадает
с использованным в trace-серии.

### 4.10. Аналитическое продолжение 8-ANALYSIS

Полный разбор cold Q1.1/Q2.1/Q4.2, native build waves, границы device JIT и
service fetch перенесён без потери текста/таблиц в
[8-ANALYSIS, раздел 4.1](../pcuda_stage8_analysis/REPORT.md#41-почему-cold-q42-существенно-дороже-q11-и-q21).
Варианты дальнейшего сокращения cold и отложенный warm-backlog — раздел 6 того
же отчёта. Исходные измерения разделов 4.1–4.9 и их raw не менялись.

Согласованный S2→overlap объём завершён: прямые compiler observations, opt-in
readiness/atomic publication, обе scheduling-регрессии и отдельное подтверждение
**112 процессов/1008 CPU-результатов**. Итоги, новые quality campaigns и raw —
[8-ANALYSIS, разделы 4.4–5](../pcuda_stage8_analysis/REPORT.md#44-прямое-s2-наблюдение-полного-sql-пути).
Q4.2 observer-free plain summary **1082,910→934,905 мс**, Q1.1/Q2.1 выигрыша не
приписываем; это отдельная парная серия, не пересчёт таблиц этого отчёта.
Эксперимент остаётся default off с отдельным timing model; SQL-цели и приёмка
этапа 8 не утверждены автоматически.

## 5. Корректность

TDD red-состояния сохранены: runner не знал новые поля, cache не имел source-ready
границы, first-result observer отсутствовал. После реализации прошли реальные
host clients, а не проверки наличия строк в production sources.
Release-проверки optimizer/translator/nullable/3VL теперь также отвергают сборку
при включённом `NDEBUG`; CMake сохраняет `-UNDEBUG`.

Известный Q1.1 file mismatch сохраняется отдельно: SQL включает 1994-01-01,
ручной файл — только 1993. Независимый CPU oracle запускается заново, исходные
`success:false` и файлы эталонов не переписываются.

Свежий Q1.1 CPU scan прочитал **59 986 217** строк четырёх физических колонок:
SQL **4486422947100**, 1993-only/file **4474878955766**, вклад 1994-01-01
**11543991334**. В performance-серии **1624** file/subset matches и **126** прежних
Q1.1 file mismatches; все 126 отдельно совпали с свежим SQL CPU oracle.
Это не «1750/1750 совпадений с файлами» и не исправление исторического oracle.

Независимые Text/star oracles проверяют полные counts и каждую видимую ячейку
окна, не полную CPU-материализацию 12 млн широких строк. Manifest 88/88 означает
исполнение текущих форм, не независимый oracle каждой ячейки всех форм.
Numeric **202 попытки** включают 200 результатов и два ожидаемых отказа;
Text **131** — 125 результатов и шесть ожидаемых отказов. Отказы не выдаются
за успешно исполненный неподдерживаемый SQL.

Первая mutation-кампания честно дала **11/12**: trailing garbage проверялся только
после номера закрытого descriptor, и другой guard всё равно отклонял ввод.
Добавлен случай с **реально открытым pipe writer + `x`**. Новая независимая
кампания дала 12/12 без изменения production-семантики/порогов; исходный survivor
сохранён в `ub/host-mutants-v1/`. Исторические неудачи не превращены в kills.

## 6. Выводы и решения

Отдельные интервалы добавлены без изменения исторического timing model
`pcuda-modules-v1`. Функциональная приёмка и численные SLA — разные gates.
SQL-цели ещё не утверждены; закрытие этапа по одному успешному тесту не заявляется.

Warm-оптимизация **отложена** по решению пользователя. Полный backlog W1–W5,
прежние предложения и ограничения сохранены в
[8-ANALYSIS, раздел 6.3](../pcuda_stage8_analysis/REPORT.md#63-найденные-warm-недостатки--сохранены-оптимизация-отложена).
Это не новый обязательный объём приёмки. Cold-кандидаты и необходимые
согласования ведутся в том же отчёте; SQL-цели по-прежнему не утверждены.

Подтверждено: отдельные native PCUDA модули, ABI v2, process-only cache,
совместимые runtime values/NULL, ограниченный typed/fallback, точное one-kernel
structural reuse, функциональный Text/numeric/nullable/manual паритет и прямой
frontend budget на полном измеренном SSB-корпусе. Контроль прототипа проходит
прежние абсолютные gate; SQL Q3.3 имеет другую реализацию и другие числа.

Не заявляется: чистое время device JIT, статистически доказанный backend/string
speedup, SLA произвольного SQL/другого SF/GPU, NVIDIA/Intel validation, device
coverage/sanitizer, zero-allocation после произвольной смены SQL, новая поддержка
`future`, replay или tiering. Новых политики integer overflow/division-by-zero,
SQL join kinds, string functions/collation не добавлено.

### Исходное предложение абсолютных SQL-целей — не утверждено, отложено до оптимизации

Область: **полный db_cli SQL-путь 13 неизменённых SSB, SF10, AMD Radeon 780M,
текущий ABI v2/toolchain, готовый совместимый PCH, штатный acpp, до четырёх chains**.
Новый процесс/SQL/module/device cache на cold, не очищенные OS/HIP/driver caches;
не менее семи cold и восьми немедленных warm в каждом. Gate по plain **медианам**;
startup/data loading и PCH cost показаны отдельно, не входят в query summary.

| SQL | Предлагаемый cold **summary ≤**, мс | Предлагаемый warm **total ≤**, мс |
| --- | ---: | ---: |
| Q1.1 /Q1.2 /Q1.3 | 600 | 15 |
| Q2.1 | 850 | 60 |
| Q2.2 | 850 | 50 |
| Q2.3 | 850 | 45 |
| Q3.1 | 850 | 30 |
| Q3.2 | 850 | 25 |
| Q3.3 /Q3.4 | 750 | 20 |
| Q4.1 | 1150 | 40 |
| Q4.2 | 1200 | 45 |
| Q4.3 | 900 | 30 |

Q3.3 warm total limit предлагается также для измеренных совместимых changed
values. Warm execution измеряется отдельно, не подменяет total. Эти абсолютные
числа получены как предложение **после замеров по выбранному пользователем
порядку**, а не выданы за ранее утверждённый SLA. Все предложенные пределы на
сохранённой серии выполнены, но это само по себе не утверждение пользователем.
Пределы прототипа **600/200/10/9** и отдельный frontend ≤10 мс не изменяются.

## 7. Воспроизведение и оставшаяся работа

Новые клиенты: `test/pcuda_stage8/{validate,measure}.py`, `test_timing.cpp`,
`host-recipe.json`, `gpu-recipe.json`. Общие quality tools — `tools/ub`, без копий.
Основные reproducible команды из корня проекта, с новым `RUN`:

```bash
RUN="$PWD/build/pcuda-stage8/reproduce"
GCC=/usr/lib/gcc/x86_64-pc-linux-gnu/14.3.1
cmake -S . -B "$RUN/engine" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm_sdk_612/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC" \
  -DAdaptiveCpp_DIR=/opt/AdaptiveCpp/lib/cmake/AdaptiveCpp \
  -DCRYSTAL_PCUDA_COMPILER=/opt/AdaptiveCpp/bin/acpp \
  "-DCRYSTAL_PCUDA_FLAGS=--gcc-install-dir=$GCC" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$RUN/engine" -j4
ctest --test-dir "$RUN/engine" --output-on-failure
python3 test/pcuda_stage8/validate.py --build "$RUN/engine" \
  --output-root "$RUN" --timing-model pcuda-modules-v1
python3 test/pcuda_stage8/measure.py measure --build "$RUN/engine" \
  --output-root "$RUN" --sessions 7 --runs 9 --timing-model pcuda-modules-v1
# После ВСЕЙ серии; подставить напечатанный ssb-series-*:
python3 test/pcuda_stage8/measure.py analyze --series "$RUN/ssb-series-ID" \
  --q11-oracle "$RUN/q11-reference.stdout" --output "$RUN/analysis"
```

`q11_reference.cpp` компилируется отдельно обычным host C++17 `-O3` и запускается
на `$PWD/data/10` **вне** измерений. Полные фактические команды, prototype/PCH
серии, final checks, независимые oracles и offline tables:
`pre_performance.py`, `run_performance.py`, `offline_finish.py`, `*.command.json`,
`ssb-analysis/report.json`, `acceptance-summary.json`, `tables.txt` в каталоге
артефактов. Машинные SDK/GCC paths в командах — конфигурация стенда, не hardcode
production-кода. Старые результаты и исходные неудачи сохранены.

Quality reproduction использует `tools/ub/{coverage,crapper,dryer,mutator}.py`
с новыми output directories. GPU mutation recipe требует SDK PATH, совместимых
`CXXFLAGS`, `CRYSTAL_PCUDA_FLAGS`, `AdaptiveCpp_DIR`; его запускать отдельно от
других GPU/measurement jobs. Baseline и каждую mutation tool собирает заново.
Все argv, причины, hashes и исходники копий находятся в соответствующих
`actions.jsonl`/command files. Doxygen-комментарии в коде, HTML не генерировался.

**Остаётся для окончательного закрытия:** утвердить и проверить абсолютные
SQL-цели, записать решение в этот отчёт/roadmap. Согласованный ограниченный
S2→overlap объём выполнен в 8-ANALYSIS; default-on adoption или дальнейшие
оптимизации требуют отдельного решения. Warm-backlog отложен и не поручен.
Следующий необязательный этап 9 не выполняется.
Commit/push и очистка чужих незакоммиченных работ не выполнялись.

Воспроизведение profiler repair/диагностики: точные CMake argv —
`rocprof-isolated-configure-v2.command.json` и `rocprof-default-configure.command.json`;
build/install — одноимённые `rocprof-*.command.json`. Без изменения `/opt` рабочий
wrapper — `build/pcuda-stage8/run-8xun1k4x/rocprof-repair-x12zi8pl/install/bin/rocprof`.
GPU-independent tests:
`python3 /home/aidar/rocm_sdk_builder/src_projects/rocprofiler/test/test_wrapper_paths.py -v`.
Полный diagnostic argv/env/stdin и native process wall/user/system сохранены в
каждом `trace-q*/`; offline attribution выполняет `analyze_matrix.py` только после
сбора всех сессий. Для нового прогона использовать новый каталог, не перезаписывать
эти traces. Изолированная установка не означает, что системный wrapper уже обновлён.
