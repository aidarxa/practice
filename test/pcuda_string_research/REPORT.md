# Строки на GPU: устройство, риски и проектирование 7-TEXT {#pcuda_strings_research}

## 1. Статус и границы

**Предреализационное исследование от 2026-11-10.** Изучены первичные источники,
текущий движок, старый конвертер и реальные текстовые SF10-файлы. Ниже предложен
контракт расширения **7-TEXT**. **После исследования пользователь 2026-11-10
утвердил полную матрицу, UTF-8 Text/бинарный порядок, decoding FixedText, ABI v2
с подробным описанием и исправление custom2 (§6.4).** Последующее внедрение и
аппаратные проверки 2026-11-10–11 описываются отдельно в основном отчёте этапа 7;
настоящий документ сохраняет исходное исследование, а не выдаёт его за GPU-тест.

Это отдельный отчёт об обработке строк, а не продолжение журнала этапа 7.
Результаты внедрения и аппаратной приёмки находятся в единственном
[основном отчёте этапа 7](../pcuda_stage7/REPORT.md). Текущие задачи —
[todo.md, 7-TEXT](../../todo.md#stage7-text).
Подробный реализуемый C-контракт вынесен в [тематическое приложение ABI v2](../pcuda_stage7/TEXT_ABI.md).

**Главный вывод:** строка — не число и не `char*` с неявным `strlen`.
Для движка нужны одновременно:

1. точное правило, какие байты составляют значение;
2. отдельный признак NULL;
3. устойчивое владение байтами на GPU;
4. сравнение полного содержимого, включая разрешение хеш-коллизий;
5. перенос ссылок/row IDs между операторами вместо постоянного копирования текста;
6. ограниченная, проверяемая материализация результата.

Рекомендация для первого рабочего пути: **единый UTF-8 Text, бинарное сравнение,
компактные колонки «offsets + bytes + validity», заимствованные POD-ссылки внутри
вычислений, ABI v2 и никакого числового словаря**. Пороги и специальные ускорения
не заимствовать из CUDA/CPU-статей без измерений на PCUDA/AMD.

### Короткий словарь

| Термин | Простыми словами |
| --- | --- |
| Логический тип | Что означает SQL-значение и какие операции над ним разрешены |
| Физическое представление | Как байты, длины/смещения и NULL размещены в памяти |
| Offset | Начало строки в общем буфере; обычно измеряется **в байтах** |
| String view / ссылка | Указатель или идентификатор буфера со смещением и длиной; не владелец текста |
| Validity | Отдельная битовая карта: действительное значение или SQL NULL |
| Coalescing | Соседние GPU-потоки обращаются к близким адресам, уменьшая число транзакций памяти |
| Divergence | Потокам одной аппаратной группы требуется разная работа; часть потоков простаивает |
| Late materialization | Сначала выбираем/соединяем/сортируем номера строк, текст копируем ближе к выдаче |
| Collation | Правила равенства и порядка строк, например с учётом языка или без учёта регистра |

## 2. Окружение, источники и исходная архитектура

### 2.1. Доказательства и воспроизводимость

Артефакты этой работы:
[`build/pcuda-stage7-text/preflight-cwwf566k`](../../build/pcuda-stage7-text/preflight-cwwf566k/).

- `initial-status.txt`, `initial.diff`, `initial-head.txt` сохраняют начальное
  состояние рабочего дерева. Чужие незакоммиченные изменения не отменялись.
- `initial-sha256.json` содержит 113 исходных hashes выбранных файлов проекта.
- `sources/manifest.json` содержит URL, effective URL, тип, размер и SHA-256
  **14 сохранённых первичных источников**. Их исходные HTML/PDF/Markdown сохранены
  рядом. Эти HTML получены с сайтов, **не сгенерированы Doxygen**.
- `profile_text.py`, `data-profile.json` — независимое чтение физических файлов,
  с hashes полностью прочитанных колонок либо отобранных записей.
- `actions.jsonl` — журнал действий и причин.

Версия PCUDA рассматривается по документу AdaptiveCpp commit
`5989e1cfa4e70d55c65e786a4c71acb124ce54ff`, использованному предыдущими этапами,
а не по обещаниям движущегося `develop` [S10]. Новая компиляция, запуск GPU,
обновление toolchain или проверка NVIDIA/Intel в этом исследовании не проводились.
AMD Radeon 780M — аппаратная цель предстоящей приёмки по правилам проекта,
не уже проверенное здесь текстовое исполнение.

### 2.2. Почему нельзя просто разрешить текст в translator

В исходном состоянии кода на момент предреализационного аудита 2026-11-10:

| Компонент | Нынешнее предположение, которое придётся расширить |
| --- | --- |
| `include/storage/ssb_schema.h` | Текстовые физические ширины уже известны и отделены от чисел |
| `include/storage/column_loader.h` | Production loader намеренно принимает только числовое хранение |
| `include/core/memory.h` | Логические выходы — Int64, UInt64 и Float64; text gate не позволяет ошибочное lowering |
| `include/core/parameter_value.h`, `src/core/query_template.cpp` | Параметры и контекстный вывод типов пока числовые |
| `src/core/expression_codegen.cpp` | CASE и операции опираются на числовые типы и helpers |
| `include/core/module_abi.h` | ABI v1 содержит 8-байтный scalar; нельзя подменить его смысл строковой ссылкой |
| `include/crystal/join.h`, `src/core/visitor.cpp` | PHT/MHT и часть группировки используют perfect-domain числовых ключей |
| `include/core/execution.h`, `src/db/database_instance.cpp` | Владельцы base columns и типизированные результаты пока числовые |
| `src/cli/terminal_app.cpp` | Публичная печать должна получить настоящий Text, не GPU-адрес или число |

**Необходим рефакторинг границ типов и результатов**, но не переписывание всех
числовых операторов. Старые быстрые SSB/PHT-пути надо сохранить. Общая семантика
Text должна жить в одном месте, а не размножаться в translator, optimizer, visitor,
CLI и независимых друг от друга строковых ветках.

## 3. Методика исследования

1. Сопоставлены стандарт формата Arrow, устройство libcudf, опыт CPU-движков,
   исследование GPU-операторов и работа по неоднородным строковым нагрузкам 2026 года.
2. Для каждого источника отделена переносимая идея от конкретного CUDA API,
   NVIDIA warp size, CPU calling convention и авторских performance-результатов.
3. Проверены фактические assumptions проекта и код старого конвертера.
4. Независимый Python-скрипт прочитал **все записи 16 текстовых dimension-файлов**
   и по **10 000 равномерно расположенных записей** двух fact-файлов, включая
   первую и последнюю. Всего **4 492 785 записей колонок**. Это не число уникальных
   строк таблиц: одна строка проверяется отдельно для каждой текстовой колонки.
5. Полные размеры файлов сверены с ожидаемыми `rows × width`. Для наблюдений
   отдельно посчитаны префикс до первого NUL, хвост после него, UTF-8, длины,
   завершающий пробел и число различных префиксов. Интерпретация префикса —
   **исследуемая гипотеза формата, не заранее утверждённая SQL-семантика**.

Скрипт не импортирует production loader, schema или device helpers. Он не меняет
данные и не измеряет время. Поэтому здесь **нет** performance-таблиц, SQL summary,
cold/warm-медиан или утверждения о превосходстве выбранного layout. Исследование
документации и формы данных не заменяет аппаратное сравнение реализаций.

## 4. Результаты: что делает строки трудными и какие практики полезны

### 4.1. Стоимость зависит не только от количества строк

Сравнить два Int32 обычно означает загрузить два значения и выполнить одну
операцию. Для строки сначала нужно найти байты, затем определить равенство или
первый различающийся байт. Стоимость зависит от:

- длины и распределения длин, а не только среднего;
- общего префикса: имена `Customer#000000001` и `Customer#000000002` отличаются поздно;
- селективности: большинство несовпадений можно отклонить по длине/первым байтам;
- повторяемости значений и числа групп;
- расположения памяти, cache locality и порядка row IDs после join/sort;
- необходимости копировать результат и multiplicity join;
- правил Unicode/collation.

Для равенства разные длины сразу означают «не равно». Для лексикографического
порядка **нельзя сначала сортировать по длине**: `"z"` идёт после `"aa"` при
бинарном порядке. Длина решает порядок только после равного общего префикса.

### 4.2. Сравнение физических представлений

| Представление | Сильная сторона | Цена и риск | Решение для 7-TEXT |
| --- | --- | --- | --- |
| `char*` с NUL, отдельная аллокация на значение | Привычно для C | Неизвестные границы, `strlen`, pointer chasing, много allocations | Не использовать |
| Fixed-stride bytes + длина строки | Простое обращение к существующему fixed-width файлу | Передаём/храним padding и исторический мусор, хуже общий variable-length путь | Контрольная альтернатива при сравнении layout |
| Compact offsets + bytes + validity | Плотные байты, явные границы, обычная схема Arrow/cuDF | Gather требует либо копирования байтов, либо отдельного массива ссылок | Рекомендуемое базовое владение колонкой |
| Заимствованный pointer+length | Дешёвый CASE, перестановка ссылок без копирования payload | Каждая ссылка требует живого владельца; host не может разыменовать device pointer | Рекомендуемое промежуточное значение |
| 16-byte German/Arrow views | Короткие строки внутри записи, prefix fast reject, дешёвый gather | Больше metadata для tiny strings, особое владение и удержание исходных buffers | Кандидат для отдельного контроля, не объявленный победитель |
| Dictionary IDs | Эффективно для малого словаря | Перекодировка join, отдельная семантика порядка, цена словаря | Не внедрять: пользователь уже отверг подмену строк кодами |
| Сжатие, например substring/symbol-table encoding | Меньше bytes/I/O | Декодирование, сложные сравнения и отдельная приёмка | Не включать автоматически |

**Важное различие:** owning column и transient value не обязаны иметь одинаковый
layout. Колонка может быть компактной, а оператор — брать из неё маленькую ссылку.
При этом не нужно держать по 16-байтному view на каждую исходную строку постоянно.

По Arrow [S01] строка `i` занимает `[offsets[i], offsets[i+1])`, offsets монотонны,
а validity независима от длины. `['', NULL]` может иметь offsets `[0,0,0]`,
но значения различаются validity. Нулевой pointer или нулевая длина сами по себе
не являются NULL. Строка UTF-8 и бинарные bytes имеют похожий layout, но **разную
семантику и требования к проверке входа**.

### 4.3. 32-битные offsets, 64-битные размеры и граница 2 GiB

Arrow Utf8 использует signed Int32 offsets, LargeUtf8 — Int64. Первая схема
ограничивает **общий буфер bytes колонки**, а не только длину отдельной строки
[S01]. libcudf расширил поддержку больших строковых колонок выбором 32/64-bit
offsets; старый пример 2022 года нельзя читать как актуальное ограничение
всего libcudf [S02, S04]. Аналогично схема «два child columns» в старой статье
объясняет идею, но не фиксирует нынешний C++ API владения chars.

Предлагается:

- считать rows, суммы bytes, ёмкости и произведения в checked 64-bit/`size_t`;
- хранить base-column offsets в Int32 только при доказанном `total_bytes <= INT32_MAX`,
  иначе в Int64; ширина offsets относится к **физическому каталогу**, не к значениям SQL;
- нормализовать чтение offset к 64 битам;
- для query-produced compact output использовать 64-битные offsets либо заранее
  подготовленный runtime dispatch: смена длины литерала не должна вызывать codegen;
- не обещать поддержку колонки больше доступной GPU-памяти только потому,
  что offset умещается в Int64.

Для нынешних 18 SF10 text columns имеется **124 445 219 физических слотов**.
Простая арифметика metadata, без validity/payload/allocator overhead:

| Форма | Байты | Значение |
| --- | ---: | --- |
| По одной Int32 length на слот | 497 780 876 | Fixed-stride альтернатива |
| Int32 offsets, включая по одному конечному на колонку | 497 780 948 | При допустимом размере каждой колонки |
| Int64 offsets | 995 561 896 | Почти 1 GB metadata ещё до самих символов |
| По одному 16-byte view на каждый исходный слот | 1 991 123 504 | Не оправдывать простотой реализации без сравнения |

Это **расчёт размеров**, не измерение throughput. Именно поэтому не рекомендуется
механически назначать 64-битные offsets или German views всем исходным значениям.
64-битная арифметика границ не требует удваивать хранилище каждого малого offset.

### 4.4. Одна нить на строку или несколько нитей на строку

Для коротких относительно равномерных строк один поток на значение — полезный
простой baseline. Для длинных строк совместная работа группы потоков лучше
загружает память. Но единая стратегия для сильно неоднородных длин может
приводить к простаиванию потоков.

HetMatch (ADMS 2026) [S08] разделяет **индексы**, а не копирует строки:
короткие строки обрабатывает thread-per-string, длинные — warp-per-string.
В работе порог 128 bytes, baseline cuDF 25.10 и NVIDIA H100 PCIe; авторы исключали
первый cold run и показывали медианы пяти trials. Это исследование substring
`contains`, **не** аппаратная приёмка нашего равенства, SQL latency или PCUDA.

Из него следует полезный принцип «смотреть распределение длин», но не следует
универсальная магическая граница 128, warp=32 или необходимость внедрить LIKE.
В текущих SF10 text files ширина не превышает 25 bytes. Поэтому обязательная
диспетчеризация на несколько length buckets до первого работающего baseline
не обоснована. Длинные runtime-литералы всё равно должны корректно обрабатываться.

PCUDA данной версии не знает целевой GPU при stage-1 compilation; `warpSize`
не `constexpr` [S10]. HIP также требует не предполагать одну ширину аппаратной
группы [S14]. CUDA-код с жёстким warp=32, intrinsics или CUB/Thrust нельзя
просто включить в portable PCUDA engine.

### 4.5. Аллокации, fusion и создание результата

Практический разбор libcudf [S02] показывает, почему `malloc/free` на каждую
строку внутри GPU-ядра — плохая отправная точка. Более подходящая схема:

1. одним проходом получить длины выходных строк;
2. prefix-scan длин даёт offsets и точное число bytes;
3. runtime проверяет budget и обеспечивает общую ёмкость;
4. отдельный проход записывает bytes в непересекающиеся диапазоны.

Два прохода могут быть выгоднее одного прохода с тысячами мелких allocations.
Но в projection, CASE и join, **не создающих новых символов**, часто не надо
копировать bytes даже два раза: достаточно ссылок на исходные значения.

Рекомендации для проекта:

- сохранять fused scan/filter/probe; не создавать kernel на каждый SQL-предикат;
- CASE выбирает Text view и validity, а не выделяет новую строку;
- sort перемещает permutation/row IDs, а не заново пишет chars на каждом шаге;
- bytes собираются при окончательной материализации/CPU fetch;
- count/scan/write для bytes и необходимые size readback являются явными
  операциями execution plan; это обоснованный pipeline breaker;
- использовать существующие RAII, memory ledger, pinned staging и high-water
  буферы, не вводить собственный allocator в сгенерированные модули;
- предел display window не разрешает получить неверный полный `row_count`.

При длинных строках и размножающем join выход может оказаться намного больше
входа. `input_bytes × selectivity` — не безопасная оценка ёмкости. Нужны точные
checked суммы длин **для фактически выдаваемых строк**.

### 4.6. Что дают German strings и чего они не гарантируют

Umbra/CedarDB и Arrow View используют 16-байтную запись: длину, короткую строку
внутри либо prefix и ссылку на длинную [S01, S05]. В Arrow граница inline —
**до 12 bytes включительно**, не 12 Unicode-символов. Prefix помогает быстро
отвергать несовпадения; полный длинный текст всё равно проверяется.

Преимущество 16-byte значения в CPU calling convention не доказывает выигрыш
в register pressure или occupancy GPU. Для имени `Customer#...` четырёхбайтный
prefix почти одинаков у всех строк и мало помогает. Для коротких значений
дескриптор может оказаться больше offsets+payload.

Опыт DataFusion [S06, S07] важен именно своими ограничениями: простой перенос
StringView сначала давал регрессии; существенными оказались UTF-8 validation,
скрытые копии, inlining и удержание больших исходных buffers после фильтра.
Избежать копии — не значит избежать расхода памяти.

В этом in-memory движке base columns и так живут всю сессию. Заимствовать их
байты удобно. Но временную parameter arena нельзя перезаписать до завершения
GPU-работы и fetch результата: `CASE ... THEN 'literal' ...` может вернуть ссылку
именно на неё. High-water realloc также не должен инвалидировать опубликованные
views до подтверждённого completion.

### 4.7. GROUP BY и join: хеш — это фильтр кандидатов, а не значение строки

У строк нет удобного числового `[min,max]`-домена. Нельзя:

- подать hash вместо полного ключа в нынешний perfect-domain PHT;
- считать совпадение hash доказательством равенства;
- группировать по адресу: одинаковый текст в разных буферах — одна SQL-группа;
- атомарно публиковать многопольный Text descriptor как обычный Int64;
- потерять duplicate keys или NULL-семантику при расширении числового MHT.

Практические варианты:

| Операция | Проверяемый начальный алгоритм | Обязательная защита корректности |
| --- | --- | --- |
| String equijoin | Hash buckets с row IDs, count/scan/write и проверкой candidates | Полное равенство bytes, nonunique multiplicity, NULL не совпадает |
| String GROUP BY | Таблица представителей immutable keys и отдельные aggregate states | Коллизии сравниваются по полным ключам; все NULL одного ключа образуют одну группу |
| Mixed/multi-column keys | Последовательный hash и сравнение компонент с явными границами | `('ab','c')` не равно `('a','bc')`; нельзя слепо конкатенировать bytes |
| GROUP BY Text CASE | Ключом является выбранное значение | Равные ветви объединяются независимо от branch ID и адреса |
| MIN/MAX(Text), если согласованы | Редукция/выбор row ID, ссылающегося на полный Text | Не атомарная запись pointer+length; отдельная presence/validity |

Особенно важен concurrent insertion: ожидание публикации другой нитью через
spin-lock может быть непереносимо. Более простой baseline — сначала сформировать
immutable key views на GPU, затем CAS публикует **один row ID**, а сравнение
читает уже готовые ключи. Агрегация/уплотнение идут на явных границах stream.
Нужны ограниченное probing, detection переполнения и корректный отказ/рост,
не потенциально бесконечный цикл при заполненной таблице.

Внутренние row IDs и номера aggregate states не являются запрещённой словарной
подменой: SQL-равенство определяется полным Text, и наружу возвращается Text.
Не нужно создавать таблицу «BUILDING = 1», чтобы использовать row IDs.

Работа по GPU join/group-by [S09] подчёркивает стоимость случайного gather и
материализации. Но её основные эксперименты используют 4/8-byte данные:
переносить численные выигрыши непосредственно на variable-length strings нельзя.
Поздняя материализация — хороший baseline, а не догма «gather всегда бесплатно».
Для high-match wide join могут понадобиться последующие locality-оптимизации.

### 4.8. ORDER BY и Top-K

Одинаковый полный comparator должен использоваться в WHERE, sort, выборе
MIN/MAX и при группировке там, где сравниваются значения. Для ORDER BY:

- сравнивать bytes как **unsigned**, не implementation-defined signed `char`;
- после равного общего префикса более короткая строка идёт раньше;
- hash не является сортировочным ключом;
- prefix — только ускорение: равные prefixes требуют продолжения сравнения;
- NULL ordering должен быть совместим с действующим числовым путём и закреплён
  отдельными тестами ASC/DESC;
- равные ключи без дополнительного ORDER BY не требуют произвольного нового
  обещания стабильности, но LIMIT должен возвращать допустимые строки;
- план должен содержать готовый comparator-based путь и runtime-выбор размеров,
  а не новую компиляцию при смене LIMIT или длины параметра.

Нельзя использовать quadratic «сравнить каждую пару» как молчаливый общий
production fallback или сортировать на CPU под видом PCUDA-оператора.

### 4.9. Unicode, пробелы, NUL и SQL NULL — четыре разные проблемы

UTF-8 поддерживает не только ASCII. Байт, Unicode code point и видимый символ
(grapheme) — не одно и то же. Для валидного UTF-8 бинарный unsigned-byte порядок
совместим с порядком последовательностей code points, но **не** с алфавитным
порядком конкретного языка. Без нормализации `é` и `e` + combining accent —
разные строки. Без case folding `A` и `a` — разные строки.

Даже у развитого libcudf отдельные Unicode-функции имеют явно документированные
ограничения [S13]. Поэтому поддержка UTF-8 хранения/сравнения не означает
готовые locale-aware collation, LOWER/UPPER, regex или длину в graphemes.

PostgreSQL различает blank-padded CHAR и сохраняющие пробелы TEXT/VARCHAR [S11].
SQLite BINARY сравнивает bytes; RTRIM — отдельное правило [S12]. Из одного
физического fixed-width файла **не следует**, что логический SQL-тип обязан быть
CHAR и игнорировать trailing spaces.

Предлагается один Text, а не два частично реализованных CHAR/VARCHAR:

| Вопрос | Предлагаемая семантика, до согласования |
| --- | --- |
| Encoding | Валидный UTF-8; ASCII является его подмножеством |
| Равенство/порядок | Полные unsigned bytes; case-sensitive; без locale/canonical normalization |
| Пробелы | Все пробелы внутри логического значения значимы, включая завершающий |
| Пустая строка | Действительное значение длины 0, не NULL |
| SQL NULL | Отдельная validity; обычные сравнения с NULL дают UNKNOWN |
| NUL в physical FixedText | Первый NUL завершает значение **внутри текущей ширины**; последующий хвост игнорируется |
| Physical field без NUL | Все `width` bytes являются значением; чтения за границей нет |
| NUL внутри логического Text/литерала | Предлагается отклонять; physical NUL не попадает в Text |
| Невалидный UTF-8 | Явная ошибка на входной границе, не замена на U+FFFD и не молчаливое усечение |
| Payload SQL NULL | Проверить physical extent/bitmap, но не интерпретировать замаскированные bytes как значение |
| Text + число | Явная ошибка типов, без словаря, `atoi` или скрытого cast |
| Ограничение длины | Предлагается `<= INT32_MAX` bytes на значение; суммарные размеры checked 64-bit и ограничены memory guard |

Предельное число bytes не является обещанием, что на стенде можно выделить
строку такого размера. Пределы row IDs/алгоритмов тоже остаются отдельными.

## 5. Корректность: фактический формат SF10 и его неоднозначности

### 5.1. Что обнаружено в файлах

Полный объём 18 физических text files — **1 637 979 720 B (1,525487 GiB)**.
Два fact files дают 1 559 641 642 B; 16 dimension files — 78 338 078 B.

Длины ниже — число bytes **до первого NUL либо вся ширина**, не согласованная
CHAR-длина. `Хвост` — количество записей, у которых после первого NUL есть
ненулевые bytes. Эти bytes не включались в UTF-8 проверку префикса.

| Файл / поле | Проверено | Ширина | Длины | Хвост | Без NUL на всю ширину |
| --- | ---: | ---: | --- | ---: | ---: |
| LINEORDER6 / orderpriority | 10 000, sample | 16 | 5–15 | 8 055 | 0 |
| LINEORDER16 / shipmode | 10 000, sample | 10 | 3–7 | 8 569 | 0 |
| SUPPLIER1 / name | 20 000, все | 25 | 18 | 20 000 | 0 |
| SUPPLIER2 / address | 20 000, все | 25 | 6–24 | 17 161 | 0 |
| SUPPLIER6 / phone | 20 000, все | 15 | 15 | 0 | 20 000 |
| CUSTOMER1 / name | 300 000, все | 25 | 18 | 300 000 | 0 |
| CUSTOMER2 / address | 300 000, все | 25 | 6–24 | 256 699 | 0 |
| CUSTOMER6 / phone | 300 000, все | 15 | 15 | 0 | 300 000 |
| CUSTOMER7 / mktsegment | 300 000, все | 10 | 8–10 | 14 956 | 59 587 |
| PART1 / name | 800 000, все | 22 | 7–21 | 754 834 | 0 |
| PART5 / color | 800 000, все | 11 | 3–10 | 621 989 | 0 |
| PART6 / type | 800 000, все | 25 | 16–25 | 573 667 | 10 617 |
| PART8 / container | 800 000, все | 10 | 6–10 | 373 843 | 59 714 |
| DDATE1 / date | 2 557, все | 18 | 11–18 | 1 662 | 147 |
| DDATE2 / dayofweek | 2 557, все | 8 | 6–8 | 1 096 | 1 096 |
| DDATE3 / month | 2 557, все | 9 | 3–9 | 1 536 | 210 |
| DDATE6 / yearmonth | 2 557, все | 7 | 7 | 0 | 2 557 |
| DDATE12 / sellingseason | 2 557, все | 12 | 4–9 | 1 886 | 0 |

У **4 676 customer addresses и 303 supplier addresses** префикс заканчивается
пробелом. Безусловный `rtrim` изменит эти значения. Пустых и невалидных UTF-8
префиксов в проверенной области не обнаружено; это **не** тест Unicode, NULL
или всех 59 986 217 fact rows. Для них необходимы отдельные fixtures/полная загрузка.

В `bench/loader/load.c` и `load_modified.c` встречается запись всего `char[N]`
после `strcpy` без очистки всего массива. Наблюдаемый хвост согласуется с такой
реализацией; полное происхождение каждого файла этим не доказано. Сам старый
конвертер также не является образцом безопасной обработки полноширинных строк:
его небезопасные операции не должны переноситься в новый C++ loader.

Следствия:

- требование zero-tail несовместимо с имеющимися файлами;
- требование обязательного NUL тоже несовместимо с файлами;
- первый NUL надо искать bounded, не читать следующую строку;
- если принять предлагаемую семантику, хвост отбрасывается на storage boundary,
  а не снова разгадывается в каждом SQL-операторе;
- полное чтение/проверка physical extent сохраняется: игнорировать хвост значения
  не значит читать только первые четыре bytes файла;
- после нормализации bytes всех 16 dimensions занимают **55 497 580 B** без
  offsets/validity. Это полное наблюдение для dimensions, не экстраполяция fact.

### 5.2. Конфликт `custom2.sql`

`olap_bench/sql/08_custom/custom2.sql` содержит `c_mktsegment = 1` и `= 2`.
Фактически CUSTOMER7 содержит:

| Text | Строк |
| --- | ---: |
| AUTOMOBILE | 59 587 |
| BUILDING | 60 359 |
| FURNITURE | 60 159 |
| HOUSEHOLD | 59 922 |
| MACHINERY | 59 973 |

В `bench/loader/convert.py` описано **опциональное** прежнее кодирование
AUTOMOBILE=0, BUILDING=1, FURNITURE=2 и т. д. Эти Int32-коды не являются
содержимым нынешнего CUSTOMER7.

Предложение на согласование: заменить в **этом запросе** `1/2` на
`'BUILDING'/'FURNITURE'`, оставить категорию manifest действующей и получить
новый независимый oracle по настоящему Text. Не менять исходные data files,
не вводить скрытый словарь в engine и не переписывать исторические raw failures.
Без согласования нельзя честно объявить эту форму выполненным Text SQL.

### 5.3. План независимых проверок

| Область | Что должно быть проверено при внедрении |
| --- | --- |
| Storage | Exact file sizes, bounded NUL, dirty tail, полная ширина, значимые пробелы, invalid UTF-8, nullable payload |
| Значения | `''`, NULL, одинаковые bytes по разным адресам, разные длины, общий длинный prefix, апостроф, кириллица, emoji, composed/decomposed Unicode |
| WHERE | Все шесть сравнений, IS NULL/IS NOT NULL, AND/OR/NOT и 3VL |
| CASE | Text/Text и Text/NULL, разные источники bytes, совпавшие значения ветвей, переходы длин параметра |
| Группировка | Text и mixed keys, NULL-группа, нулевые/отрицательные числовые aggregates, принудительные hash collisions |
| Join | Уникальные и duplicate keys, many-to-many, NULL, разные bytes одного hash, OR overlap без лишнего дублирования пары |
| Сортировка | Полный текст, общий prefix, ASC/DESC, NULL, multi-key, LIMIT 0 и переходы алгоритмов |
| CPU result | Тип Text и реальное содержимое, пустое против NULL, частичное display window, отсутствие device pointers в публичных значениях |
| Reuse | Повтор/смена Text bytes, длины и typed NULL в одной сессии; нулевые приращения codegen/acpp/link/load/symbol counters после подготовки |
| Ошибки | ABI version/schema/length, checked sums, memory guard, безопасный следующий запрос и завершение сессии |
| Паритет | 13 SQL SSB, весь действующий manifest, 13 независимых manual SSB, nullable/numeric/MHT/OR fixtures |

Ожидаемые строки должны строиться независимым Python/SQLite oracle после
нормализации физических полей по согласованному контракту. SQLite BINARY
подходит для Text/Text, но его dynamic typing не является oracle для предлагаемого
строгого отказа Text/число. NULL ordering также надо задавать согласно engine,
а не случайно наследовать иной default SQLite. Raw outputs и исторический
Q1.1 mismatch сохраняются.

## 6. Выводы, проектное предложение и открытые решения

### 6.1. Что подтверждено, а что ещё гипотеза

**Подтверждено исследованием:** реальный формат имеет dirty tails, полноширинные значения и
значимые завершающие пробелы; исходная типизация/ABI/PHT не поддерживали
настоящий Text; авторитетные реализации отделяют bytes, границы, NULL и владение.

**Обоснованные проектные рекомендации:** одна семантика Text, компактные buffers,
заимствованные views, collision-safe ключи, GPU-resident intermediates, runtime-owned
parameter arena и точная финальная материализация.

**Не подтверждено измерениями:** лучший layout именно на AMD/PCUDA; выигрыш
German views; оптимальные block size/length thresholds; стоимость UTF-8 validation;
соотношение hash/sort group-by; абсолютные Text cold/warm latency. Нет утверждения,
что чужие CPU/CUDA speedups распространяются на этот движок.

### 6.2. Предлагаемая архитектура реализации

```text
FixedText files + validity
  → строгая storage boundary: extent, bounded NUL, UTF-8
  → RAII-owned GPU bytes + offsets + validity
  → типизированные scan/filter/CASE/join/group/sort над Text views
  → выбранные row IDs / Text views остаются на GPU
  → размеры bytes → checked scan → runtime capacity → GPU gather
  → типизированный CPU Text-результат → CLI formatting
```

Ответственность разделяется так:

- **Storage** один раз определяет логические bytes и проверяет внешний ввод.
- **Frontend/type layer** выводит Text/NULL и отклоняет несовместимые операции;
  значения не передаются optimizer для специализации.
- **Device primitives** предоставляют full compare/hash/view access без allocation,
  SQL parser и тяжёлых host headers.
- **Planner/codegen** выбирают структурный путь и явные операции, не владеют памятью.
- **Runtime** владеет base/result/scratch/parameter bytes, staging и stream.
- **Fetch/CLI** получают Text и печатают его; они не выполняют WHERE/join/group/sort.

### 6.3. ABI v2 — предлагаемая область контракта

ABI v1 не переинтерпретируется. Предлагается **v2 для всего нового native SQL-пути**
с фиксированными `db_kernel_descriptor_v2`, `db_kernel_launch_v2` и identity v2.
Числовая SQL-семантика остаётся прежней; старые v1 fixtures не переписываются
задним числом. Обязательство runtime запускать произвольные старые v1 modules
не вводится: SQL module cache и так process-only.

Предлагаемая строковая единица значения: **16-byte POD** из device pointer и
64-bit byte length, без владельца, STL, NUL-терминатора и признака NULL в указателе.
Validity передаётся отдельно. Конкретные offsets/layout assertions всех v2 structs
необходимо зафиксировать в C11/C++ tests **до подключения generated wrappers**.

Схема аргументов явно различает:

1. byte buffers и offsets Int32/Int64 базовой колонки, их реальные extents;
2. bitmap validity;
3. Text scalar/parameter view с device pointer, byte length и отдельной validity;
4. массивы промежуточных Text views, row IDs и compact-output offsets/bytes.

Wrapper копирует значения в device closure, не передаёт GPU указатель на host
invocation/`std::string`. Runtime проверяет extents и соответствие allocation
owners; сама структура offsets проверяется при создании колонки. Descriptor
проверяет schema/version, но не заменяет проверку содержимого внешнего файла.

Parameter bytes загружаются в reusable arena **перед** kernel launch в тот же
stream; arena и её staging живут до завершения потребителей. Значение, длина,
validity, адреса и capacity не входят в module key. ABI, семантика Text/сравнения,
schema и существенные layout/toolchain settings входят. Числовой bounded cache
8 typed + tagged fallback не превращается в комбинаторный набор специализаций
по строковой длине или содержимому.

### 6.4. Предлагаемая функциональная область для согласования

В рамках 7-TEXT предлагаются projection/star, литералы, WHERE/NULL/3VL и шесть
сравнений, Text CASE, GROUP BY (включая Text CASE/mixed keys), числовые aggregates
над выбранными строками, COUNT(Text), MIN/MAX(Text), HAVING, ORDER BY/LIMIT и
поддерживаемые inner equijoins/OR joins с Text. Text SUM/AVG, арифметика Text,
неявные Text/number conversions явно запрещены.

LIKE/ILIKE, regex, concatenation, substring/length/case functions, locale collation,
Unicode normalization, новые join kinds, новый SQL DDL/import API, compression,
persistent compiled cache и out-of-core execution автоматически не добавляются.
Если пользователь выберет меньшую область, оставшиеся пункты будут открытыми,
а не объявленными завершёнными.

**Явное решение пользователя 2026-11-10:** принять полный набор §6.4, единый
UTF-8 Text и все правила таблицы §4.9, bounded FixedText из §5, перейти на ABI v2
с его подробным описанием и исправить custom2 на BUILDING/FURNITURE. Это ответы
на пять отдельных вопросов, не вывод из молчания. Выбор оптимального layout/
порогов по производительности по-прежнему не подтверждён измерениями.

## 7. Воспроизведение, дальнейшие эксперименты и источники

### 7.1. Повторить независимую проверку формы данных

Из корня проекта, при наличии сохранённого скрипта артефактов, с **новым** output:

```bash
python3 build/pcuda-stage7-text/preflight-cwwf566k/profile_text.py \
  --data data/10 --output build/pcuda-stage7-text/profile-recheck.json
```

Ширины в скрипте выписаны независимо из старой schema; полный extent ожидается
строго. Файлы LINEORDER проверяются sample, dimensions — целиком. Hash sample
не является hash полного fact-файла. Артефакты `build/` локальны, не гарантируются
доступными после переноса репозитория. Полная методика и наблюдения сохранены выше.

План, составленный до внедрения (фактические результаты — в основном отчёте этапа 7):

1. Red→green host-тесты значения/loader/типизации/ABI; Doxygen-комментарии без HTML.
2. Вертикальный GPU-срез Text filter/projection/typed fetch, затем CASE/sort,
   далее collision-safe group/join; числовой паритет после каждого среза.
3. Локальные `tools/ub/{coverage,crapper,dryer,mutator}.py` для выбранных реальных
   изменённых C++ clients, свежие independent mutation baselines, raw logs.
    Host coverage не выдаётся за device coverage. Само исследование не является
    quality-кампанией изменённых позднее C++-компонентов.
4. Native PCUDA tests и реальный `db_cli` с независимыми Text oracles, counters,
   safety errors, всем действующим manifest и 13 SQL/manual SSB.
5. Если нужно выбрать layout по производительности: одинаковые данные и SQL,
   compact offsets против fixed-stride/length и дополнительного German-view
   кандидата. Не сравнивать разные алгоритмы как эффект одного layout.

Матрица для будущего сравнения должна включать число rows **и** bytes, длины
0/1/7/12/13/25/128/длинные, различие первого/последнего байта, длинные общие
prefixes, много повторов и high cardinality, NULL, low/high match, wide join,
projection и byte materialization. Измерять peak resident bytes, allocations,
запуски, kernel/execute/fetch и полный SQL **summary** раздельно. Не считать
throughput сканирования bytes скоростью полного запроса.

Для performance-серий действуют [AGENTS.md](../../AGENTS.md) и
[timing.md](../../timing.md): native Bash/GNU time, C++ steady_clock,
Clang/LLD traces и контроль без traces, private device cache на cold process,
raw samples/порядок/разброс, oracle вне интервала. Полный непрерывный summary
считается до готового CPU Text result; formatting/печать не входят. Загрузку,
нормализацию и H2D base columns нельзя спрятать: при eager load они отмечаются
отдельно до data-ready, при query-time работе входят в SQL-интервал.
Цели прототипа 600/200/10/9 ms на строки не переносятся.

### 7.2. Первичные источники

Прочитаны и сохранены в `sources/`; актуальность чтения — 2026-11-10.
`latest`-страницы могут меняться: содержание и SHA-256 фиксирует manifest.

| ID | Источник | Для чего использован / граница |
| --- | --- | --- |
| S01 | [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html), format 1.5 | Bytes/offsets/validity, Utf8/LargeUtf8 и View; не обещание Arrow interop движка |
| S02 | [Mastering String Transformations in RAPIDS libcudf](https://developer.nvidia.com/blog/mastering-string-transformations-in-rapids-libcudf/), 2022 | Preallocation, fusion и sizes/scan/write; API и timings исторические, NVIDIA A6000 |
| S03 | [libcudf Strings Classes](https://docs.nvidia.com/cudf/latest/libcudf/api_docs/strings_classes/) | Non-owning device views, bytes против characters, lifetime |
| S04 | [Scaling Up to One Billion Rows with cuDF](https://developer.nvidia.com/blog/processing-one-billion-rows-of-data-with-rapids-cudf-pandas-accelerator-mode/), 2024 | Раздельные пределы rows/bytes и выбор ширины offsets; не рекомендация включить managed memory |
| S05 | [CedarDB: Why German Strings are Everywhere](https://cedardb.com/blog/german_strings/), 2024 | Inline bytes, prefix и borrowed lifetime; CPU-опыт не доказывает GPU-win |
| S06 | [DataFusion StringView, part 1](https://datafusion.apache.org/blog/2024/09/13/string-view-german-style-strings-part-1/), 2024 | Цена UTF-8 validation, hidden copies и интеграции; CPU |
| S07 | [DataFusion StringView, part 2](https://datafusion.apache.org/blog/2024/09/13/string-view-german-style-strings-part-2/), 2024 | View gather, memory retention, GC и реальные отрицательные случаи; CPU |
| S08 | [Kim, Yu: HetMatch](https://vldb.org/2026/Workshops/VLDB-Workshops-2026/ADMS/ADMS26_8.pdf), ADMS 2026 | Length skew и heterogeneous dispatch; contains/cuDF 25.10/H100, не весь SQL |
| S09 | [Wu, Koutsoukos, Alonso: Efficiently Processing Joins and Grouped Aggregations on GPUs](https://arxiv.org/html/2312.00720v2), 2025, DOI 10.1145/3709689 | Materialization/random gather и workload dependence; основные опыты не variable-length Text |
| S10 | [AdaptiveCpp PCUDA, pinned commit](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/5989e1cfa4e70d55c65e786a4c71acb124ce54ff/doc/pcuda.md) | Native launch, generic JIT, supported atomics и отсутствие compile-time target |
| S11 | [PostgreSQL 18: Character Types](https://www.postgresql.org/docs/18/datatype-character.html) | Различие CHAR/TEXT padding, длина в characters/bytes; не SQL-совместимость по умолчанию |
| S12 | [SQLite: Datatypes and Collating Sequences](https://www.sqlite.org/datatype3.html) | BINARY/RTRIM, ограничения независимого oracle при dynamic typing |
| S13 | [libcudf Unicode Limitations](https://docs.nvidia.com/cudf/latest/libcudf/unicode_limitations/) | Хранение UTF-8 не равно всем Unicode-функциям |
| S14 | [HIP programming model, version 6.4.2](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.2/understand/programming_model.html) | Locality и architecture-dependent execution groups; не API установленного PCUDA |

Отдельно просмотрено руководство NVIDIA CUDA Best Practices по памяти и divergence;
оно не используется для обещаний portable API, default-stream semantics или
конкретных порогов нашего engine. Новые библиотеки и зависимости не устанавливались.
