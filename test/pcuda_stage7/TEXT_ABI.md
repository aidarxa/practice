# Native module ABI v2 и Text

Это тематическое приложение к [основному отчёту этапа 7](REPORT.md), а не второй
отчёт о прогонах. Оно нужно для точного описания согласованного C-контракта:
его layout, адресов, NULL и времени жизни нельзя заменить общим описанием строк.
Исследование альтернатив остаётся в [отдельном исследовании](../pcuda_string_research/REPORT.md).

## 1. Граница и совместимость

Публичные определения: [`module_abi_v2.h`](../../include/core/module_abi_v2.h).
Проверки без выделения памяти/запуска GPU:
[`module_contract_v2.h`](../../include/core/module_contract_v2.h).
Контракт рассчитан на согласованную 64-bit C ABI, C11 и C++17; он не является
сетевой сериализацией, Arrow interoperability или обещанием 32-bit ABI.

Все новые native SQL-модули, в том числе **числовые**, используют три фиксированных
имени C linkage:

```c
const char* db_kernel_identity_v2(void);
const KernelModuleDescriptorV2* db_kernel_descriptor_v2(void);
int32_t db_kernel_launch_v2(const KernelInvocationV2*, KernelErrorV2*);
```

Одна `.so` содержит одно специализированное ядро и его host-wrapper. Wrapper
проверяет вызов и ставит ровно один `pcudaParallelFor` в переданный stream.
Он не выделяет query memory, не выполняет SQL, не читает host `std::string` из
device-кода и не синхронизирует stream после каждого запуска.

ABI v1 остаётся в [`module_abi.h`](../../include/core/module_abi.h) **без изменения**.
Его argument размером 48 bytes нельзя приводить к v2 argument размером 56 bytes.
Общий размер invocation 80 bytes не делает указатели на разные argument arrays
совместимыми. Legacy SYCL backend не принимается native v2. Исторические v1
negative fixtures сохраняются; совместное исполнение произвольных старых v1
модулей движком не обещается. Постоянного SQL-module cache нет, поэтому загрузка
старого скомпилированного SQL из другого запуска также не требуется.

## 2. Layout: размеры и смещения в bytes

Заголовок проверяет размеры/смещения C11 `_Static_assert` и C++ `static_assert`.
Изменение полей **или их смысла** требует следующей версии ABI.

| POD | Size | Поля и offsets |
| --- | ---: | --- |
| `DbTextViewV2` | 16 | `const uint8_t* data` @0, `uint64_t bytes` @8 |
| `KernelArgumentSchemaV2` | 16 | `kind` @0, `type` @4, `access` @8, `flags` @12; четыре uint32 |
| `KernelScalarV2` | 16 | union `text`, `u64`, `i64`, `f64` @0; Text занимает все 16 bytes |
| `KernelArgumentV2` | 56 | `schema` @0, `void* data` @16, `element_count` @24, `value` @32, `valid` @48, `reserved` @52 |
| `KernelLaunchConfigV2` | 40 | `uint64_t grid[3]` @0, `uint32_t block[3]` @24, `dynamic_shared_bytes` @36 |
| `KernelInvocationV2` | 80 | `abi_version` @0, `struct_size` @4, `backend` @8, `reserved` @12, `stream` @16, `arguments` @24, `argument_count` @32, `launch` @40 |
| `KernelModuleDescriptorV2` | 32 | `abi_version` @0, `struct_size` @4, `backend` @8, `reserved` @12, `arguments` @16, `argument_count` @24 |
| `KernelErrorV2` | 208 | `struct_size` @0, `status` @4, `argument_index` @8, `char message[192]` @16 |

Schema/launch/error повторяют неизменившиеся POD v1 через явные typedef. Scalar,
argument, invocation и descriptor определены для v2 отдельно. Числовой scalar
содержит точные Int64/UInt64/Float64 bits в младших восьми bytes; верхние восемь
обязаны быть нулевыми. Для проверки union читается как object representation
через `memcpy`, не через неактивный член. Первым членом union является Text view:
инициализация `{}` обнуляет и pointer, и length, в том числе для Text NULL.

## 3. Типы и роли

Сохранены tags v1: Int32=1, UInt64=2, Int64=3, Float64=4, UInt8=5,
NULL-only=6, tagged-numeric=7. **Text=8** добавлен только в v2.

- **BUFFER**: `data` — device-readable/writable адрес, `element_count` — число
  *типизированных элементов*, `valid=1`. Для UInt8 единица — byte, для Text —
  16-byte view, для UInt64 bitmap — 64-bit word. Проверяются размер произведения
  и alignment (Text — 8, не 16). NULL pointer разрешён только optional buffer с
  нулевым extent. Числовые/указательные payload поля не используются как данные
  buffer и создаются runtime нулевыми.
- **SCALAR**: служебное числовое значение runtime control state; `data=nullptr`,
  `element_count=0`. Text в этой роли отклоняется.
- **PARAMETER**: SQL-slot, `data=nullptr`, `element_count=0`; значение в `value`,
  validity отдельно. Text parameter имеет `access=READ`, обычно nullable schema.
  Scalar Text нельзя маскировать tagged numeric. Tagged fallback принимает только
  согласованные числовые/NULL-only типы, не Text.

Access — READ/WRITE/READ_WRITE; `DB_ARG_NULLABLE=1`, прочие flags запрещены.
Все `reserved` нулевые. `valid` — строго 0 либо 1. Descriptor задаёт ожидаемые
kind/type/access/flags по слотам; invocation повторяет фактическую схему.
Несовместимый tag не становится совместимым из-за одинакового размера payload.

## 4. Text: содержимое и NULL

Text — валидный UTF-8, без логического U+0000, до `INT32_MAX` **bytes** на значение.
Сравнение полное, unsigned-byte, case-sensitive; нет нормализации/collation/rtrim.
Пробелы значимы. `é` и `e` + combining acute — разные значения.

`DbTextViewV2` — **заимствованный диапазон**, не владелец и не C string.
Нуль-терминатор не нужен; чтение за `bytes` запрещено. Пустой валидный Text имеет
`bytes=0` и может иметь NULL data. Это **не** SQL NULL.

- Валидный Text scalar: `valid=1`, длина в пределах, ненулевой адрес при длине >0.
- Text NULL scalar: `valid=0`, view `{nullptr,0}`, nullable schema.
- Text columns: NULL задаётся отдельным LSB-first bitmap. Отсутствующий bitmap
  означает all-valid. Invalid payload не интерпретируется как SQL-значение.
- GROUP BY объединяет NULL keys; обычное `NULL = value` остаётся UNKNOWN.
  COUNT(Text) пропускает NULL, MIN/MAX возвращают full Text выбранной строки.

UTF-8 проверяется на внешней границе storage/parameter, а не внутри каждой GPU
операции. ABI validator проверяет форму view, но не разыменовывает bytes и не
доказывает GPU-доступность произвольного чужого адреса. За реальные owners и extents
отвечает runtime; это не API для недоверенных указателей.

## 5. Компактная базовая колонка и промежуточные данные

На GPU колонка состоит из трёх отдельных owners:

1. `bytes`: UInt8 buffer логических bytes подряд;
2. `offsets`: `rows+1` неубывающих границ, начало 0, terminal offset = число bytes;
3. optional validity: `ceil(rows/64)` UInt64 words.

Loader читает **весь** файл FixedText по точному `rows*physical_width`. Первое
bounded NUL завершает значение, хвост физической записи игнорируется. Без NUL
используется вся ширина. NULL-record читается, но его payload не декодируется.
Ошибки размера, UTF-8, ширины и bitmap не исправляются тихим padding/truncation.

Int32 offsets выбираются, когда checked верхняя граница `rows*physical_width`
помещается в `INT32_MAX`. Иначе используются неотрицательные 64-bit offsets,
передаваемые как UInt64 storage. Нормализованный terminal offset ограничен
`INT64_MAX`. Один и тот же schema rule используется каталогом и upload; смена
ширины offsets структурная, не зависит от текущего SQL литерала.

Join сохраняет GPU row IDs, buckets и полные проверки равенства; совпадение hash
не равно совпадению Text. OR suppresses уже найденную *пару строк*, не одинаковые
значения разных строк. GROUP BY публикует immutable representative index через
CAS, затем обновляет отдельные states. Проекции, CASE и Text extrema переносят
views; строки между операторами не копируются на CPU. Sort сортирует GPU permutation
с полным сравнением, NULL-last для ASC/DESC и tie-break по входному индексу.

## 6. Владение, stream и материализация

`LoadedTextColumn` владеет базовыми буферами, `ExecutionContext` — parameter arena,
typed results и scratch. Borrowed result views могут ссылаться на base bytes
или parameter arena; ни одна строка в этой реализации не ссылается на bytes,
принадлежащие временной промежуточной строке.

Все зависимые H2D/kernel/copy/scan/gather операции используют один явный PCUDA
stream. Host wrapper копирует POD значения из invocation в closure до возврата;
GPU не читает host argument array. Arena загружается из validated `ParameterValue`
один раз на исполнение; transfer staging не уничтожается до завершения копии.
Arena нельзя перезаписывать, освобождать или переиспользовать до завершения всех
потребителей и fetch. Последовательная сессия соблюдает эту границу.

`TextFetch` — prebuilt sizes → GPU prefix → runtime capacity → GPU gather → D2H.
На CPU приходят **owned `std::string`** и отдельная validity, не device pointers,
hashes или словарные IDs. Display window ограничивает материализацию, но не меняет
SQL row count. CLI печатает Text как кавычкованные JSON-строки: `""`, `"NULL"`,
`"a "`, `"a|b"` и управляющие escapes однозначны; SQL NULL печатается без кавычек.

Все суммы/произведения размеров проверяются до выделения/запуска; memory guard
учитывает device, pinned staging и peak роста. Scratch coalescing использует
непересекающиеся времена жизни в явном плане; loop/branch — консервативный общий
region. Это не CPU fallback. Runtime удерживает HWM-workspace одного активного
value-free плана: переход к другой структуре/схеме освобождает прежние query-specific scratch и
result owners. Совпавшее имя scratch не отождествляет два плана. Повтор/совместимые
значения того же плана сохраняют HWM; process-only cache планов/модулей не очищается.
Универсальные scan/fetch services имеют собственный фиксированный набор reusable
buffers, общий для последовательной сессии; их фактическая ёмкость также входит
в runtime memory ledger. Это не обещание освобождения всех service buffers между SQL.
Освобождение/выгрузка допускаются только после подтверждённого completion;
ошибка teardown — fail-closed.

## 7. Ошибки, кеш и воспроизведение

Status codes неизменны: OK=0, ABI=1, schema=2, argument=3, launch=4, backend=5.
Диагностика пишется лишь в корректно sized caller-owned `KernelErrorV2`.
Wrapper не выпускает C++ exceptions через C boundary. Submission error и поздняя
асинхронная ошибка stream — разные границы проверки; success launch не означает
завершение выполнения.

Loader проверяет identity, descriptor/version/backend/schema и наличие всех C
entries перед публикацией модуля в process-only cache. Generated source использует
`pcuda-module-v2` и namespace SHA-256 module key. Обычные Text bytes, длина,
validity, адрес arena и scalar values в ключ не входят. Структура, аргументная
схема, ABI/code semantics и fingerprint compiler/PCH/flags входят. Совместимые
значения/typed NULL не должны вызывать plan/module codegen, compiler/linker,
dlopen/dlsym после подготовки; это проверяется счётчиками, не одним временем.

Независимая C11 layout-проверка:

```bash
cc -std=c11 -pedantic-errors -Wall -Wextra -Iinclude \
  test/pcuda_stage7_text/abi_c11.c -o "$RUN/abi-c11"
"$RUN/abi-c11"
```

Поведенческие host tests: `test_pcuda_text_values`, `test_pcuda_text_frontend`,
`test_pcuda_modules` и `test_optimizer`. GPU/CLI commands и результаты — в основном
отчёте. Host layout/coverage сами по себе не доказывают device correctness или
поддержку NVIDIA/Intel.
