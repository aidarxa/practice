# ub-mode: инструменты для C++17

Общий набор для следующих работ по [roadmap](../../todo.md), без изменения
GPU-runtime и без запуска следующего этапа миграции. Это **локальные аналоги**
crapper/dryer/mutator, а не совместимые реализации одноимённых внешних пакетов.
Использованы подходы этапов [1](../../test/pcuda_stage1/REPORT.md) и
[2](../../test/pcuda_stage2/REPORT.md); новые инструменты от них не импортируют код,
не зависят от их targets и не переписывают исторические результаты.

## Что запускать

| Команда | Назначение | Успешный gate по умолчанию |
| --- | --- | --- |
| `coverage.py` | Сборка/тесты по recipe, `.profraw`, merge и сырой `llvm-cov export` | Тесты проходят, профили есть, LLVM не выдаёт диагностику |
| `crapper.py` | Clang AST + измеренное LLVM code-region coverage | Для каждой выбранной definition/instantiation CRAP-region **<10**, покрытие известно |
| `dryer.py` | Сходство токенов полных тел функций, в том числе между файлами | Нет пар с rank **≥0,8**, где оба тела имеют ≥50 токенов |
| `mutator.py discover` | Генерация мутаций builtin binary operators из настоящего AST | Получен непустой список; **это ещё не проверка тестов** |
| `mutator.py run` | Свежая baseline-сборка и отдельная копия исходников на каждую мутацию | Baseline проходит, **все** перечисленные мутации killed |

Зависимости: Python **3.10+**, Clang с JSON AST и совместимые `llvm-cov`,
`llvm-profdata`, runtime profile instrumentation. Python-пакеты, GPU SDK и сеть
для самопроверки/примера не нужны. Проверено на Python 3.11 / Clang и LLVM 17.
Пути задаются через `--clang`, `--cxx`, `--llvm-cov`, `--llvm-profdata`;
компилятор — один executable, не строка вида `ccache clang++`.

## Быстрый запуск на реальном API проекта

Все команды ниже выполняются из корня репозитория в Bash. Каждый `--output`
должен быть **новым** каталогом: существующий каталог не очищается и не переиспользуется.

```bash
mkdir -p build/ub
RUN=$(mktemp -d build/ub/run-XXXXXXXX)
FLAGS=()
```

Если Clang выбирает несовместимые системные C++ headers, задайте совместимый GCC
явно. На текущем стенде Clang 17 автоматически выбирал GCC 16 и не компилировал
`<string>`; явный GCC 14 исправил именно конфигурацию toolchain, не исходники.
Скрытого подбора headers или привязанных к машине путей в инструментах нет:

```bash
# При необходимости: export UB_GCC_INSTALL_DIR=/путь/к/совместимому/gcc
if [[ -n ${UB_GCC_INSTALL_DIR:-} ]]; then
    FLAGS+=("--flag=--gcc-install-dir=$UB_GCC_INSTALL_DIR")
fi
```

Готовый recipe — [`examples/size-expression.json`](examples/size-expression.json).
Он компилирует настоящий `src/core/size_expression.cpp` и независимый
[`examples/size_checks.cpp`](examples/size_checks.cpp), а не копию алгоритмов.
Обычные значения и переполнения проверяются по ожидаемым результатам C++17,
без `assert`, который мог бы отключить `NDEBUG`.

```bash
python3 tools/ub/coverage.py \
    --project . --recipe tools/ub/examples/size-expression.json \
    "${FLAGS[@]}" --output "$RUN/coverage"

python3 tools/ub/crapper.py \
    --source src/core/size_expression.cpp --flag=-Iinclude "${FLAGS[@]}" \
    --ast-filter db --function checkedSizeAdd --function checkedSizeMultiply \
    --coverage "$RUN/coverage/coverage.json" --output "$RUN/crapper"

python3 tools/ub/dryer.py \
    --source src/core/size_expression.cpp --flag=-Iinclude "${FLAGS[@]}" \
    --ast-filter db --output "$RUN/dryer"

python3 tools/ub/mutator.py run \
    --project . --recipe tools/ub/examples/size-expression.json \
    "${FLAGS[@]}" --output "$RUN/targeted-mutants"

python3 tools/ub/mutator.py discover \
    --project . --source src/core/size_expression.cpp --flag=-Iinclude "${FLAGS[@]}" \
    --ast-filter db --function checkedSizeAdd --function checkedSizeMultiply \
    --output "$RUN/generated"

python3 tools/ub/mutator.py run \
    --project . --recipe tools/ub/examples/size-expression.json \
    --mutations "$RUN/generated/report.json" "${FLAGS[@]}" --output "$RUN/auto-mutants"
```

`FLAGS` передаются всем фазам явно; `coverage.py` добавляет instrumentation через
`{flags}`. Для другой установленной версии LLVM передайте соответствующие пути
и в coverage, и в AST-анализ. Не смешивайте compiler/profile tools разных версий.

## Применение к своим изменениям

1. Сначала воспроизведите поведение тестом и получите ожидаемый fail.
2. Выберите изменённые файлы/функции и настоящих клиентов API. Не объявляйте
   локальную проверку приёмкой всего движка.
3. Соберите покрытие тем же toolchain и из тех же исходников; затем запустите
   `crapper` и межфайловый `dryer`. Проверьте не только итоговый `success`, но и
   scope, неизвестные/неисполненные функции, coverage и пары дубликатов.
4. Задайте семантические мутации проверяемых контрактов и/или сгенерируйте
   операторные. Запустите их через независимую пересборку и поведенческие тесты.
5. Для survivors добавляйте независимые граничные/ошибочные случаи. Equivalent
   mutants требуют разбора человеком; автоматически equivalent они не объявляются.
6. Упростите код, если требуется, и повторите проверки. Не снижайте пороги,
   не сужайте scope и не меняйте oracle ради зелёного результата.
7. Сохраните артефакты и опишите границы в основном отчёте соответствующего этапа.

Для `crapper`, `dryer`, `mutator discover` общие параметры:

- `--source PATH` повторяется; именно эти файлы, а не все включённые headers,
  входят в scope. `--function NAME` повторяется и выбирает точное **имя без qualifiers**;
  все перегрузки/инстанцирования с этим именем остаются. Неизвестное имя/пустой
  scope — ошибка, не успех.
- Для header используйте `--source include/...h --tu test/...cpp`, чтобы получить
  реально инстанцированные определения. Без `--tu` каждый source разбирается как
  отдельная C++17 translation unit. Макроопределённое тело функции отвергается,
  если его написанные byte ranges нельзя достоверно восстановить.
- `--compile-db build/.../compile_commands.json` берёт flags и working directory
  выбранного TU. `--compile-entry N` явно выбирает одну из нескольких записей
  этого TU (индексация с 0, без неявного выбора первого target). `-c`, output и
  dependency-output flags удаляются; source заменяется абсолютным путём.
  Joined output вроде `-ofoo.o` безопасно перекрывается новым output внутри
  артефактов: `-fsyntax-only` не создаёт object. Другие `-o...` driver options
  не выкидываются по общему префиксу.
  Response files нужно раскрыть заранее. AdaptiveCpp-specific flags не выкидываются
  молча: если обычный Clang их не понимает, используйте явные host flags.
- `--flag=-Iinclude`, `--flag=-DMODE=1` и другие flags повторяются. C++17 задаётся
  явно. AST-анализ не собирает executable и не использует старые `.o`.
- `--ast-filter db` ограничивает Clang по qualified-name substring и уменьшает
  большой JSON стандартных headers. Он не должен исключать выбранные функции.
- `--timeout` ограничивает AST subprocess; `run`/`coverage` имеют отдельно
  `--build-timeout` и `--test-timeout` в секундах. Это deadline корректности,
  **не performance-измеритель**.

Например, для cross-file проверки перечислите несколько `--source`; для функций
из разных namespace опустите `--ast-filter` либо выберите общий фильтр.
`--help` доступен у каждой команды и обоих подрежимов mutator.

## Recipe и мутации

Recipe — версионированный JSON, пример выше. Термины:

- `inputs`: непустой список **относительных** файлов/каталогов проекта. Для
  мутаций они копируются целиком, включая текущие незакоммиченные изменения.
  Executable mode build-скриптов сохраняется; меняется только принадлежащая
  инструменту копия mutation target, даже если оригинал read-only.
  Перечисляйте все необходимые headers, source, тесты и CMake-файлы; `..`, absolute
  paths и symlinks в inputs запрещены. Не включайте build/data/.git целиком.
- `build`, `test`: непустые массивы команд; каждая команда — **argv array**, без shell.
  Рабочий каталог — `{root}`. Любой ненулевой результат прерывает фазу.
- `{root}`: исходный project для coverage, независимая source-копия для мутации.
  `{out}`: новый build-каталог данного клиента; все создаваемые файлы направляйте туда.
- `{cxx}`: executable из `--cxx`. Отдельный argv элемент `{flags}` разворачивается
  в список compiler flags; без flags он исчезает. Это обязательно для coverage.
  Пробелы и кириллица в подставляемых путях не требуют shell quoting внутри JSON.
- `objects`: для coverage — непустой список instrumentation executables/shared
  objects, например `["{out}/checks"]`. Все тестовые процессы пишут отдельные
  `%p-%m.profraw` в один новый каталог текущей коллекции.
- `mutations`: для `run` — непустой список объектов с уникальным безопасным `id`,
  `file`, `before`, `after`. Пустое/неизменяющее редактирование отвергается.
  `before` должен встречаться **ровно один раз** во всём указанном файле.
  Для неоднозначного места задайте `offset` в **UTF-8 bytes** и обязательный
  `source_sha256`. Discovery формирует именно такой защищённый от устаревания список.
- `--mutations FILE` заменяет список recipe; принимает JSON-массив или объект с
  `mutations`, в том числе `generated/report.json`.

Каждый mutant компилируется с нуля; baseline также пересобирается в собственной
копии. Подмена `.o`, link-manifest heuristics и старые binaries не используются.
Копии и build/logs находятся под `baseline/` и `mutants/<id>/`. После baseline
проверяется неизменность её source inputs. Между mutants нет общей mutable source
или build-директории; исходный рабочий каталог не патчится.

Recipe — **доверенный исполняемый код**, не sandbox: команда с абсолютным путём
может обращаться к исходному проекту или внешним ресурсам. Для честной изоляции
все проектные sources/includes берите из `{root}`, результаты — из `{out}`.
Зависимости вне inputs и фоновая GPU-работа не изолируются автоматически.
При GPU-тестах клиент обязан дождаться завершения работы и корректно освободить
ресурсы; внешние общие JIT/PCH/data directories нельзя делать конкурентно writable.

Статусы `run`:

| Статус | Значение |
| --- | --- |
| `killed` | Мутант собран; реальный тест завершился ненулевым кодом или сигналом |
| `survived` | Собранный мутант прошёл все тестовые команды |
| `build-failed` / `build-timeout` | Не удалось собрать, тест ничего не доказал |
| `timeout` | Собран, но тест не завершился до deadline; **не kill** |

При timeout завершается вся process group, не только launcher. Невозможность
запуска команды и неуспех baseline — инфраструктурная ошибка. Score равен
`killed/(killed+survived)`; unresolved публикуются отдельно. Даже score=1 не даёт
`success`, если есть unresolved. Пустая кампания не считается успешной.

## Метрики и ограничения

**Crapper.** `CC² × (1 − region_coverage)³ + CC`. CC учитывает if, loops, case labels,
catch, conditional operators и short-circuit `&&`/`||`; вложенная lambda измеряется
отдельно, function-try и constructor initializers учитываются. Implicit/defaulted
методы без написанного тела вне scope. Это **не statement coverage** внешнего crapper.
LLVM `CodeRegion` сравниваются по mangled identity и полному имени исходного файла;
expansion/skipped/gap/branch records не входят в знаменатель. Constructor/destructor
Itanium aliases поддержаны. Template instantiations измеряются отдельно; шаблон,
который вообще не инстанцирован, не получает придуманного покрытия.

Отсутствующее соответствие — `coverage_status=missing`, `region_coverage=null`,
`crap_region=null` и неуспешный gate; измеренные, но неисполненные регионы — ноль.
Низкий CRAP **не означает полное покрытие**: у простой непроверенной функции он
может быть <10. Не прячьте сложный switch сменой правил подсчёта case.

Для экспорта от `coverage.py` проверяются завершённая коллекция, хеши выбранных
исходников из `inputs.json` и хеш самого экспорта. Все diagnostics от
`llvm-profdata`/`llvm-cov`, включая exit-0 mismatch warnings, запрещают публикацию
успешного `coverage.json`; сырые stdout/stderr сохраняются. Можно передать внешний
полный `llvm-cov export` (не `--summary-only`/`--skip-functions`), но тогда отчёт
явно говорит, что свежесть исходников **не проверена**: это ответственность вызывающего.
Относительные LLVM filenames нужно разрешить при export через `--compilation-dir`.
Отдельный самодельный export не должен использоваться для обхода mismatch warnings.

**Dryer.** Rank — максимум двух направлений `difflib.SequenceMatcher(...,
autojunk=False).ratio()` по C++ lexical tokens. Комментарии/пробелы игнорируются,
операторы, идентификаторы, обычные/raw literals сохраняются. Инстанцирования одного
написанного тела не являются дубликатами. Вложенная lambda/локальный метод не
сравнивается со своей содержащей функцией (это одни написанные байты), но обе
definitions по-прежнему сравниваются с другими телами. Пороги `--rank` и `--min-tokens` видны
в отчёте. Это whole-function heuristic, **не** поиск всех partial/renamed clones,
не полноценный C++ preprocessor и не автоматический refactoring. Совпадения
требуют проверки семантики; независимые ручные SQL/GPU oracles не надо объединять
с проверяемым кодом ради уменьшения duplication.

**Mutator.** Discovery поддерживает builtin `< ↔ <=`, `> ↔ >=`, `== ↔ !=`,
`&& ↔ ||`, `+ ↔ -`, `* ↔ /`. Только AST operator positions, не комментарии/строки;
неоднозначные macro ranges пропускаются. Overloaded operators, универсальное
удаление statements, все типы C++-мутаций и автоматическое доказательство
эквивалентности не реализованы. Семантические мутации ABI/NULL/ownership задавайте
в recipe. Это first-order targeted campaign, не доказательство отсутствия
ошибок или dead code. Никакого автоматического удаления production-кода нет.

**GPU.** LLVM host coverage не является device coverage. AST-фильтр не делает
PCUDA/SYCL семантику поддержанной обычным host Clang. Для неподдержанного device
пути используйте аппаратные поведенческие fixtures и независимые oracles из
проекта; явно отмечайте непроверенное. Этот набор не заменяет `db_cli`, 13 SSB,
manifest/NULL/3VL, sanitizer и performance-проверки из `AGENTS.md`.

## Артефакты, exit codes и самопроверка

- **0**: выбранный gate пройден; у discover — только список сгенерирован.
- **1**: анализ завершён, но есть нарушения, survivors или unresolved.
- **2**: неправильные входы, существующий output, toolchain/LLVM/AST ошибка,
  пустой scope, устаревшая мутация или неуспешная baseline. Результат не завершён.

Завершённые запуски публикуют `report.json`. `actions.jsonl` хранит команды,
причины действий, cwd, существенные env overrides, scope/hashes и результаты;
`*.command.json`, `*.stdout`, `*.stderr`, `*.result.json` сохраняют сырой процессный
вывод, включая ошибки. Coverage дополнительно хранит profiles, merged profile,
export, input/binary hashes и версии tools; AST хранится в `ast-N.stdout`.
При ошибке до создания Session каталог может отсутствовать; смотрите stderr/exit
code, не принимайте незавершённый каталог за прошлый успешный запуск.

Это инструменты **корректности**, не измерители производительности. Python timeout
не является таймером компиляции/SQL latency. Здесь нет performance-чисел, traces
и заявлений о выполнении целей PERF-COLD. Для таких измерений действует отдельная
методика проекта.

```bash
python3 -m unittest discover -s tools/ub/tests -v
```

Тесты действительно компилируют C++17, снимают LLVM profiles и запускают mutants;
проверяют также paths с пробелами/кириллицей, headers/TU/compile database, template
и lambda identities, stale hashes, mismatch warnings, timeout process groups,
неизменность исходников и различение всех исходов. Executables можно выбрать
через `UB_CXX`, `UB_LLVM_COV`, `UB_LLVM_PROFDATA`. Временные тестовые каталоги
удаляются только своим `TemporaryDirectory`; чужие build/worktrees не затрагиваются.
Инструменты не делают commit/push, установку зависимостей и уборку чужих артефактов.

Журнал добавления и исходные проверки этой задачи:
[`build/ub-tools-wh51wsBL/actions.jsonl`](../../build/ub-tools-wh51wsBL/actions.jsonl).
Историческая автоматическая кампания **6/7** сохранена в
[`project-auto-mutants-gcc14/report.json`](../../build/ub-tools-wh51wsBL/project-auto-mutants-gcc14/report.json):
100% code-region coverage двух checked-функций не обнаружило пропущенный
граничный overflow-case. Его добавление в пример не переписывает старый результат.
Артефакты `build/` локальные, не входят в Git; команды выше воспроизводят проверки.

Итоговая проверка: [23/23 теста](../../build/ub-tools-wh51wsBL/tests-complete.log),
[5/5 семантических](../../build/ub-tools-wh51wsBL/final-targeted-mutants/report.json) и
[7/7 операторных](../../build/ub-tools-wh51wsBL/verified-auto-mutants/report.json) mutants killed,
без unresolved. [CRAP-region](../../build/ub-tools-wh51wsBL/verified-crapper/report.json)
двух выбранных checked-функций — 2 и 3;
[whole-function duplication gate](../../build/ub-tools-wh51wsBL/verified-dryer/report.json) пройден.
Это проверка инструментария и примера, не всего движка или device-кода.
