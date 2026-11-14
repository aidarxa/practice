# Дамп PCUDA-кода Q1.1, Q2.1 и Q4.2

Это **фактически скомпилированные исходники** из session 0/plain серии
`build/pcuda-stage8/run-8xun1k4x/ssb-series-y58itcgc`, а не новая генерация.
Каждая копия побайтно проверена; исходный путь, SHA-256 и назначение — в
[`manifest.json`](manifest.json). Исходные измерения и файлы не изменялись.

| Каталог | Генерируемые модули | Порядок GPU-работы |
| --- | ---: | --- |
| [`q11/generated`](q11/generated) | 3 | Scan → DenseCount → служебный prefix → DenseWrite |
| [`q21/generated`](q21/generated) | 4 | DDATE/SUPPLIER/PART builds → fused fact scan/probe/aggregate → служебная compaction при fetch |
| [`q42/generated`](q42/generated) | 5 | DDATE/PART/SUPPLIER/CUSTOMER builds → fused fact scan/probe/aggregate → служебная compaction при fetch |

В каждом каталоге запроса:

- `query.sql` — исходный SQL;
- `generated/*.cpp` — отдельное фактическое GPU-ядро и его host launch wrapper
  в одной единице трансляции; имена файлов сделаны читаемыми, содержимое не менялось;
- `execution-plan.txt` — read-only реконструкция логического/физического плана,
  identities которой совпали с измеренными модулями; содержит binding slots,
  GPU launches, инициализацию и размеры;
- `commands/` — настоящие compile/link argv, native wall/user/system timers и
  конфигурация `jit_jobs=4`;
- `measurements.json` — все семь cold samples, распределения, порядок build batches
  и отдельные диагностические GPU durations.

`commands/build.historical.sh` содержит **исторические абсолютные пути** к исходным
артефактам. Он сохранён для чтения, не сделан исполняемым; не запускайте его для
пересборки поверх raw. Для воспроизведения нужен новый build/artifact directory и
совместимый PCH. Это не автономный проект: используются PCUDA, заголовки движка и
ABI v2 из исходного репозитория.

`shared/` содержит отдельно помеченные **не генерируемые из SQL** исходники
служебных prefix/compaction kernels и общий `jit_headers.h`. Эти generic services
подготовлены при сборке движка, но могут требовать device JIT при первом запуске.
Объектные файлы, `.so` и постоянный кеш SQL-модулей сюда не переносились.

Сводка: [`cold-breakdown.csv`](cold-breakdown.csv), полный offline-разбор и границы
метрик — [`cold-comparison.json`](cold-comparison.json). Объяснение cold-разницы,
варианты дальнейшей работы и warm-backlog:
[`8-ANALYSIS/REPORT.md`](../../test/pcuda_stage8_analysis/REPORT.md).
Исходные проверки/измерения — в [`отчёте этапа 8`](../../test/pcuda_stage8/REPORT.md).
Суммы native wall отдельных параллельных команд не равны build elapsed; медианы
компонентов не складываются в медиану summary. Чистое время device JIT здесь
отдельно не измерено. Warm-кандидаты сохранены в 8-ANALYSIS и отложены по
решению пользователя от 2026-11-12.
