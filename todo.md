## Компилятор SQL -> C++
Для AST используется hyrise/sql-parser

1) **Query Optimizer (Rule-Based Optimization)** - состоит из:
- Predicate Pushdown + Late Materialization
- Dead Column Elimination
- Star Schema Heuristic JOIN (фильтруем маленькие таблицы, строим PHT), одним сканом по lineorder пробаем все хеш таблицы

2) **Code Generator** - генерирует C++ код по AST дереву после оптимизации

3) **Executor** - выполняет сгенерированный код

# Классы



- Catalog - хранит информацию о таблицах и столбцах (минимум, максимум, количество уникальных). В случае ошибки объект не создается, выбрасывается исключение.

- IOptimizationRule - интерфейс правила оптимизации. Обладает методом apply(AST)

- PredicatePushdownRule - правило для pushdown предикатов, реализует IOptimizationRule

- DeadColumnEliminationRule - правило для удаления мертвых столбцов, реализует IOptimizationRule

- StarSchemaJoinRule - правило для фильтрации маленьких таблиц и построения PHT, реализует IOptimizationRule

- IPhysicalOperator - интерфейс оператора физического плана

- HashBuildOp - оператор построения хеш таблицы, реализует IPhysicalOperator

- HashProbeOp - оператор поиска в хеш таблице, реализует IPhysicalOperator

- FilterOp - оператор фильтрации, реализует IPhysicalOperator

- TableScanOp - оператор сканирования колонки, реализует IPhysicalOperator

- PhysicalPlan - общий контейнер, определяющий план выполнения, а конкретно:
- - Список условий
- - Список таблиц, откуда брать данные
- - Список возвращаемых данных

- Query Optimizer - получает AST дерево, Catalog, выполняет ряд преобразований в соотсветсвиии с std::vector<IOptimizationRule*>, хранит в себе оптимизированное AST дерево, если нужно, то может вернуть его. В случае ошибки объект не создается, выбрасывается исключение. Возвращает PhysicalPlan

- Code Generator - получает PhysicalPlan, генерирует код согласно плану, возвращает Program

- Program - конечный c++ файл. Состоит из:
- - Includes - includes для компиляции
- - Timer - замер времени. Один экземпляр.
- - Kernel - ядра для исполнения на GPU, части Program. Несколько экземпляров. Собираются из:
- - - Preparation - подготовка данных для ядра: num_tiles, local, global, выделение массивов, tid, tile_offset, num_tiles, num_tile_items, в единственном экземпляре
- - - IBlock - интерфейс блоков, emitCode() выполняет генерацию нужного кода.
- - - Postprocessing - пост-обработка данных (возврат), в единственном экземпляре

- Executor - получает Program, компилирует его, выполняет, возвращает результат.

