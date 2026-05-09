Техническое Задание (Часть 3/3): Интеграция Конвейера и Динамическое Управление Памятью

Роль и задача:
Это финальный этап архитектурного рефакторинга нашей СУБД. У нас уже готовы абстрактные деревья (Expression Tree, Operator Tree), Транслятор, Оптимизатор с правилом Predicate Pushdown и JIT-кодогенератор (паттерн Visitor).

Твоя задача на этом этапе — собрать все эти модули в единый рабочий конвейер в QueryEngine::processQuery и решить критическую проблему управления памятью (из-за которой мы ловили исключения кучи AdaptiveCPP в сложных запросах типа SSB Q4.3).
Глава 6. Расчет размера буфера результатов (Dynamic Sizing)

Раньше мы выделяли жестко захардкоженные 21000 * 4 байт, что приводило к Segfault на GPU при больших группировках. Теперь размер должен вычисляться динамически на основе метаданных Каталога и структуры AggregateNode.
6.1 Модификация AggregateNode

Добавь в класс AggregateNode (в operators.h) метод для точного расчета кардинальности результата:
C++

uint64_t calculateResultSize(const Catalog& catalog) const {
    uint64_t total_size = 1;
    for (const auto& group_expr : group_by_exprs_) {
        // Если это ColumnRefExpr, извлекаем имя колонки и таблицы
        // Ищем статистику в catalog.getTableMetadata(...)
        // Если статистика есть, умножаем total_size на stats.cardinality_
        // Если статистики нет или это сложное выражение, используем фолбэк (например, берем общий размер таблицы)
        // ВАЖНО: Никакого хардкода `card = 7` для городов! Города должны брать свою реальную кардинальность.
    }
    return total_size;
}

6.2 Модификация ExecutionContext

В файле include/core/execution.h добавь поле для передачи размера в DatabaseInstance:
C++

struct ExecutionContext {
    sycl::queue* q_;
    std::unordered_map<std::string, void*> buffers_;
    void* result_buffer_ = nullptr;
    
    // НОВОЕ ПОЛЕ:
    size_t expected_result_size_{0}; 
    
    // ...
};

Глава 7. Интеграция главного конвейера (QueryEngine::processQuery)

Теперь нам нужно выбросить старый монолитный buildPhysicalPlan из QueryEngine и заменить его на элегантный вызов новых подсистем.

Измени метод processQuery (в execution.cpp или optimizer.cpp в зависимости от твоей структуры):
C++

void QueryEngine::processQuery(hsql::SelectStatement* stmt, ExecutionContext* ctx) {
    // ШАГ 1: Трансляция (AST -> Naive Operator Tree)
    QueryTranslator translator(catalog_);
    auto naive_tree = translator.translate(stmt);

    // ШАГ 2: Оптимизация (Naive Tree -> Optimized Operator Tree)
    Optimizer optimizer;
    auto optimized_tree = optimizer.optimize(std::move(naive_tree));

    // ШАГ 3: Расчет памяти
    // Ищем AggregateNode в корне или под корнем
    const AggregateNode* agg_node = findAggregateNode(optimized_tree.get());
    if (agg_node) {
        uint64_t tuples = agg_node->calculateResultSize(*catalog_);
        uint64_t tuple_size = agg_node->group_by_exprs_.size() + agg_node->aggregations_.size();
        ctx->expected_result_size_ = tuples * tuple_size;
    } else {
        ctx->expected_result_size_ = 1; // Скалярная агрегация
    }

    // ШАГ 4: JIT Генерация (Optimized Tree -> C++ Code)
    JITContext jit_ctx;
    JITOperatorVisitor visitor(jit_ctx);
    optimized_tree->accept(visitor);

    std::string final_code = assembleFinalKernel(jit_ctx);

    // ШАГ 5: Динамическая компиляция (.so)
    auto lib_path = compiler_->compile(final_code);

    // ШАГ 6: Выполнение
    // ВНИМАНИЕ: Вызов выделения памяти должен происходить именно ЗДЕСЬ,
    // когда мы точно знаем ctx->expected_result_size_!
    db_instance_->ensureCapacity(ctx->expected_result_size_);
    db_instance_->zeroResultBuffer();

    auto executor = std::make_unique<JITExecutor>(lib_path);
    executor->execute(ctx);
}

(Примечание: Убедись, что метод сборки финального C++ файла assembleFinalKernel правильно склеивает jit_ctx.includes_and_globals, jit_ctx.build_kernels и jit_ctx.probe_kernel в единую функцию extern "C" void execute_query(db::ExecutionContext* ctx)).
Глава 8. Финальный План Действий (Action Plan) для тебя

Твоя задача — завершить рефакторинг движка. Выдай мне готовый production-ready код в следующем порядке:

Шаг 1: AggregateNode и Управление Памятью

    Реализуй метод calculateResultSize в AggregateNode с правильным обращением к Catalog.

    Добавь expected_result_size_ в ExecutionContext.

    Обнови логику в DatabaseInstance, чтобы он НЕ аллоцировал буфер слепо в начале, а ждал точного размера от QueryEngine.

Шаг 2: QueryEngine::processQuery

    Напиши полную, обновленную версию processQuery, реализующую все 6 шагов конвейера (Трансляция -> Оптимизация -> Память -> Visitor -> Компиляция -> Выполнение).

Шаг 3: Сборка JIT-кода

    Покажи код функции assembleFinalKernel, которая склеивает результаты работы JITOperatorVisitor в валидный C++ файл, готовый для компилятора LLVM/SYCL.

Убедись, что нигде не осталось следов старого buildPhysicalPlan. Наша новая архитектура: Дерево -> Оптимизатор -> Visitor.