TODO 3: Conditional Probe и Поздняя Фильтрация (Уровень 3)

Цель: Поддержать кросс-табличные условия внутри OR (Запрос 2) и соединения таблиц, где условие JOIN само является OR-выражением (Запрос 1: lo_orderdate = d_datekey OR lo_commitdate = d_datekey).

Суть проблемы: 1. Классический HashJoinNode умеет делать только строгое зондирование (Probe) по одному внешнему ключу (fk_col).
2. Сложные OR-предикаты, затрагивающие несколько измерений сразу, не могут быть спущены (Pushdown) ни в одну из веток. Они остаются "висеть" наверху дерева и должны вычисляться после того, как все хеш-таблицы пробиты, но до агрегации.
Шаг 3.1: Адаптация FilterNode для "Поздней фильтрации" (Post-Probe)

Благодаря тому, что вы грамотно спроектировали JITOperatorVisitor, этот шаг почти готов, если вы реализовали TODO 2 (Inline JIT).

Когда PredicatePushdownRule видит условие (c_mktsegment = 1 AND d_year = 1998) OR ..., оно не сможет протолкнуть его ни в CUSTOMER, ни в DDATE (так как требуются обе таблицы). Условие останется в FilterNode на самом верху select_kernel.

Вам нужно убедиться, что метод visit(const FilterNode& node) в visitor.cpp корректно обрабатывает этот случай:
C++

void JITOperatorVisitor::visit(const FilterNode& node) {
    // 1. Сначала обходим детей (это сгенерирует все загрузки фактов и Hash Probes!)
    for (const auto& child : node.getChildren()) {
        child->accept(*this);
    }

    // 2. Теперь применяем фильтр "на лету" к уже пробитым данным
    // Если вы сделали TODO 2, JITExprVisitor сгенерирует Inline C++ код 
    // прямо здесь, используя переменные, которые уже лежат в регистрах.
    if (node.predicate) {
        bool first_pred = false; // Важно: мы не перезаписываем flags, мы их срезаем (AND)
        JITExprVisitor expr_vis(ctx_, *active_stream_, "flags", false, &first_pred);
        node.predicate->accept(expr_vis);
    }
}

Шаг 3.2: Поддержка OR-джойнов в HashJoinNode

Теперь решаем проблему Запроса 1 (lo_orderdate = d_datekey OR lo_commitdate = d_datekey).
Нам нужно научить HashJoinNode извлекать оба ключа и генерировать двойной Probe с объединением результатов.

Обновите логику поиска ключей в JITOperatorVisitor::visit(const HashJoinNode& node):
C++

    std::vector<std::string> fk_cols; // Теперь это массив ключей!
    
    // ... внутри findCols ...
    if (e->getType() == ExprType::OP_EQ) {
        // ... старая логика определения fk_col ...
        fk_cols.push_back(rc->column_name); // или lc->column_name
    } else if (e->getType() == ExprType::OP_AND || e->getType() == ExprType::OP_OR) {
        // Рекурсивно спускаемся и ищем все OP_EQ
        findCols(bin->left.get());
        findCols(bin->right.get());
    }

А в конце метода visit, при генерации кода в probe_kernel, добавьте ветвление:
C++

    if (build_idx < build_infos_.size()) {
        const auto& bi = build_infos_[build_idx];
        
        if (fk_cols.size() == 1) {
            // СТАНДАРТНЫЙ JOIN (Один ключ)
            // ... ваш текущий код с BlockLoad и BlockProbeAndPHT ...
        } 
        else if (fk_cols.size() > 1) {
            // OR-JOIN (Несколько ключей к одной таблице)
            std::string combined_mask = ctx_.getNewMask();
            ctx_.probe_kernel << "            int " << combined_mask << "[ITEMS_PER_THREAD];\n";
            ctx_.probe_kernel << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>(" << combined_mask << ");\n";
            
            for (size_t k = 0; k < fk_cols.size(); ++k) {
                std::string fk = fk_cols[k];
                std::string temp_flags = ctx_.getNewMask();
                
                // 1. Копируем текущие валидные флаги во временную маску
                ctx_.probe_kernel << "            int " << temp_flags << "[ITEMS_PER_THREAD];\n";
                ctx_.probe_kernel << "            BlockApplyMaskAnd<...>(tid, " << temp_flags << ", flags);\n";
                
                // 2. Загружаем очередной внешний ключ
                ctx_.external_columns.insert("d_" + fk);
                ctx_.probe_kernel << "            BlockLoad<...>(d_" << fk << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
                
                // 3. Делаем Probe в temp_flags (а не в главные flags!)
                ctx_.probe_kernel << "            BlockProbeAndPHT_1<...>(tid, items, " << temp_flags << ", " << bi.ht_name << ", ...);\n";
                
                // 4. Добавляем результат в комбинированную маску (OR)
                ctx_.probe_kernel << "            BlockApplyMaskOr<...>(tid, " << combined_mask << ", " << temp_flags << ");\n";
            }
            
            // Наконец, применяем комбинированный результат к главным флагам потока
            ctx_.probe_kernel << "            BlockApplyMaskAnd<...>(tid, flags, " << combined_mask << ");\n";
        }
    }

Результат TODO 3

С этим кодом движок сможет элегантно переварить двойной OR-джойн. Он сгенерирует код, который:

    Создаст пустую маску combined.

    Попробует пробить lo_orderdate. Если найдет совпадение в DDATE, запишет 1 в combined.

    Попробует пробить lo_commitdate. Если найдет, добавит 1 в combined.

    Оставит активными только те треды, где хотя бы одна из дат совпала!