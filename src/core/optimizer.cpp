#include "../../include/core/optimizer.h"
#include <algorithm>
#include <functional>
#include <set>
#include <string>

using namespace db;

// ============================================================================
// Block 1: Visitor accept() implementations
// ============================================================================

void OpBlockLoad::accept(ICodeGeneratorVisitor &visitor) const {
  visitor.visit(*this);
}

void OpBlockFilter::accept(ICodeGeneratorVisitor &visitor) const {
  visitor.visit(*this);
}

void OpBlockBuildHashtable::accept(ICodeGeneratorVisitor &visitor) const {
  visitor.visit(*this);
}

void OpBlockProbeHashtable::accept(ICodeGeneratorVisitor &visitor) const {
  visitor.visit(*this);
}

void OpBlockAggregate::accept(ICodeGeneratorVisitor &visitor) const {
  visitor.visit(*this);
}

void OpBlockScalarAggregate::accept(ICodeGeneratorVisitor &visitor) const {
  visitor.visit(*this);
}

// ============================================================================
// Block 2: CodeGenerator — JIT SYCL code generation
// ============================================================================

static std::string predTypeToString(PredType pt) {
  switch (pt) {
  case PredType::EQ:  return "Eq";
  case PredType::GT:  return "GT";
  case PredType::LT:  return "LT";
  case PredType::GTE: return "GTE";
  case PredType::LTE: return "LTE";
  case PredType::NEQ: return "NEq";
  }
  return "Eq";
}

std::string CodeGenerator::generate(const PhysicalPlan &plan) {
  code.str("");
  code.clear();
  code << "#include \"core/execution.h\"\n";
  code << "#include <sycl/sycl.hpp>\n";
  code << "#include \"crystal/load.h\"\n";
  code << "#include \"crystal/pred.h\"\n";
  code << "#include \"crystal/join.h\"\n";
  code << "#include \"crystal/utils.h\"\n\n";
  code << "using namespace sycl;\n\n";
  code << "constexpr int BLOCK_THREADS = 128;\n";
  code << "constexpr int ITEMS_PER_THREAD = 4;\n";
  code << "constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;\n\n";
  for (const auto &k : plan.kernels)
    code << "class " << k.name_ << ";\n";
  code << "\nextern \"C\" void execute_query(db::ExecutionContext* ctx) {\n";
  code << "    sycl::queue& q = *(ctx->q_);\n";
  for (const auto &buf : plan.data_columns_) {
    if (buf.scope_ == BufferScope::EXTERNAL_INPUT) {
      code << "    " << buf.type_ << "* " << buf.name_
           << " = ctx->getBuffer<" << buf.type_ << ">(\"" << buf.name_ << "\");\n";
    }
  }
  if (plan.device_result_buffer_.scope_ == BufferScope::EXTERNAL_OUTPUT) {
    code << "    " << plan.device_result_buffer_.type_ << "* "
         << plan.device_result_buffer_.name_ << " = ctx->getResultPointer();\n";
  }
  for (const auto &ht : plan.hash_tables_) {
    if (ht.scope_ == BufferScope::INTERNAL_TEMP) {
      code << "    " << ht.type_ << "* " << ht.name_
           << " = sycl::malloc_device<" << ht.type_ << ">("
           << ht.size_ << ", q);\n";
      if (ht.needs_zeroing_)
        code << "    q.memset(" << ht.name_ << ", 0, " << ht.size_
             << " * sizeof(" << ht.type_ << "));\n";
    }
  }
  if (plan.device_result_buffer_.needs_zeroing_)
    code << "    q.memset(" << plan.device_result_buffer_.name_
         << ", 0, " << plan.device_result_buffer_.size_
         << " * sizeof(" << plan.device_result_buffer_.type_ << "));\n";
  code << "\n";
  for (const auto &kernel : plan.kernels) {
    generateKernel(kernel);
  }
  code << "    q.wait();\n\n";
  for (const auto &ht : plan.hash_tables_)
    if (ht.scope_ == BufferScope::INTERNAL_TEMP)
      code << "    sycl::free(" << ht.name_ << ", q);\n";
  code << "}\n";
  return code.str();
}

void CodeGenerator::generateKernel(const Kernel &kernel) {
  first_filter_in_kernel_ = true;
  code << "    q.submit([&](sycl::handler& h) {\n";
  code << "        int num_tiles = (" << kernel.iteration_size_
       << " + TILE_SIZE - 1) / TILE_SIZE;\n";
  code << "        size_t local = BLOCK_THREADS;\n";
  code << "        size_t global = num_tiles * BLOCK_THREADS;\n";
  code << "        h.parallel_for<class " << kernel.name_
       << ">(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {\n";
  for (const auto &reg : kernel.registers_)
    code << "            int " << reg << "[ITEMS_PER_THREAD];\n";
  code << "\n            int tid = it.get_local_linear_id();\n";
  code << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
  code << "            int num_tiles_local = (" << kernel.iteration_size_
       << " + TILE_SIZE - 1) / TILE_SIZE;\n";
  code << "            int num_tile_items = TILE_SIZE;\n";
  code << "            if (it.get_group_linear_id() == num_tiles_local - 1) {\n";
  code << "                num_tile_items = " << kernel.iteration_size_
       << " - tile_offset;\n";
  code << "            }\n\n";
  code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n\n";
  for (const auto &op : kernel.operations_)
    op->accept(*this);
  code << "        });\n    });\n\n";
}

void CodeGenerator::visit(const OpBlockLoad &op) {
  code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
       << op.column_device_pointer_ << " + tile_offset, tid, tile_offset, "
       << op.reg_ << ", num_tile_items);\n";
}

void CodeGenerator::visit(const OpBlockFilter &op) {
  std::string pn = predTypeToString(op.pred_type);
  std::string pfx = first_filter_in_kernel_ ? "BlockPred" : "BlockPredA";
  first_filter_in_kernel_ = false;
  code << "            " << pfx << pn
       << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, "
       << op.reg_ << ", " << op.flags_reg_ << ", "
       << op.value << ", num_tile_items);\n";
}

void CodeGenerator::visit(const OpBlockBuildHashtable &op) {
  if (op.variant_ == 1) {
    code << "            BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
         << "tid, " << op.reg_ << ", " << op.flags_reg_ << ", "
         << op.hashtable_pointer_ << ", " << op.table_len_ << ", "
         << op.key_mins_ << ", num_tile_items);\n";
  } else {
    code << "            BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>("
         << "tid, " << op.reg_ << ", " << op.val_reg_ << ", "
         << op.flags_reg_ << ", " << op.hashtable_pointer_ << ", "
         << op.table_len_ << ", " << op.key_mins_ << ", num_tile_items);\n";
  }
}

void CodeGenerator::visit(const OpBlockProbeHashtable &op) {
  if (op.variant_ == 1) {
    code << "            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
         << "tid, " << op.reg_ << ", " << op.flags_reg_ << ", "
         << op.hashtable_pointer_ << ", " << op.table_len_ << ", "
         << op.key_mins_ << ", num_tile_items);\n";
  } else {
    code << "            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>("
         << "tid, " << op.reg_ << ", " << op.val_reg_ << ", "
         << op.flags_reg_ << ", " << op.hashtable_pointer_ << ", "
         << op.table_len_ << ", " << op.key_mins_ << ", num_tile_items);\n";
  }
}

void CodeGenerator::visit(const OpBlockAggregate &op) {
  code << "\n            #pragma unroll\n";
  code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
  code << "                if (" << op.flags_
       << "[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
  code << "                    int hash = " << op.hash_expr_ << ";\n";
  for (uint8_t g = 0; g < op.group_regs_.size(); ++g) {
    code << "                    " << op.res_ptr_ << "[hash*"
         << std::to_string(op.tuple_size_) << "+"
         << std::to_string(g) << "] = "
         << op.group_regs_[g] << "[i];\n";
  }
  uint8_t agg_off = static_cast<uint8_t>(op.group_regs_.size());
  for (uint8_t a = 0; a < op.agg_values_.size(); ++a) {
    uint8_t slot = agg_off + a;
    code << "                    sycl::atomic_ref<unsigned long long, "
         << "sycl::memory_order::relaxed, sycl::memory_scope::device, "
         << "sycl::access::address_space::global_space> atomic_agg_"
         << std::to_string(a) << "(" << op.res_ptr_ << "[hash*"
         << std::to_string(op.tuple_size_) << "+"
         << std::to_string(slot) << "]);\n";
    code << "                    atomic_agg_" << std::to_string(a)
         << ".fetch_add(" << op.agg_values_[a] << "[i]);\n";
  }
  code << "                }\n            }\n";
}

void CodeGenerator::visit(const OpBlockScalarAggregate &op) {
  code << "\n            unsigned long long sum = 0;\n";
  code << "            #pragma unroll\n";
  code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
  code << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
  code << "                    if (" << op.flags_ << "[i]) {\n";
  code << "                        sum += " << op.agg_expr_ << ";\n";
  code << "                    }\n                }\n            }\n";
  code << "            unsigned long long aggregate = "
       << "sycl::reduce_over_group(it.get_group(), sum, "
       << "sycl::plus<unsigned long long>());\n";
  code << "            if (tid == 0) {\n";
  code << "                sycl::atomic_ref<unsigned long long, "
       << "sycl::memory_order::relaxed, sycl::memory_scope::device, "
       << "sycl::access::address_space::global_space> "
       << "atomic_result(" << op.res_ptr_ << "[0]);\n";
  code << "                atomic_result.fetch_add(aggregate);\n";
  code << "            }\n";
}

// ============================================================================
// Block 3: Planner
// ============================================================================

static std::string getTablePrefix(const std::string &tbl_name) {
  if (tbl_name == "LINEORDER" || tbl_name == "lineorder") return "lo";
  if (tbl_name == "SUPPLIER" || tbl_name == "supplier") return "s";
  if (tbl_name == "CUSTOMER" || tbl_name == "customer") return "c";
  if (tbl_name == "PART" || tbl_name == "part") return "p";
  if (tbl_name == "DDATE" || tbl_name == "ddate" || tbl_name == "date") return "d";
  return "";
}

static std::string getSizeMacro(const std::string &tbl_name) {
  if (tbl_name == "LINEORDER" || tbl_name == "lineorder") return "LO_LEN";
  if (tbl_name == "SUPPLIER" || tbl_name == "supplier") return "S_LEN";
  if (tbl_name == "CUSTOMER" || tbl_name == "customer") return "C_LEN";
  if (tbl_name == "PART" || tbl_name == "part") return "P_LEN";
  if (tbl_name == "DDATE" || tbl_name == "ddate" || tbl_name == "date") return "D_LEN";
  return "0";
}

static std::string toLower(const std::string &s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(), ::tolower);
  return r;
}

static void collectTableRefs(hsql::TableRef *ref, std::vector<hsql::TableRef *> &out) {
  if (!ref) return;
  if (ref->type == hsql::kTableCrossProduct && ref->list) {
    for (auto *t : *ref->list) collectTableRefs(t, out);
  } else {
    out.push_back(ref);
  }
}

Planner::Planner(std::shared_ptr<Catalog> catalog) : catalog_(std::move(catalog)) {}

std::string Planner::translateMathExpression(
    hsql::Expr *expr,
    const std::unordered_map<std::string, std::string> &col_to_reg) const {
  if (!expr) return "";
  if (expr->isType(hsql::kExprColumnRef)) {
    std::string col = expr->name;
    auto it = col_to_reg.find(col);
    if (it != col_to_reg.end()) return it->second + "[i]";
    return col + "[i]";
  }
  if (expr->isType(hsql::kExprLiteralInt)) {
    return std::to_string(expr->ival);
  }
  if (expr->isType(hsql::kExprLiteralFloat)) {
    return std::to_string(expr->fval);
  }
  if (expr->isType(hsql::kExprOperator)) {
    std::string left = translateMathExpression(expr->expr, col_to_reg);
    std::string right = translateMathExpression(expr->expr2, col_to_reg);
    std::string op;
    switch (expr->opType) {
    case hsql::kOpPlus:     op = " + "; break;
    case hsql::kOpMinus:    op = " - "; break;
    case hsql::kOpAsterisk: op = " * "; break;
    case hsql::kOpSlash:    op = " / "; break;
    default: op = " ? "; break;
    }
    return "(" + left + op + right + ")";
  }
  if (expr->isType(hsql::kExprFunctionRef)) {
    if (expr->exprList && !expr->exprList->empty())
      return translateMathExpression((*expr->exprList)[0], col_to_reg);
  }
  return "0";
}

std::shared_ptr<LogicalPlan> Planner::buildLogicalPlan(hsql::SelectStatement *ast) {
  auto lp = std::make_shared<LogicalPlan>();
  // SELECT
  if (ast->selectList) {
    for (auto *e : *ast->selectList) {
      if (e->isType(hsql::kExprColumnRef)) {
        lp->columns_.push_back(e);
      } else if (e->isType(hsql::kExprFunctionRef)) {
        Aggregation agg;
        agg.func_name_ = e->name ? std::string(e->name) : "SUM";
        agg.expr_ = (e->exprList && !e->exprList->empty()) ? (*e->exprList)[0] : nullptr;
        lp->aggregations_.push_back(agg);
      }
    }
  }
  // FROM
  collectTableRefs(ast->fromTable, lp->tables_);
  // WHERE
  if (ast->whereClause) {
    std::vector<hsql::Expr *> flat;
    flattenAndConditions(ast->whereClause, flat);
    for (auto *cond : flat) {
      Expression ex;
      ex.expr = cond;
      if (cond->isType(hsql::kExprOperator) && cond->expr && cond->expr2) {
        bool left_col = cond->expr->isType(hsql::kExprColumnRef);
        bool right_col = cond->expr2->isType(hsql::kExprColumnRef);
        if (left_col && right_col) {
          std::string lt = getTableName(cond->expr->name);
          std::string rt = getTableName(cond->expr2->name);
          ex.type = (lt != rt) ? ExpressionType::JOIN : ExpressionType::FILTER;
        } else {
          ex.type = ExpressionType::FILTER;
        }
      } else {
        ex.type = ExpressionType::FILTER;
      }
      lp->conditions_.push_back(ex);
    }
  }
  // GROUP BY
  if (ast->groupBy && ast->groupBy->columns) {
    for (auto *e : *ast->groupBy->columns)
      lp->group_by_.push_back(e);
  }
  return lp;
}

static PredType opTypeToPredType(hsql::OperatorType op) {
  switch (op) {
  case hsql::kOpEquals:    return PredType::EQ;
  case hsql::kOpNotEquals: return PredType::NEQ;
  case hsql::kOpGreater:   return PredType::GT;
  case hsql::kOpGreaterEq: return PredType::GTE;
  case hsql::kOpLess:      return PredType::LT;
  case hsql::kOpLessEq:    return PredType::LTE;
  default: return PredType::EQ;
  }
}

std::shared_ptr<PhysicalPlan> Planner::buildPhysicalPlan(std::shared_ptr<LogicalPlan> lp) {
  auto pp = std::make_shared<PhysicalPlan>();
  // Identify fact and dimension tables
  std::string fact_table_name;
  std::vector<std::string> dim_table_names;
  for (auto *tref : lp->tables_) {
    std::string tname = tref->name ? toLower(std::string(tref->name)) : "";
    std::string canonical = getTableName(getTablePrefix(tname) + "_dummy");
    if (canonical.empty()) canonical = tname;
    try {
      const auto &meta = catalog_->getTableMetadata(canonical);
      if (meta.isFactTable()) fact_table_name = canonical;
      else dim_table_names.push_back(canonical);
    } catch (...) {
      // Try uppercase
      std::string upper = tname;
      std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
      try {
        const auto &meta = catalog_->getTableMetadata(upper);
        if (meta.isFactTable()) fact_table_name = upper;
        else dim_table_names.push_back(upper);
      } catch (...) {}
    }
  }

  // Collect columns needed per table, and build register maps
  std::set<std::string> all_columns;
  std::unordered_map<std::string, std::string> col_to_reg;
  int reg_counter = 0;

  // Gather all referenced columns
  auto addCol = [&](const std::string &col) { all_columns.insert(col); };
  for (auto *c : lp->columns_) if (c->name) addCol(c->name);
  for (auto &agg : lp->aggregations_) {
    if (agg.expr_) {
      std::function<void(hsql::Expr*)> walk = [&](hsql::Expr* e) {
        if (!e) return;
        if (e->isType(hsql::kExprColumnRef) && e->name) addCol(e->name);
        walk(e->expr); walk(e->expr2);
        if (e->exprList) for (auto *x : *e->exprList) walk(x);
      };
      walk(agg.expr_);
    }
  }
  for (auto &cond : lp->conditions_) {
    if (cond.expr && cond.expr->expr && cond.expr->expr->isType(hsql::kExprColumnRef))
      addCol(cond.expr->expr->name);
    if (cond.expr && cond.expr->expr2 && cond.expr->expr2->isType(hsql::kExprColumnRef))
      addCol(cond.expr->expr2->name);
  }
  for (auto *g : lp->group_by_) if (g->name) addCol(g->name);

  // Create device buffers for all columns
  for (const auto &col : all_columns) {
    DeviceBuffer db;
    db.name_ = "d_" + col;
    db.type_ = "int";
    db.size_ = getSizeMacro(getTableName(col));
    db.needs_zeroing_ = false;
    db.scope_ = BufferScope::EXTERNAL_INPUT;
    pp->data_columns_.push_back(db);
  }

  // For each dimension, find JOIN condition, filters, and build kernel
  for (const auto &dim_name : dim_table_names) {
    const auto &dim_meta = catalog_->getTableMetadata(dim_name);
    std::string prefix = getTablePrefix(dim_name);
    std::string pk_col, fk_col, val_col;
    uint8_t variant = 1;

    // Find JOIN condition for this dimension
    for (auto &cond : lp->conditions_) {
      if (cond.type != ExpressionType::JOIN) continue;
      std::string lc = cond.expr->expr->name;
      std::string rc = cond.expr->expr2->name;
      std::string lt = getTableName(lc), rt = getTableName(rc);
      if (lt == dim_name || toLower(lt) == toLower(dim_name)) {
        pk_col = lc; fk_col = rc; break;
      }
      if (rt == dim_name || toLower(rt) == toLower(dim_name)) {
        pk_col = rc; fk_col = lc; break;
      }
    }
    if (pk_col.empty()) continue;

    // Find value column for PHT_2 (GROUP BY or SELECT referencing this dim)
    for (auto *g : lp->group_by_) {
      if (g->name && getTableName(g->name) == dim_name) {
        std::string gn = g->name;
        if (gn != pk_col) { val_col = gn; variant = 2; break; }
      }
    }
    if (val_col.empty()) {
      for (auto *c : lp->columns_) {
        if (c->name && getTableName(c->name) == dim_name) {
          std::string cn = c->name;
          if (cn != pk_col) { val_col = cn; variant = 2; break; }
        }
      }
    }

    // Determine hash table sizing
    std::string ht_name = "d_" + prefix + "_hash_table";
    std::string ht_size, key_mins = "0";
    if (dim_meta.hasColumnStats(pk_col)) {
      auto &st = dim_meta.getColumnStats(pk_col);
      int64_t range = st.max_value_ - st.min_value_ + 1;
      key_mins = std::to_string(st.min_value_);
      ht_size = (variant == 2)
        ? "2*" + std::to_string(range)
        : std::to_string(range);
    } else {
      std::string sm = getSizeMacro(dim_name);
      ht_size = (variant == 2) ? "2*" + sm : sm;
    }

    DeviceBuffer ht_buf;
    ht_buf.name_ = ht_name;
    ht_buf.type_ = "int";
    ht_buf.size_ = ht_size;
    ht_buf.needs_zeroing_ = true;
    ht_buf.scope_ = BufferScope::INTERNAL_TEMP;
    pp->hash_tables_.push_back(ht_buf);

    // Build kernel
    Kernel bk;
    bk.name_ = "build_hashtable_" + prefix;
    bk.iteration_size_ = getSizeMacro(dim_name);
    bk.registers_.push_back("items");
    bk.registers_.push_back("flags");
    if (variant == 2) bk.registers_.push_back("items2");

    // Find FILTER conditions for this dimension
    std::vector<Expression*> dim_filters;
    for (auto &cond : lp->conditions_) {
      if (cond.type != ExpressionType::FILTER) continue;
      hsql::Expr *col_expr = nullptr, *val_expr = nullptr;
      if (cond.expr->expr && cond.expr->expr->isType(hsql::kExprColumnRef))
        col_expr = cond.expr->expr;
      if (cond.expr->expr2 && cond.expr->expr2->isType(hsql::kExprLiteralInt))
        val_expr = cond.expr->expr2;
      if (!col_expr && cond.expr->expr2 && cond.expr->expr2->isType(hsql::kExprColumnRef))
        col_expr = cond.expr->expr2;
      if (!val_expr && cond.expr->expr && cond.expr->expr->isType(hsql::kExprLiteralInt))
        val_expr = cond.expr->expr;
      if (col_expr && getTableName(col_expr->name) == dim_name)
        dim_filters.push_back(&cond);
    }

    // Emit: Load filter col -> Filter -> Load PK -> [Load val] -> Build HT
    if (!dim_filters.empty()) {
      std::string first_filter_col;
      for (auto *fp : dim_filters) {
        hsql::Expr *col_e = fp->expr->expr->isType(hsql::kExprColumnRef)
          ? fp->expr->expr : fp->expr->expr2;
        hsql::Expr *val_e = fp->expr->expr->isType(hsql::kExprLiteralInt)
          ? fp->expr->expr : fp->expr->expr2;
        std::string fcol = col_e->name;
        if (first_filter_col.empty() || fcol != first_filter_col) {
          auto load = std::make_unique<OpBlockLoad>();
          load->column_device_pointer_ = "d_" + fcol;
          load->reg_ = "items";
          bk.operations_.push_back(std::move(load));
          first_filter_col = fcol;
        }
        auto filt = std::make_unique<OpBlockFilter>();
        filt->reg_ = "items";
        filt->flags_reg_ = "flags";
        filt->pred_type = opTypeToPredType(fp->expr->opType);
        filt->value = std::to_string(val_e->ival);
        bk.operations_.push_back(std::move(filt));
      }
    } else {
      // No filters — InitFlags handles it (all 1s)
    }

    // Load PK
    auto load_pk = std::make_unique<OpBlockLoad>();
    load_pk->column_device_pointer_ = "d_" + pk_col;
    load_pk->reg_ = "items";
    bk.operations_.push_back(std::move(load_pk));

    // Load value for PHT_2
    if (variant == 2) {
      auto load_val = std::make_unique<OpBlockLoad>();
      load_val->column_device_pointer_ = "d_" + val_col;
      load_val->reg_ = "items2";
      bk.operations_.push_back(std::move(load_val));
    }

    // Build hashtable op
    auto build_op = std::make_unique<OpBlockBuildHashtable>();
    build_op->reg_ = "items";
    build_op->flags_reg_ = "flags";
    build_op->hashtable_pointer_ = ht_name;
    build_op->table_len_ = ht_size;
    build_op->key_mins_ = key_mins;
    build_op->variant_ = variant;
    if (variant == 2) build_op->val_reg_ = "items2";
    bk.operations_.push_back(std::move(build_op));
    pp->kernels.push_back(std::move(bk));

    // Store FK->HT mapping for probe phase
    col_to_reg[fk_col] = "items";
    if (variant == 2) col_to_reg[val_col] = val_col;
  }

  // SELECT kernel (probe + aggregate)
  Kernel sk;
  sk.name_ = "select_kernel";
  sk.iteration_size_ = fact_table_name.empty() ? "LO_LEN" : getSizeMacro(fact_table_name);
  sk.registers_.push_back("items");
  sk.registers_.push_back("flags");

  // Register named regs for group-by value columns
  std::set<std::string> named_regs;
  for (auto *g : lp->group_by_) {
    if (g->name) {
      std::string gn = g->name;
      if (getTableName(gn) != fact_table_name && named_regs.find(gn) == named_regs.end()) {
        sk.registers_.push_back(gn);
        named_regs.insert(gn);
        col_to_reg[gn] = gn;
      }
    }
  }

  // Add revenue/agg column registers
  for (auto &agg : lp->aggregations_) {
    if (agg.expr_) {
      std::function<void(hsql::Expr*)> walk = [&](hsql::Expr *e) {
        if (!e) return;
        if (e->isType(hsql::kExprColumnRef) && e->name) {
          std::string cn = e->name;
          if (named_regs.find(cn) == named_regs.end()) {
            sk.registers_.push_back(cn);
            named_regs.insert(cn);
            col_to_reg[cn] = cn;
          }
        }
        walk(e->expr); walk(e->expr2);
        if (e->exprList) for (auto *x : *e->exprList) walk(x);
      };
      walk(agg.expr_);
    }
  }

  // Probe ops for each dimension
  for (size_t di = 0; di < dim_table_names.size(); ++di) {
    const auto &dim_name = dim_table_names[di];
    std::string prefix = getTablePrefix(dim_name);
    std::string fk_col_probe, val_col_probe;
    uint8_t pv = 1;
    std::string ht_name_probe = "d_" + prefix + "_hash_table";
    std::string key_mins_probe = "0";
    std::string ht_len_probe;

    for (auto &cond : lp->conditions_) {
      if (cond.type != ExpressionType::JOIN) continue;
      std::string lc = cond.expr->expr->name, rc = cond.expr->expr2->name;
      std::string lt = getTableName(lc), rt = getTableName(rc);
      if (lt == dim_name) { fk_col_probe = rc; break; }
      if (rt == dim_name) { fk_col_probe = lc; break; }
    }

    // Find matching hash table
    for (auto &ht : pp->hash_tables_) {
      if (ht.name_ == ht_name_probe) {
        ht_len_probe = ht.size_;
        break;
      }
    }

    // Determine variant and val_col for probe
    for (auto *g : lp->group_by_) {
      if (g->name && getTableName(g->name) == dim_name) {
        std::string gn = g->name;
        // Check it's not the PK
        for (auto &cond : lp->conditions_) {
          if (cond.type != ExpressionType::JOIN) continue;
          std::string lc = cond.expr->expr->name, rc = cond.expr->expr2->name;
          if ((getTableName(lc) == dim_name && lc != gn) ||
              (getTableName(rc) == dim_name && rc != gn)) {
            val_col_probe = gn; pv = 2; break;
          }
        }
        if (pv == 2) break;
      }
    }

    // Get key_mins from catalog
    const auto &dm = catalog_->getTableMetadata(dim_name);
    for (auto &cond : lp->conditions_) {
      if (cond.type != ExpressionType::JOIN) continue;
      std::string lc = cond.expr->expr->name, rc = cond.expr->expr2->name;
      std::string pk = (getTableName(lc) == dim_name) ? lc : rc;
      if (dm.hasColumnStats(pk)) {
        key_mins_probe = std::to_string(dm.getColumnStats(pk).min_value_);
      }
      break;
    }

    auto load_fk = std::make_unique<OpBlockLoad>();
    load_fk->column_device_pointer_ = "d_" + fk_col_probe;
    load_fk->reg_ = "items";
    sk.operations_.push_back(std::move(load_fk));

    auto probe = std::make_unique<OpBlockProbeHashtable>();
    probe->reg_ = "items";
    probe->flags_reg_ = "flags";
    probe->hashtable_pointer_ = ht_name_probe;
    probe->table_len_ = ht_len_probe;
    probe->key_mins_ = key_mins_probe;
    probe->variant_ = pv;
    if (pv == 2) probe->val_reg_ = val_col_probe;
    sk.operations_.push_back(std::move(probe));
  }

  // Load aggregation columns in select kernel
  for (const auto &rn : named_regs) {
    if (getTableName(rn) == fact_table_name || fact_table_name.empty()) {
      auto ld = std::make_unique<OpBlockLoad>();
      ld->column_device_pointer_ = "d_" + rn;
      ld->reg_ = rn;
      sk.operations_.push_back(std::move(ld));
    }
  }

  // Aggregate or ScalarAggregate
  bool has_group_by = !lp->group_by_.empty();
  if (has_group_by) {
    auto agg_op = std::make_unique<OpBlockAggregate>();
    agg_op->flags_ = "flags";
    agg_op->res_ptr_ = "d_result";
    for (auto *g : lp->group_by_)
      agg_op->group_regs_.push_back(std::string(g->name));
    for (auto &a : lp->aggregations_) {
      if (a.expr_ && a.expr_->isType(hsql::kExprColumnRef))
        agg_op->agg_values_.push_back(std::string(a.expr_->name));
      else if (a.expr_)
        agg_op->agg_values_.push_back(translateMathExpression(a.expr_, col_to_reg));
    }
    agg_op->tuple_size_ = static_cast<uint8_t>(agg_op->group_regs_.size() + agg_op->agg_values_.size());

    // Build hash_expr from group-by cardinalities
    std::string hash_expr = "(";
    uint64_t res_size = 1;
    for (size_t i = 0; i < lp->group_by_.size(); ++i) {
      std::string gn = lp->group_by_[i]->name;
      std::string gt = getTableName(gn);
      uint64_t card = 7;
      try {
        const auto &gm = catalog_->getTableMetadata(gt);
        if (gm.hasColumnStats(gn)) card = gm.getColumnStats(gn).cardinality_;
      } catch (...) {}
      if (i > 0) hash_expr += " + ";
      hash_expr += gn + "[i]";
      if (gn.find("year") != std::string::npos) {
        hash_expr += " - 1992";
        card = 7; // 1992-1998
      }
      if (i < lp->group_by_.size() - 1) hash_expr += ") * " + std::to_string(card);
      res_size *= card;
    }
    hash_expr += ") % " + std::to_string(res_size);
    agg_op->hash_expr_ = hash_expr;

    // Result buffer
    uint64_t res_array_size = res_size * agg_op->tuple_size_;
    pp->device_result_buffer_.name_ = "d_result";
    pp->device_result_buffer_.type_ = "unsigned long long";
    pp->device_result_buffer_.size_ = std::to_string(res_array_size);
    pp->device_result_buffer_.needs_zeroing_ = true;
    pp->device_result_buffer_.scope_ = BufferScope::EXTERNAL_OUTPUT;

    sk.operations_.push_back(std::move(agg_op));
  } else {
    auto sagg = std::make_unique<OpBlockScalarAggregate>();
    sagg->flags_ = "flags";
    sagg->res_ptr_ = "d_result";
    if (!lp->aggregations_.empty() && lp->aggregations_[0].expr_) {
      sagg->agg_expr_ = "(unsigned long long)" +
        translateMathExpression(lp->aggregations_[0].expr_, col_to_reg);
    } else {
      sagg->agg_expr_ = "0";
    }
    pp->device_result_buffer_.name_ = "d_result";
    pp->device_result_buffer_.type_ = "unsigned long long";
    pp->device_result_buffer_.size_ = "1";
    pp->device_result_buffer_.needs_zeroing_ = true;
    pp->device_result_buffer_.scope_ = BufferScope::EXTERNAL_OUTPUT;
    sk.operations_.push_back(std::move(sagg));
  }

  pp->kernels.push_back(std::move(sk));
  return pp;
}

// ============================================================================
// Block 4: QueryOptimizer — Rule-Based Optimization
// ============================================================================

QueryOptimizer::QueryOptimizer(std::shared_ptr<Catalog> &catalog)
    : catalog_(catalog) {}

void QueryOptimizer::optimize(std::shared_ptr<LogicalPlan> lp) {
  // 1. Predicate Pushdown: reorder conditions so FILTERs for each dimension
  //    come before JOINs for that dimension
  std::vector<Expression> reordered;
  std::set<std::string> processed_tables;

  // Collect all dimension tables from JOIN conditions
  std::vector<std::string> dim_order;
  for (auto &cond : lp->conditions_) {
    if (cond.type == ExpressionType::JOIN && cond.expr &&
        cond.expr->expr && cond.expr->expr2) {
      std::string lc = cond.expr->expr->name;
      std::string rc = cond.expr->expr2->name;
      std::string lt = getTableName(lc), rt = getTableName(rc);
      // The non-fact table is the dimension
      for (const auto &t : {lt, rt}) {
        bool is_fact = false;
        try {
          is_fact = catalog_->getTableMetadata(t).isFactTable();
        } catch (...) {}
        if (!is_fact && processed_tables.find(t) == processed_tables.end()) {
          dim_order.push_back(t);
          processed_tables.insert(t);
        }
      }
    }
  }

  // For each dimension: emit FILTERs then JOIN
  for (const auto &dim : dim_order) {
    // FILTERs for this dimension
    for (auto &cond : lp->conditions_) {
      if (cond.type != ExpressionType::FILTER) continue;
      if (!cond.expr || !cond.expr->expr) continue;
      hsql::Expr *col_expr = nullptr;
      if (cond.expr->expr->isType(hsql::kExprColumnRef))
        col_expr = cond.expr->expr;
      else if (cond.expr->expr2 && cond.expr->expr2->isType(hsql::kExprColumnRef))
        col_expr = cond.expr->expr2;
      if (col_expr && getTableName(col_expr->name) == dim)
        reordered.push_back(cond);
    }
    // JOIN for this dimension
    for (auto &cond : lp->conditions_) {
      if (cond.type != ExpressionType::JOIN) continue;
      std::string lc = cond.expr->expr->name;
      std::string rc = cond.expr->expr2->name;
      if (getTableName(lc) == dim || getTableName(rc) == dim)
        reordered.push_back(cond);
    }
  }

  // Remaining FILTERs (fact table filters)
  for (auto &cond : lp->conditions_) {
    if (cond.type != ExpressionType::FILTER) continue;
    bool already = false;
    for (auto &r : reordered) {
      if (r.expr == cond.expr) { already = true; break; }
    }
    if (!already) reordered.push_back(cond);
  }

  // Any remaining conditions not yet added
  for (auto &cond : lp->conditions_) {
    bool already = false;
    for (auto &r : reordered) {
      if (r.expr == cond.expr) { already = true; break; }
    }
    if (!already) reordered.push_back(cond);
  }

  lp->conditions_ = std::move(reordered);

  // 2. Dead Column Elimination: remove columns from columns_ that are
  //    not referenced in conditions, aggregations, or group_by
  std::set<std::string> used_cols;

  // Columns used in conditions
  for (auto &cond : lp->conditions_) {
    if (!cond.expr) continue;
    if (cond.expr->expr && cond.expr->expr->isType(hsql::kExprColumnRef))
      used_cols.insert(cond.expr->expr->name);
    if (cond.expr->expr2 && cond.expr->expr2->isType(hsql::kExprColumnRef))
      used_cols.insert(cond.expr->expr2->name);
  }

  // Columns used in aggregations
  for (auto &agg : lp->aggregations_) {
    std::function<void(hsql::Expr *)> walk = [&](hsql::Expr *e) {
      if (!e) return;
      if (e->isType(hsql::kExprColumnRef) && e->name)
        used_cols.insert(e->name);
      walk(e->expr);
      walk(e->expr2);
      if (e->exprList)
        for (auto *x : *e->exprList)
          walk(x);
    };
    walk(agg.expr_);
  }

  // Columns used in GROUP BY
  for (auto *g : lp->group_by_) {
    if (g->name) used_cols.insert(g->name);
  }

  // Remove unused columns from columns_ (keep those in used_cols or in SELECT)
  // Only remove if column is truly dead (not referenced anywhere)
  auto it = std::remove_if(lp->columns_.begin(), lp->columns_.end(),
                           [&](hsql::Expr *e) {
                             if (!e->name) return false;
                             return used_cols.find(e->name) == used_cols.end();
                           });
  lp->columns_.erase(it, lp->columns_.end());
}
