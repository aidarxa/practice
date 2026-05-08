#pragma once
#include "../crystal/utils.h"
#include "../deps/include/SQLParser.h"
#include "memory.h"
#include "sql/Expr.h"
#include "sql/Table.h"
#include "utils.h"
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace db {
// TODO
// Тип JOIN гооврит о том, что мы можем сделать неявный JOIN для данного
// выражения, тип FILTER говорит, что это выражение просто фильт, и обязательно
// стоит до JOIN
enum class ExpressionType { NONE, JOIN, FILTER };
enum class PredType { GT, LT, GTE, LTE, EQ, NEQ };
struct Expression {
  ExpressionType type{ExpressionType::NONE};
  hsql::Expr *expr;
};
struct Aggregation {
  std::string func_name_; // "SUM"
  hsql::Expr *expr_;      // Указатель на выражение (lo_revenue - lo_supplycost)
};
// Хранит логическое описание SQL запроса, ничего не знает о том, как будут
// выполняться вычисления
struct LogicalPlan {
  std::vector<hsql::Expr *> columns_;    // Select
  std::vector<hsql::TableRef *> tables_; // from
  std::vector<Expression> conditions_; // where                  // Сканирование
  std::vector<hsql::Expr *> group_by_; // GROUP BY
  std::vector<Aggregation> aggregations_; // Aggregations (SUM)
};
class ICodeGeneratorVisitor;
class IPhysicalOperator {
public:
  virtual ~IPhysicalOperator() = default;
  virtual void accept(ICodeGeneratorVisitor &visitor) const = 0;
};
// block_load<...>(d_pointer + tile_offset, tid, target_reg, ...)
class OpBlockLoad : public IPhysicalOperator {
public:
  std::string column_device_pointer_; // "d_s_region"
  std::string reg_;                   // "items"
  void accept(ICodeGeneratorVisitor &visitor) const override;
};
// BlockPredEq<...>(tid, source_reg, mask_reg, val, ...)
class OpBlockFilter : public IPhysicalOperator {
public:
  std::string reg_;       // "items"
  std::string flags_reg_; // "flags"
  PredType pred_type;     // "Eq", "GTE", "ALTE"
  std::string value;      // "3" или "260"
  void accept(ICodeGeneratorVisitor &visitor) const override;
};
class OpBlockBuildHashtable : public IPhysicalOperator {
public:
  std::string reg_;               // items
  std::string flags_reg_;         // flags
  std::string hashtable_pointer_; // d_s_hash_table
  std::string key_mins_;          // for K key_mins
  std::string table_len_;         // S_LEN, D_LEN
  uint8_t variant_;               // PHT_1 or PHT_2
  std::string val_reg_; // value register for PHT_2, for example "items2"
  void accept(ICodeGeneratorVisitor &visitor) const override;
};
class OpBlockProbeHashtable : public IPhysicalOperator {
public:
  std::string reg_;               // "items"
  std::string val_reg_;           // "brand"
  std::string flags_reg_;         // "flags"
  std::string hashtable_pointer_; // "d_p_hash_table"
  std::string key_mins_;          // for K key_mins
  std::string table_len_;         // "P_LEN"
  uint8_t variant_;               // 1 или 2
  void accept(ICodeGeneratorVisitor &visitor) const override;
};
class OpBlockAggregate : public IPhysicalOperator {
public:
  std::string flags_; // flags
  // Строка с математикой хеша, которую сгенерирует Оптимизатор,
  // например: "(brand[i] * 7 + (year[i] - 1992)) % ((1998-1992+1) * (5*5*40))"
  std::string hash_expr_;
  std::vector<std::string> group_regs_; // ["year", "brand"]
  std::vector<std::string> agg_values_; // ["revenue"]
  std::string res_ptr_;                 // "d_result"
  // Размер одного элемента в выходном массиве (в q22 это 3: year, brand,
  // revenue)
  uint8_t tuple_size_;
  void accept(ICodeGeneratorVisitor &visitor) const override;
};
// Скалярная агрегация для Q1x-запросов (без GROUP BY)
// Использует reduce_over_group + atomic_ref в d_result[0]
class OpBlockScalarAggregate : public IPhysicalOperator {
public:
  std::string flags_;    // "flags"
  std::string agg_expr_; // "(unsigned long long)items2[i] * items[i]"
  std::string res_ptr_;  // "d_result"
  void accept(ICodeGeneratorVisitor &visitor) const override;
};
class ICodeGeneratorVisitor {
public:
  virtual void visit(const OpBlockLoad &op) = 0;
  virtual void visit(const OpBlockFilter &op) = 0;
  virtual void visit(const OpBlockBuildHashtable &op) = 0;
  virtual void visit(const OpBlockProbeHashtable &op) = 0;
  virtual void visit(const OpBlockAggregate &op) = 0;
  virtual void visit(const OpBlockScalarAggregate &op) = 0;
};
// Область видимости буфера в JIT-коде
enum class BufferScope {
  EXTERNAL_INPUT,  // Передается как аргумент (колонки таблиц)
  EXTERNAL_OUTPUT, // Передается как аргумент (буфер результата)
  INTERNAL_TEMP    // Выделяется и освобождается внутри функции (хеш-таблицы)
};
// Описание одного буфера устройства
struct DeviceBuffer {
  std::string name_;   // "d_lo_orderdate"
  std::string type_;   // "int", "unsigned long long"
  std::string size_;   // "LO_LEN"
  bool needs_zeroing_; // true для хеш-таблиц и result
  BufferScope scope_;  // область видимости в JIT-коде
};
class Kernel {
public:
  // Example: build_hashtable_s, select
  std::string name_;
  // Example: D_LEN, S_LEN
  std::string iteration_size_;
  // Example: items1, items2, flags, year, c_nation
  std::vector<std::string> registers_;
  // Example: block_load, BlockPredEq
  std::vector<std::unique_ptr<IPhysicalOperator>> operations_;
};

// Хранит физическое описание блочных функций, полностью определяет будущий C++
// код
class PhysicalPlan {
public:
  // 1. Выделение памяти (до запуска ядер)
  std::vector<DeviceBuffer> data_columns_; // Колонки таблиц (EXTERNAL_INPUT)
  std::vector<DeviceBuffer> hash_tables_;  // Хеш-таблицы (INTERNAL_TEMP)
  DeviceBuffer device_result_buffer_;      // Буфер ответа устройства (EXTERNAL_OUTPUT)
  uint8_t tuple_size_ = 1;

  // 2. Ядра (выполняются последовательно)
  std::vector<Kernel> kernels;
};
// Builder для планов
class Planner {
public:
  explicit Planner(std::shared_ptr<Catalog> catalog);
  std::shared_ptr<LogicalPlan> buildLogicalPlan(hsql::SelectStatement *ast);
  std::shared_ptr<PhysicalPlan>
  buildPhysicalPlan(std::shared_ptr<LogicalPlan> lp);

private:
  std::shared_ptr<Catalog> catalog_;
  // Рекурсивно транслирует AST математики в C++ строку с именами регистров
  std::string translateMathExpression(
      hsql::Expr *expr,
      const std::unordered_map<std::string, std::string> &col_to_reg,
      bool cast_to_ull = false) const;

  // Формирует ядра для сборки хеш-таблиц (Build Phase)
  void buildDimensionKernels(
      std::shared_ptr<LogicalPlan> lp, 
      std::shared_ptr<PhysicalPlan> pp,
      const std::vector<std::string> &dim_table_names,
      std::unordered_map<std::string, std::string> &col_to_reg) const;

  // Формирует фазу поиска и фильтрации фактовой таблицы (Probe Phase)
  void buildProbePhase(
      std::shared_ptr<LogicalPlan> lp,
      std::shared_ptr<PhysicalPlan> pp,
      Kernel &sk,
      const std::string &fact_table_name,
      std::unordered_map<std::string, std::string> &col_to_reg) const;

  // Генерирует сбалансированное C++ выражение для Perfect Hash без хардкода
  std::pair<std::string, uint64_t> generatePerfectHashExpression(
      const std::vector<hsql::Expr*> &group_by) const;
};
// Выполняет оптимизации LogicalPlan на месте
class QueryOptimizer {
public:
  QueryOptimizer(std::shared_ptr<Catalog> &catalog_);
  void optimize(std::shared_ptr<LogicalPlan> lp);

private:
  std::shared_ptr<Catalog> catalog_;
};

// Генерирует JIT SYCL код (extern "C" void execute_query(...))
class CodeGenerator : public ICodeGeneratorVisitor {
public:
  std::string generate(const PhysicalPlan &plan);

private:
  void generateKernel(const Kernel &kernel);
  void visit(const OpBlockLoad &op) override;
  void visit(const OpBlockFilter &op) override;
  void visit(const OpBlockBuildHashtable &op) override;
  void visit(const OpBlockProbeHashtable &op) override;
  void visit(const OpBlockAggregate &op) override;
  void visit(const OpBlockScalarAggregate &op) override;

private:
  std::stringstream code;
  bool first_filter_in_kernel_ = true;
};
} // namespace db