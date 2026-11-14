#pragma once

/** @file
 * @brief Host-side, value-free query model. This is not the PCUDA module ABI.
 */
#include "core/operators.h"
#include "core/parameter_value.h"

#include <optional>
#include <variant>

namespace db {

/**
 * @brief Logical plan and ordered parameter schema, independent of an execution.
 * @details Slots are numbered by deterministic pre-order traversal before
 * optimization. Neither literal values, validity nor AST addresses enter a key.
 * Keys are versioned, length-delimited representations, not lossy hashes. The
 * module ABI/toolchain key and the bounded typed/fallback cache belong to stage 5.
 */
struct QueryTemplate {
    std::unique_ptr<OperatorNode> plan;
    ParameterSchema parameter_schema;

    /** @brief Encode operators, resolved expression references, output mapping and slot identities. */
    std::string structureKey() const;
    /** @brief Encode ordered types, excluding ordinary values and validity. */
    std::string schemaKey() const;
    /** @brief Combine lossless structure and schema representations for host plan caching. */
    std::string cacheKey() const;
};

/** @brief One prepared query; separate executions own separate value vectors. */
struct PreparedQuery {
    QueryTemplate query;
    ParameterValues parameters;
};

/**
 * @brief Extract literal occurrences from an already resolved plan.
 * @details Must run before optimization. Reference matching still uses original
 * values in the translator; here equal independent occurrences never merge.
 * Numeric context types NULL when unambiguous, otherwise it remains NULL-only.
 */
PreparedQuery parameterizeQuery(std::unique_ptr<OperatorNode> resolved_plan);

/**
 * @brief Validate count/types and retain an existing slot's type for untyped NULL.
 * @throws std::runtime_error If a new type requires another schema.
 */
ParameterValues bindParameterValues(const ParameterSchema& schema, const ParameterValues& values);

/**
 * @brief Clone and bind a template for historical host tests only.
 * @details This compatibility helper substitutes literals only after optimization.
 * Native PCUDA execution never uses it to specialize ordinary values.
 * It never changes the template or another execution's parameter snapshot.
 */
std::unique_ptr<OperatorNode> bindLegacyPlan(const QueryTemplate& query, const ParameterValues& values);

} // namespace db
