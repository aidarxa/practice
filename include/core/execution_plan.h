#pragma once
/** @file
 * @brief Backend-independent physical plan and its host interpreter.
 * @details Glossary: a buffer is a named runtime-owned allocation/view; a scalar
 * is execution-local control state; a kernel contract is an ordered ABI schema;
 * a block is a happens-before sequence on one explicit ordered stream. Branches
 * and loops contain blocks, not C++ source. Kernel bodies live outside this plan.
 */
#include "core/module_abi_v2.h"
#include "core/parameter_value.h"
#include "core/size_expression.h"
#include <variant>
#include <string>
#include <stdexcept>

namespace db {
/** @brief Storage ownership/binding, resolved by the backend, never by a module. */
enum class BufferRole { External, NullBitmap, Result, ResultValidity, ResultColumn,
                        ResultColumnValidity, Scratch, Alias, TextBytes, TextOffsets, TextValidity };
/** @brief One declared storage identity; scratch capacities are EnsureBuffer steps. */
struct PlanBuffer {
    std::string name;
    uint32_t type;
    BufferRole role;
    std::string external_name;
    size_t column = 0;
    bool optional = false;
    uint64_t external_elements = 0;
    std::string legacy_cpp_type;
};
/** @brief Borrowed runtime pointer and accessible extent in typed elements. */
struct BufferView { void* data; uint64_t elements; };
/** @brief An independent frontend slot; equal current values do not merge slots. */
struct ParameterSlot { size_t index; };
/** @brief Ordered binding from one ABI argument to a buffer, size expression or SQL slot. */
struct KernelArgumentBinding {
    std::string name;
    KernelArgumentSchemaV2 schema;
    std::variant<std::string, SizeExpr, ParameterSlot> value;
    std::string legacy_cpp_type;
};
/** @brief A kernel's complete argument schema, independent of generated C++ scope. */
struct KernelContract {
    std::string name;
    std::vector<KernelArgumentBinding> arguments;
    /** @brief Device requirements inferred by the temporary body lowering, not assumed warp properties. */
    bool requires_fp64 = false;
    bool requires_atomic64 = false;
    uint64_t static_shared_bytes = 0;
};

/** @brief Assign execution-local size/control state. */
struct SetScalar { std::string name; SizeExpr value; };
/** @brief Ensure a scratch allocation's high-water capacity, without implicit initialization. */
struct EnsureBuffer { std::string buffer; SizeExpr elements; };
/** @brief Initialize an explicit typed range with one byte pattern in the same stream. */
struct FillBuffer { std::string buffer; SizeExpr offset; SizeExpr elements; uint8_t value; };
/** @brief Ordered device-to-device service copy; offsets/count are typed elements. */
struct CopyBuffer { std::string destination; std::string source; SizeExpr destination_offset; SizeExpr source_offset; SizeExpr elements; };
/** @brief Select an existing buffer for ping-pong algorithms without transferring ownership. */
struct AliasBuffer { std::string destination; std::string source; };
/** @brief Submit exactly one kernel; zero work is skipped without inventing a launch. */
struct LaunchKernel { size_t kernel; SizeExpr grid; uint32_t block = 1; uint32_t shared_bytes = 0; };
/** @brief Read one nonnegative counter, waiting after its copy and before CPU use. */
struct ReadScalar { std::string destination; std::string buffer; SizeExpr offset; };
/** @brief Legacy count-prefix service: download counts, checked CPU scan, upload offsets. */
struct HostPrefix { std::string counts; std::string offsets; SizeExpr elements; std::string total; };
/** @brief Prebuilt GPU prefix service, with checked CPU total at a capacity/control boundary. */
struct DevicePrefix { std::string counts; std::string offsets; SizeExpr elements; std::string total; };
/** @brief Legacy counter reduction, not host materialization of query rows. */
struct HostReduce {
    enum class Kind { Sum, Minimum, Maximum } kind;
    std::string buffer;
    SizeExpr offset;
    SizeExpr elements;
    std::string destination;
};
/** @brief Choose the next descending radix bucket from a copied count vector. */
struct SelectRadixBucket { std::string buffer; std::string rank; std::string prefix; };
/** @brief Allocate/initialize typed columnar results in runtime, not inside kernels. */
struct EnsureColumns { std::vector<uint32_t> types; SizeExpr rows; };
/** @brief Result metadata consumed by the engine's existing typed fetch path. */
struct PlanResultShape { uint64_t rows; uint64_t values; bool dense; uint64_t validity_words; };
/** @brief Publish logical shape separately from the capacity of its storage. */
struct PublishResult { SizeExpr rows; SizeExpr values; bool dense; SizeExpr validity_words; };
/** @brief Reject unsupported sizes/algorithm bounds before unsafe work. */
struct Require { SizeExpr condition; std::string message; };
/** @brief An explicitly justified synchronization boundary. */
struct Wait { std::string reason; };
struct PlanStep;
using PlanBlock = std::vector<PlanStep>;
/** @brief Execute only the selected sequence; no opaque host C++ is stored. */
struct Branch { SizeExpr condition; PlanBlock yes; PlanBlock no; };
/** @brief Runtime loop with a finite safety bound; state is local to the execution. */
struct Loop { SizeExpr condition; PlanBlock body; uint64_t maximum_iterations; };
/** @brief Closed set of physical operations; an absent implementation is a compile error. */
struct PlanStep {
    using Operation = std::variant<SetScalar, EnsureBuffer, FillBuffer, CopyBuffer,
        AliasBuffer, LaunchKernel, ReadScalar, HostPrefix, HostReduce, SelectRadixBucket,
        EnsureColumns, PublishResult, Require, Wait, Branch, Loop, DevicePrefix>;
    Operation operation;
    template<class T> PlanStep(T value) : operation(std::move(value)) {}
};
/** @brief An immutable plan can be shared; no pointers/current parameter values are stored. */
struct ExecutionPlan {
    std::vector<PlanBuffer> buffers;
    std::vector<KernelContract> kernels;
    PlanBlock steps;
    size_t tuple_size = 1;
    /** @brief LIMIT-like slots validated as nonnegative Int64 before any backend side effects. */
    std::vector<size_t> size_parameters;
};

/**
 * @brief Resource/backend boundary used by the host interpreter.
 * @details Implementations own storage, guard allocation budgets, preserve host
 * copy lifetimes and check asynchronous errors at wait(). All operations use
 * stream(). Stage 3 owns resources through PCUDA and retains only a temporary
 * same-executor legacy launch view; this interface has no GPU-header dependency.
 */
class PlanBackend {
public:
    virtual ~PlanBackend() = default;
    virtual BufferView resolve(const PlanBuffer&) = 0;
    virtual BufferView ensure(const PlanBuffer&, uint64_t elements) = 0;
    virtual void fill(void*, uint8_t value, size_t bytes) = 0;
    virtual void copy(void* destination, const void* source, size_t bytes) = 0;
    virtual void wait() = 0;
    virtual void* stream() = 0;
    virtual uint32_t backend() const = 0;
    virtual void validateKernel(size_t index, const KernelContract&) = 0;
    virtual void launch(size_t index, const KernelInvocationV2&) = 0;
    /** @brief Bind one validated Text slot to device-readable owned bytes, after plan validation. */
    virtual DbTextViewV2 textParameter(size_t, const ParameterValue&) { throw std::runtime_error("backend has no Text parameter arena"); }
    virtual void ensureColumns(const std::vector<uint32_t>& types, uint64_t rows) = 0;
    virtual void publish(const PlanResultShape&) = 0;
    /** @brief Backends that execute a native prefix plan must implement its explicit service boundary. */
    virtual uint64_t prefix(void*,const void*,uint32_t,uint64_t) { throw std::runtime_error("backend has no device prefix service"); }
};

/** @brief Validate all contracts before side effects, execute and wait before returning/unwinding. */
void executePlan(const ExecutionPlan& plan, PlanBackend& backend, const ParameterValues& parameters);
/** @brief Inspect physical operations without parsing generated source. */
std::string describePlan(const ExecutionPlan& plan);
} // namespace db
