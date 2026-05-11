# JIT Engine Push-Model Refactoring (Data-Centric Code Generation)

This detailed document outlines the complete architectural roadmap for shifting our In-Memory JIT engine to a pipelined "Push" model (Data-Centric Code Generation).

By inverting the control flow, leaf generators push data upwards into parent operators. This allows us to embed execution logic directly inside tight inner loops, preventing the "Overwrite Phenomenon", natively supporting Row Expansion (e.g., 1-to-N joins, complex OR conditions), and maintaining strict isolation of variables.

## User Review Required

> [!WARNING]
> The JIT compilation process will be fully disconnected from the `accept()` visitor pattern in the AST. The AST will remain a pure data tree. `JITOperatorVisitor` will act as a pipeline compiler containing explicit `produce` and `consume` dispatcher methods.

> [!IMPORTANT]
> The Multi-value Hash Table (MHT) for 1-to-N joins will use a **Two-Pass Allocation** workflow executed **entirely on the GPU**. We will not transfer arrays back to the host. Instead, the JIT generator will emit a highly optimized **3-step SYCL Global Prefix Sum** using `sycl::joint_exclusive_scan` from the Group Algorithms Library.

## Proposed Architecture

---

### 1. The Push-Model Compiler Interface

#### [MODIFY] [visitor.h](file:///home/aidar/practice/include/core/visitor.h) and [visitor.cpp](file:///home/aidar/practice/src/core/visitor.cpp)
We will introduce strict signature definitions for data flow inside the `JITOperatorVisitor` (or `PipelineCompiler`):

```cpp
class JITOperatorVisitor : public OperatorVisitor {
public:
    // Core Dispatchers
    void produce(const OperatorNode* node, JITContext& ctx);
    void consume(const OperatorNode* node, JITContext& ctx, const OperatorNode* sender, const std::vector<std::string>& active_vars);

private:
    // Handlers
    void produceTableScan(const TableScanNode* node, JITContext& ctx);
    void consumeFilter(const FilterNode* node, JITContext& ctx, const OperatorNode* sender, const std::vector<std::string>& active_vars);
    void produceHashJoin(const HashJoinNode* node, JITContext& ctx);
    void consumeHashJoin(const HashJoinNode* node, JITContext& ctx, const OperatorNode* sender, const std::vector<std::string>& active_vars);
    // ...
};
```

**Variable Isolation:**
`active_vars` acts as a dynamic registry of valid registers currently in scope. `JITContext` will no longer maintain global `loaded_columns`. Instead:
1. `TableScan` reads columns, builds `std::vector<std::string> scan_vars = {"lo_orderdate", "lo_revenue"}`, and calls `consume(parent, ctx, this, scan_vars)`.
2. `HashJoin` probes the Hash Table, unpacks a new column `d_year`, creates a new vector `std::vector<std::string> join_vars = active_vars; join_vars.push_back("d_year");`, and passes it up to its parent.

#### [MODIFY] [operators.h](file:///home/aidar/practice/include/core/operators.h)
Add a `OperatorNode* parent_ = nullptr;` field to the base `OperatorNode` class, updated via `addChild`. This is required for `consume` to climb the tree.

---

### 2. Multi-value Hash Table & 3-Step Global Scan

For 1-to-N joins, we need a directory structure: `[Key, Offset, Count]` mapping to a contiguous `Payload Chunk`. The memory allocation requires knowing the exact size, which we calculate purely on the GPU.

#### [MODIFY] [join.h](file:///home/aidar/practice/include/crystal/join.h)
We will add SYCL primitives for the Multi-value Hash Table:
- `BlockBuildMHT_Count`: Iterates over the build table, hashing keys, and doing `sycl::atomic_fetch_add` to a globally allocated `d_counts` array (size = Hash Table length).
- `BlockBuildMHT_Write`: Takes the prefix-summed `d_offsets` array, re-hashes keys, grabs an atomic offset, and writes the payload to the dynamically allocated global `Payload Chunk`.
- `ProbeMultiHT`: Probes the MHT, returning the `Offset` and `Count` for the inner Probe loops.

#### [NEW] `prefix_sum_generator.h / .cpp` (or integrated into JIT generation)
The JIT compiler must dynamically generate the 3-step Global Prefix Sum kernels:
1. **Block Scan Kernel**: 
   - Divides `d_counts` into blocks.
   - Uses `sycl::joint_exclusive_scan` across the work-group.
   - Writes the local sum to `d_offsets` and the total sum of the block to `d_block_sums`.
2. **Scan of Block Sums Kernel**:
   - A single work-group kernel that scans `d_block_sums` using `sycl::joint_exclusive_scan` in-place.
3. **Add Block Sums Kernel**:
   - Every work-group adds the prefix sum from `d_block_sums` to their respective chunk in `d_offsets`.

---

### 3. Pipeline Generation & Pipeline Breakers

#### [MODIFY] [visitor.h](file:///home/aidar/practice/include/core/visitor.h)
Refactor `JITContext` to manage disjoint SYCL kernels efficiently:
```cpp
struct Pipeline {
    std::string kernel_name;
    std::stringstream includes_and_globals;
    std::stringstream kernel_body;
};
// List of sequential kernels (e.g. Build_Count -> BlockScan -> BlockSumScan -> AddSums -> Build_Write -> Probe_Main)
std::vector<Pipeline> pipelines;
```

#### [MODIFY] [visitor.cpp](file:///home/aidar/practice/src/core/visitor.cpp)

**HashJoinNode (produceHashJoin):** (The Pipeline Breaker)
1. If simple PK join (unique): Use ultra-fast 1-to-1 PHT (Single pass).
2. If complex or OR join:
   - Pipeline 1: Generate `Pass 1 (Count Kernel)` to populate `d_counts`.
   - Pipelines 2,3,4: Generate the **3-Step Global Prefix Sum** kernels.
   - Pipeline 5: Generate `Pass 2 (Write Kernel)` to populate `Payload Chunk`.
3. Finally, trigger `produce(node->children_[1].get(), ctx)` to start the Fact table Scan (Probe phase).

**HashJoinNode (consumeHashJoin):**
- **1-to-N MHT**: Generate a call to `ProbeMultiHT`. Wrap the parent invocation inside an expansion loop:
  ```cpp
  // Generated code inside kernel:
  int offset = 0, count = 0;
  ProbeMultiHT(ht, key_var, &offset, &count);
  for (int j = 0; j < count; ++j) {
      int val = payload_chunk[offset + j];
      // Generate parent->consume call here
  }
  ```

**AggregateNode (consumeAggregate):**
- Simply loop over `active_vars` mapped to the grouping keys and aggregate functions, emitting the appropriate `atomic_add` calls.

---

### 4. Execution Logic Output

The final step is translating `JITContext::pipelines` into `execution.cpp`.
Instead of a single `q.submit`, the generated file will contain consecutive kernel submissions, interspersed with explicit `sycl::malloc_device` for intermediate arrays (like `d_counts`, `d_block_sums`, `d_offsets`, and `payload_chunk`) taking advantage of the sizes determined on the device. (Note: Allocating `payload_chunk` requires reading the very last element of `d_offsets` + `d_counts` back to the host, which is a single ULL PCIe transfer—acceptable and required for `malloc_device`).

## Open Questions

> [!NOTE]
> No further questions. The architecture is solid and the requirements are well-defined. Unless there are modifications, I am ready to begin implementing Phase 1 and 2.
