/** @file
 * @brief Execute structured control flow against an independent host-memory backend.
 */
#include "core/execution_plan.h"
#include "core/physical_planner.h"
#include "core/module_codegen.h"

#include <cassert>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>

using namespace db;

class HostBackend final : public PlanBackend {
public:
    std::map<std::string, std::vector<uint64_t>> memory;
    std::vector<std::string> events;
    uint64_t rows = 0;
    std::function<void(const KernelInvocationV2&)> action;
    std::vector<std::function<void()>> pending_copies;
    bool defer_copies = false;
    unsigned failed_waits = 0;
    BufferView resolve(const PlanBuffer& buffer) override {
        auto& data = memory.at(buffer.name);
        return {data.data(), data.size()};
    }
    BufferView ensure(const PlanBuffer& buffer, uint64_t elements) override {
        auto& data = memory[buffer.name];
        data.resize(std::max(data.size(), static_cast<size_t>(elements)));
        events.push_back("ensure");
        return {data.data(), elements};
    }
    void fill(void* data, uint8_t value, size_t bytes) override { std::memset(data, value, bytes); }
    void copy(void* dst, const void* src, size_t bytes) override {
        events.push_back("copy");
        if (defer_copies) pending_copies.push_back([=] { std::memcpy(dst, src, bytes); });
        else std::memcpy(dst, src, bytes);
    }
    void wait() override {
        events.push_back("wait");
        if (failed_waits) { --failed_waits; throw std::runtime_error("injected completion failure"); }
        for (const auto& copy : pending_copies) copy();
        pending_copies.clear();
    }
    void* stream() override { return this; }
    uint32_t backend() const override { return DB_BACKEND_PCUDA; }
    void validateKernel(size_t, const KernelContract&) override { events.push_back("validate"); }
    void launch(size_t, const KernelInvocationV2& call) override {
        events.push_back("launch");
        if (action) { action(call); return; }
        auto* output = static_cast<uint64_t*>(call.arguments[0].data);
        output[0] += call.arguments[1].value.u64;
    }
    void ensureColumns(const std::vector<uint32_t>&, uint64_t) override { events.push_back("columns"); }
    void publish(const PlanResultShape& shape) override { rows = shape.rows; }
    uint64_t prefix(void* destination,const void* source,uint32_t type,uint64_t count) override {
        assert(type == DB_TYPE_UINT64);
        events.push_back("device-prefix");
        auto* out = static_cast<uint64_t*>(destination);
        const auto* in = static_cast<const uint64_t*>(source);
        uint64_t sum = 0;
        for (uint64_t i = 0; i < count; ++i) { const auto value = in[i]; out[i] = sum; sum = checkedSizeAdd(sum,value); }
        return sum;
    }
};

template<class F> static void rejects(F fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}

static void checkServices() {
    ExecutionPlan plan;
    for (const auto& name : {"counts", "offsets", "copied"})
        plan.buffers.push_back({name, DB_TYPE_UINT64, BufferRole::Scratch, {}});
    plan.steps = {EnsureBuffer{"counts", 3}, EnsureBuffer{"offsets", 3}, EnsureBuffer{"copied", 3},
        HostPrefix{"counts", "offsets", 3, "total"}, CopyBuffer{"copied", "offsets", 0, 0, 3},
        ReadScalar{"last", "copied", 2},
        HostReduce{HostReduce::Kind::Minimum, "counts", 0, 3, "minimum"},
        HostReduce{HostReduce::Kind::Maximum, "counts", 0, 3, "maximum"},
        HostReduce{HostReduce::Kind::Sum, "counts", 0, 3, "sum"},
        Require{equal(variable("last"), 2), "wrong prefix"},
        Require{both(equal(variable("minimum"), 0), equal(variable("maximum"), 3)), "wrong extrema"},
        Require{equal(variable("sum"), variable("total")), "wrong counter sum"},
        Branch{0, {Require{0, "unselected branch executed"}}, {EnsureColumns{{DB_TYPE_UINT64}, variable("sum")}}},
        Wait{"test boundary"}, PublishResult{variable("total"), variable("total"), true, 0}};
    HostBackend backend;
    backend.memory["counts"] = {2,0,3};
    executePlan(plan, backend, {});
    assert(backend.rows == 5 && backend.memory["copied"] == std::vector<uint64_t>({0,2,2}));

    ExecutionPlan radix;
    radix.buffers.push_back({"hist", DB_TYPE_UINT64, BufferRole::Scratch, {}});
    radix.steps = {EnsureBuffer{"hist", 256}, SetScalar{"rank", 2}, SetScalar{"prefix", 1},
        SelectRadixBucket{"hist", "rank", "prefix"}, Require{equal(variable("rank"), 1), "wrong rank"},
        PublishResult{variable("prefix"), 1, true, 0}};
    backend.memory["hist"].resize(256);
    backend.memory["hist"][255] = 1;
    backend.memory["hist"][254] = 3;
    executePlan(radix, backend, {});
    assert(backend.rows == 510);
    backend.memory["hist"][255] = backend.memory["hist"][254] = 0;
    rejects([&] { executePlan(radix, backend, {}); });

    ExecutionPlan aliases;
    for (const auto& name : {"a", "b"}) aliases.buffers.push_back({name, DB_TYPE_UINT64, BufferRole::Scratch, {}});
    for (const auto& name : {"selected", "previous"}) aliases.buffers.push_back({name, DB_TYPE_UINT64, BufferRole::Alias, {}});
    aliases.kernels.push_back({"accumulate", {
        {"out", {DB_ARG_BUFFER, DB_TYPE_UINT64, DB_ACCESS_READ_WRITE, 0}, std::string("previous")},
        {"amount", {DB_ARG_SCALAR, DB_TYPE_UINT64, DB_ACCESS_READ, 0}, SizeExpr(7)}}});
    aliases.steps = {EnsureBuffer{"a",1}, EnsureBuffer{"b",1}, FillBuffer{"a",0,1,0}, FillBuffer{"b",0,1,0},
        AliasBuffer{"selected","a"}, AliasBuffer{"previous","selected"}, AliasBuffer{"selected","b"},
        LaunchKernel{0,1,1,0}, ReadScalar{"a_value","a",0}, ReadScalar{"b_value","b",0},
        Require{equal(variable("b_value"),0), "alias snapshot followed later rebinding"},
        PublishResult{variable("a_value"),1,true,0}};
    executePlan(aliases, backend, {});
    assert(backend.rows == 7);

    ExecutionPlan signed_counter;
    signed_counter.buffers.push_back({"signed",DB_TYPE_INT32,BufferRole::Scratch,{}});
    signed_counter.steps = {EnsureBuffer{"signed",1},ReadScalar{"value","signed",0},PublishResult{variable("value"),1,true,0}};
    backend.memory["signed"].resize(1);
    int32_t count = 4;
    std::memcpy(backend.memory["signed"].data(), &count, sizeof(count));
    executePlan(signed_counter, backend, {});
    assert(backend.rows == 4);
    count = -1;
    std::memcpy(backend.memory["signed"].data(), &count, sizeof(count));
    rejects([&] { executePlan(signed_counter, backend, {}); });
}

static void checkExactParameters() {
    ExecutionPlan plan;
    plan.kernels.push_back({"parameter", {{"p", {DB_ARG_PARAMETER, DB_TYPE_INT64, DB_ACCESS_READ, DB_ARG_NULLABLE}, ParameterSlot{0}}}});
    plan.steps = {LaunchKernel{0,1,1,0}};
    HostBackend backend;
    backend.action = [](const auto& call) { assert(call.arguments[0].value.i64 == INT64_C(9007199254740993)); };
    executePlan(plan, backend, {ParameterValue::integer(INT64_C(9007199254740993))});
    backend.action = [](const auto& call) { assert(call.arguments[0].valid == 0 && call.arguments[0].schema.type == DB_TYPE_INT64); };
    executePlan(plan, backend, {ParameterValue::null(ParameterType::Int64)});
    plan.kernels[0].arguments[0].schema.type = DB_TYPE_TAGGED_NUMERIC;
    backend.action = [](const auto& call) { assert(call.arguments[0].value.f64 == 1.25 && call.arguments[0].schema.type == DB_TYPE_FLOAT64); };
    executePlan(plan, backend, {ParameterValue::floating(1.25)});
}

/** @brief The interpreter passes only validated typed views and consumes the returned total as control state. */
static void checkDevicePrefix() {
    ExecutionPlan plan;
    plan.buffers = {{"counts",DB_TYPE_UINT64,BufferRole::Scratch,{}},
                    {"offsets",DB_TYPE_UINT64,BufferRole::Scratch,{}}};
    plan.steps = {EnsureBuffer{"counts",3},EnsureBuffer{"offsets",3},
                  DevicePrefix{"counts","offsets",3,"total"},PublishResult{variable("total"),1,true,0}};
    HostBackend backend;
    backend.memory["counts"] = {2,0,3};
    executePlan(plan,backend,{});
    assert(backend.rows == 5 && backend.memory["offsets"] == std::vector<uint64_t>({0,2,2}));
    assert(std::count(backend.events.begin(),backend.events.end(),"device-prefix") == 1);
    assert(std::count(backend.events.begin(),backend.events.end(),"copy") == 0);
    plan.steps[2] = DevicePrefix{"counts","counts",0,"total"};
    executePlan(plan,backend,{});
    assert(backend.rows == 0 && backend.memory["counts"] == std::vector<uint64_t>({2,0,3}));
    plan.steps[2] = DevicePrefix{"counts","offsets",4,"total"};
    rejects([&] { executePlan(plan,backend,{}); });
    plan.steps[2] = DevicePrefix{"counts","offsets",3,"total"};
    plan.buffers[1].type = DB_TYPE_INT32;
    rejects([&] { executePlan(plan,backend,{}); });
}

/** @brief Offset copies and empty launches must preserve observable backend behavior. */
static void checkCopyWindowsAndEmptyLaunches() {
    ExecutionPlan plan;
    plan.buffers = {{"source",DB_TYPE_UINT64,BufferRole::Scratch,{}},
                    {"destination",DB_TYPE_UINT64,BufferRole::Scratch,{}}};
    plan.steps = {EnsureBuffer{"source",4},EnsureBuffer{"destination",4},
                  FillBuffer{"destination",0,4,0},CopyBuffer{"destination","source",1,1,2}};
    HostBackend backend;
    backend.memory["source"] = {10,20,30,40};
    executePlan(plan, backend, {});
    assert(backend.memory["destination"] == std::vector<uint64_t>({0,20,30,0}));

    ExecutionPlan empty;
    empty.kernels.push_back({"no_work",{}});
    empty.steps = {LaunchKernel{0,0,128,0}};
    size_t dispatched = 0;
    backend.action = [&](const auto&) { ++dispatched; };
    executePlan(empty, backend, {});
    assert(dispatched == 0);
}

/** @brief A failed read wait drains while its local CPU destination still exists (ASan checks it too). */
static void checkTransferFailureLifetime() {
    for (auto type : {DB_TYPE_INT32, DB_TYPE_UINT64}) {
        ExecutionPlan plan;
        plan.buffers.push_back({"counter", static_cast<uint32_t>(type), BufferRole::External, {}, 1});
        plan.steps = {ReadScalar{"counter_value", "counter", 0}};
        HostBackend backend;
        backend.memory["counter"] = {7};
        backend.defer_copies = true;
        backend.failed_waits = 1;
        rejects([&] { executePlan(plan, backend, {}); });
        assert(backend.events.front() == "copy"); // D2H must be enqueued before its failing wait
        assert(backend.pending_copies.empty());
        assert(std::count(backend.events.begin(), backend.events.end(), "wait") == 3);
    }
    ExecutionPlan upload;
    upload.buffers = {{"counts", DB_TYPE_UINT64, BufferRole::Scratch, {}},
                      {"offsets", DB_TYPE_UINT64, BufferRole::Scratch, {}}};
    upload.steps = {EnsureBuffer{"counts", 3}, EnsureBuffer{"offsets", 3},
                   HostPrefix{"counts", "offsets", 3, "total"}, Require{0, "injected after upload"}};
    HostBackend backend;
    backend.memory["counts"] = {2,3,4};
    backend.defer_copies = true;
    rejects([&] { executePlan(upload, backend, {}); });
    assert(backend.pending_copies.empty() && backend.memory["offsets"] == std::vector<uint64_t>({0,2,5}));
}

static void checkBuilder() {
    PhysicalPlanBuilder builder;
    assert(builder.declareBuffer({"d_result", DB_TYPE_UINT64, BufferRole::Result, {}}));
    assert(!builder.declareBuffer({"d_result", DB_TYPE_UINT64, BufferRole::Result, {}}));
    rejects([&] { builder.declareBuffer({"d_result", DB_TYPE_INT32, BufferRole::Result, {}}); });
    rejects([&] { builder.declareBuffer({"bad;name", DB_TYPE_UINT64, BufferRole::Result, {}}); });
    builder.declareBuffer({"unreferenced", DB_TYPE_UINT64, BufferRole::Scratch, {}});
    builder.commonScalar("rows", 3);
    const std::string body = "d_result[gid[0]]=rows; const char* note=\"unreferenced\\\"\"; char c='x'; /* unreferenced */ // nonexistent\n";
    builder.range(builder.plan.steps, "only_kernel", body, 3);
    builder.range(builder.plan.steps, "only_kernel", body, 1);
    rejects([&] { builder.range(builder.plan.steps, "only_kernel", "changed", 1); });
    const auto query = builder.finish(1, 3);
    assert(query.plan.kernels.size() == 1 && query.plan.kernels[0].arguments.size() == 3);
    assert(query.plan.kernels[0].arguments[0].name == "d_result");
    const auto source = describePcudamodules(query);
    assert(source.find("malloc_device") == std::string::npos);
    assert(source.find("db_kernel_launch_v2") != std::string::npos);
    KernelBody local{KernelBody::Kind::NdRange,"local_kernel","it",
                     "local[it.get_local_linear_id()]=0;",{{"local",DB_TYPE_UINT64,32}}, {}};
    builder.kernel(builder.plan.steps, local, 1, 32);
    local.locals[0].elements = 64;
    rejects([&] { builder.kernel(builder.plan.steps, local, 1, 32); });
}

int main() {
    checkServices();
    checkExactParameters();
    checkDevicePrefix();
    checkBuilder();
    checkCopyWindowsAndEmptyLaunches();
    checkTransferFailureLifetime();
    const auto max_value = std::numeric_limits<uint64_t>::max();
    assert(ceilDiv(SizeExpr(max_value), SizeExpr(64)).evaluate({}) == max_value / 64 + 1);
    rejects([&] { (SizeExpr(max_value) + 1).evaluate({}); });
    rejects([&] { (SizeExpr(max_value) * 2).evaluate({}); });
    rejects([&] { (SizeExpr(0) - 1).evaluate({}); });
    rejects([&] { (SizeExpr(1) / 0).evaluate({}); });
    rejects([&] { variable("missing").evaluate({}); });
    rejects([&] { nextPowerOfTwo(max_value).evaluate({}); });
    rejects([&] { shiftLeft(1,64).evaluate({}); });
    rejects([&] { shiftLeft(max_value,1).evaluate({}); });
    rejects([&] { SizeExpr::reference(""); });
    rejects([&] { SizeExpr::operation(SizeExpr::Op::Constant,0); });
    assert(nextPowerOfTwo(3).evaluate({}) == 4);
    assert(shiftRight(16,2).evaluate({}) == 4);
    assert(bitOr(8,2).evaluate({}) == 10 && bitAnd(10,2).evaluate({}) == 2);
    assert(minimum(1,2).evaluate({}) == 1 && maximum(1,2).evaluate({}) == 2);
    assert(both(0,variable("not_read")).evaluate({}) == 0);
    assert(either(1,variable("not_read")).evaluate({}) == 1);
    assert(both(1,1).evaluate({}) == 1 && both(1,0).evaluate({}) == 0);
    assert(either(0,1).evaluate({}) == 1 && either(0,0).evaluate({}) == 0);
    assert(variable("x").describe() == "$x" && (SizeExpr(1)+2).describe().find("op") == 0);

    ExecutionPlan plan;
    plan.buffers.push_back({"counter", DB_TYPE_UINT64, BufferRole::Scratch, "", 0, false});
    KernelContract kernel{"accumulate", {
        {"counter", {DB_ARG_BUFFER, DB_TYPE_UINT64, DB_ACCESS_READ_WRITE, 0}, std::string("counter")},
        {"amount", {DB_ARG_SCALAR, DB_TYPE_UINT64, DB_ACCESS_READ, 0}, variable("i")}
    }};
    plan.kernels.push_back(kernel);
    plan.steps = {
        EnsureBuffer{"counter", 1}, FillBuffer{"counter", 0, 1, 0}, SetScalar{"i", 1},
        Loop{lessEqual(variable("i"), 3), {
            LaunchKernel{0, 1, 1, 0}, SetScalar{"i", variable("i") + 1}
        }, 4},
        ReadScalar{"answer", "counter", 0},
        Branch{equal(variable("answer"), 6), {PublishResult{variable("answer"), 1, true, 0}},
               {Require{0, "wrong accumulated answer"}}}
    };
    HostBackend backend;
    executePlan(plan, backend, {});
    assert(backend.rows == 6 && backend.events.front() == "validate");
    executePlan(plan, backend, {});
    assert(backend.rows == 6); // snapshots and initialization cannot leak between executions
    auto broken = plan;
    broken.steps.insert(broken.steps.begin() + 2, CopyBuffer{"counter", "counter", 0, 0, max_value});
    rejects([&] { executePlan(broken, backend, {}); });
    auto infinite = plan;
    infinite.steps = {Loop{1, {}, 2}};
    rejects([&] { executePlan(infinite, backend, {}); });
    auto unbound = plan;
    unbound.kernels[0].arguments[1].value = ParameterSlot{0};
    unbound.kernels[0].arguments[1].schema = {DB_ARG_PARAMETER, DB_TYPE_INT64, DB_ACCESS_READ, DB_ARG_NULLABLE};
    rejects([&] { executePlan(unbound, backend, {}); });
    std::cout << "execution plan checks passed\n";
}
