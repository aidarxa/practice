/** @file @brief Structural scratch register allocation for Text plans; never inspect runtime values. */
#include "text_plan_detail.h"
#include "core/code_symbols.h"
#include <algorithm>
#include <set>
#include <type_traits>

namespace db { namespace text_plan {
namespace {
/** @brief Enumerate buffer references, treating a runtime loop/branch as one conservative lifetime region. */
template<class Consumer>
void buffers(PlanBlock& block, ExecutionPlan& plan, const Consumer& consume) {
    for (auto& step : block) std::visit([&](auto& operation) {
        using T = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<T,EnsureBuffer> || std::is_same_v<T,FillBuffer> ||
                      std::is_same_v<T,ReadScalar> || std::is_same_v<T,HostReduce> || std::is_same_v<T,SelectRadixBucket>) {
            consume(operation.buffer);
        } else if constexpr (std::is_same_v<T,CopyBuffer> || std::is_same_v<T,AliasBuffer>) {
            consume(operation.source); consume(operation.destination);
        } else if constexpr (std::is_same_v<T,HostPrefix> || std::is_same_v<T,DevicePrefix>) {
            consume(operation.counts); consume(operation.offsets);
        } else if constexpr (std::is_same_v<T,LaunchKernel>) {
            for (auto& binding : plan.kernels.at(operation.kernel).arguments)
                if (auto* buffer = std::get_if<std::string>(&binding.value)) {
                    consume(*buffer); consume(binding.name);
                }
        } else if constexpr (std::is_same_v<T,Loop>) buffers(operation.body,plan,consume);
        else if constexpr (std::is_same_v<T,Branch>) { buffers(operation.yes,plan,consume); buffers(operation.no,plan,consume); }
    },step.operation);
}
struct Lifetime { std::string name; uint32_t type; size_t first, last; };
struct Slot { std::string name; uint32_t type; size_t last; };
std::vector<Lifetime> lifetimes(ExecutionPlan& plan) {
    std::map<std::string,Lifetime> intervals;
    for (const auto& buffer : plan.buffers)
        if (buffer.role==BufferRole::Scratch) intervals.emplace(buffer.name,Lifetime{buffer.name,buffer.type,SIZE_MAX,0});
    for (size_t i=0;i<plan.steps.size();++i) {
        PlanBlock region{plan.steps[i]};
        buffers(region,plan,[&](std::string& name) {
            const auto found = intervals.find(name);
            if (found==intervals.end()) return;
            found->second.first = std::min(found->second.first,i);
            found->second.last = i;
        });
    }
    std::vector<Lifetime> result;
    for (const auto& entry : intervals) result.push_back(entry.second);
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b) { return std::tie(a.first,a.name)<std::tie(b.first,b.name); });
    return result;
}
std::map<std::string,std::string> assignments(ExecutionPlan& plan) {
    std::vector<Slot> slots;
    std::map<std::string,std::string> result;
    for (const auto& interval : lifetimes(plan)) {
        auto slot = std::find_if(slots.begin(),slots.end(),[&](const auto& slot) { return slot.type==interval.type && slot.last<interval.first; });
        if (slot==slots.end()) {
            slots.push_back({"text_scratch_slot_"+std::to_string(slots.size()),interval.type,interval.last});
            slot = std::prev(slots.end());
        } else slot->last = interval.last;
        result.emplace(interval.name,slot->name);
    }
    return result;
}
}
PlannedQuery coalesceScratch(PlannedQuery query) {
    // Aliases would extend their owner's lifetime; this Text lowering does not generate them.
    for (const auto& buffer : query.plan.buffers) if (buffer.role==BufferRole::Alias) return query;
    const auto names = assignments(query.plan);
    const auto rename = [&](std::string& name) { const auto found=names.find(name); if(found!=names.end()) name=found->second; };
    buffers(query.plan.steps,query.plan,rename);
    for (auto& body : query.bodies) body.code=replaceCodeIdentifiers(body.code,names);
    std::set<std::string> retained;
    std::vector<PlanBuffer> declarations;
    for (auto buffer : query.plan.buffers) {
        rename(buffer.name);
        if (retained.insert(buffer.name).second) declarations.push_back(std::move(buffer));
    }
    query.plan.buffers=std::move(declarations);
    return query;
}
}} // namespace db::text_plan
