#include "jit_headers.h"
namespace db_module_de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6 {
constexpr int BLOCK_THREADS=128;
constexpr int ITEMS_PER_THREAD=4;
constexpr int TILE_SIZE=512;
static const KernelArgumentSchemaV2 schema[] = {
{1,2,3,0},
{1,2,3,0},
{1,3,3,0},
{1,2,3,0},
{1,2,3,0},
{1,2,3,0},
{2,2,1,0},
{2,2,1,0},
{2,2,1,0},
};
static const KernelModuleDescriptorV2 descriptor = {2,sizeof(KernelModuleDescriptorV2),DB_BACKEND_PCUDA,0,schema,9};
int32_t launch(const KernelInvocationV2* call, KernelErrorV2* error) {
if(!call) return db::kernelError(error,DB_KERNEL_ARGUMENT_ERROR,0,"null invocation");
const auto status=db::abi2::validateInvocation(db_module_de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6::descriptor,*call,error); if(status) return status;
if(call->launch.block[0]!=128 || call->launch.grid[0]>UINT32_MAX || call->launch.grid[1]!=1 || call->launch.grid[2]!=1 || call->launch.block[1]!=1 || call->launch.block[2]!=1 || call->launch.dynamic_shared_bytes)
return db::kernelError(error,DB_KERNEL_LAUNCH_ERROR,0,"incompatible 1D/static-shared module launch");
try {
unsigned long long* d_aggregate_counts = static_cast<unsigned long long*>(call->arguments[0].data);
unsigned long long* d_aggregate_offsets = static_cast<unsigned long long*>(call->arguments[1].data);
std::int64_t* d_result_col_0 = static_cast<std::int64_t*>(call->arguments[2].data);
std::uint64_t* d_result_validity_col_0 = static_cast<std::uint64_t*>(call->arguments[3].data);
unsigned long long* d_result = static_cast<unsigned long long*>(call->arguments[4].data);
std::uint64_t* d_result_validity = static_cast<std::uint64_t*>(call->arguments[5].data);
const unsigned long long aggregate_group_count = call->arguments[6].value.u64;
const unsigned long long aggregate_tuple_size = call->arguments[7].value.u64;
const unsigned long long db_range_items = call->arguments[8].value.u64;
const auto submitted=pcudaParallelFor(dim3(static_cast<unsigned>(call->launch.grid[0])),dim3(128),0,static_cast<pcudaStream_t>(call->stream),[=] {
const std::uint64_t gid[1] = {static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x};
if(gid[0] >= db_range_items) return;
                unsigned long long g = static_cast<unsigned long long>(gid[0]);
                if (g >= aggregate_group_count || d_aggregate_counts[g] == 0ULL) return;
                unsigned long long out_row = d_aggregate_offsets[g];
                unsigned long long src = g * aggregate_tuple_size;
                d_result_col_0[out_row] = static_cast<std::int64_t>(d_result[src + 0]);
                if (db::bitmap_valid_at(d_result_validity, src + 0)) db::atomic_set_valid_bit(d_result_validity_col_0, out_row);

});
if(submitted!=pcudaSuccess) return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,pcudaGetErrorString(submitted));
return DB_KERNEL_OK;
} catch(const std::exception& exception) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,exception.what());}
catch(...) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,"unknown PCUDA launch failure");}
}
}
extern "C" const char* db_kernel_identity_v2() {return "de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6";}
extern "C" const KernelModuleDescriptorV2* db_kernel_descriptor_v2() {return &db_module_de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6::descriptor;}
extern "C" int32_t db_kernel_launch_v2(const KernelInvocationV2* call, KernelErrorV2* error) {return db_module_de118fdb12c2620fa72fa403c834c2ce9e0a5f565dcfa5c7ac5b8254fa07f9b6::launch(call,error);}
