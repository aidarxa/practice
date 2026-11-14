#include "jit_headers.h"
namespace db_module_57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b {
constexpr int BLOCK_THREADS=128;
constexpr int ITEMS_PER_THREAD=4;
constexpr int TILE_SIZE=512;
static const KernelArgumentSchemaV2 schema[] = {
{1,2,3,0},
{1,2,3,0},
{1,1,1,0},
{1,1,1,0},
{1,1,1,0},
{1,1,1,0},
{2,2,1,0},
{3,3,1,1},
{3,3,1,1},
{3,3,1,1},
{3,3,1,1},
{3,3,1,1},
};
static const KernelModuleDescriptorV2 descriptor = {2,sizeof(KernelModuleDescriptorV2),DB_BACKEND_PCUDA,0,schema,12};
int32_t launch(const KernelInvocationV2* call, KernelErrorV2* error) {
if(!call) return db::kernelError(error,DB_KERNEL_ARGUMENT_ERROR,0,"null invocation");
const auto status=db::abi2::validateInvocation(db_module_57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b::descriptor,*call,error); if(status) return status;
if(call->launch.block[0]!=128 || call->launch.grid[0]>UINT32_MAX || call->launch.grid[1]!=1 || call->launch.grid[2]!=1 || call->launch.block[1]!=1 || call->launch.block[2]!=1 || call->launch.dynamic_shared_bytes)
return db::kernelError(error,DB_KERNEL_LAUNCH_ERROR,0,"incompatible 1D/static-shared module launch");
try {
unsigned long long* d_result = static_cast<unsigned long long*>(call->arguments[0].data);
std::uint64_t* d_result_validity = static_cast<std::uint64_t*>(call->arguments[1].data);
int* d_lo_discount = static_cast<int*>(call->arguments[2].data);
int* d_lo_extendedprice = static_cast<int*>(call->arguments[3].data);
int* d_lo_orderdate = static_cast<int*>(call->arguments[4].data);
int* d_lo_quantity = static_cast<int*>(call->arguments[5].data);
const unsigned long long rows_LINEORDER = call->arguments[6].value.u64;
const bool db_parameter_0_valid = call->arguments[7].valid != 0;
const std::int64_t db_parameter_0 = db_parameter_0_valid ? call->arguments[7].value.i64 : 0;
const bool db_parameter_1_valid = call->arguments[8].valid != 0;
const std::int64_t db_parameter_1 = db_parameter_1_valid ? call->arguments[8].value.i64 : 0;
const bool db_parameter_2_valid = call->arguments[9].valid != 0;
const std::int64_t db_parameter_2 = db_parameter_2_valid ? call->arguments[9].value.i64 : 0;
const bool db_parameter_3_valid = call->arguments[10].valid != 0;
const std::int64_t db_parameter_3 = db_parameter_3_valid ? call->arguments[10].value.i64 : 0;
const bool db_parameter_4_valid = call->arguments[11].valid != 0;
const std::int64_t db_parameter_4 = db_parameter_4_valid ? call->arguments[11].value.i64 : 0;
const auto submitted=pcudaParallelFor(dim3(static_cast<unsigned>(call->launch.grid[0])),dim3(128),0,static_cast<pcudaStream_t>(call->stream),[=] {
const db::KernelIndex it;
__shared__ unsigned long long db_collective_u64[128];
__shared__ std::int64_t db_collective_i64[128];
            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tile_items = (it.get_group_linear_id() == it.get_group_range(0) - 1) ? rows_LINEORDER - tile_offset : TILE_SIZE;
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];
            int items2[ITEMS_PER_THREAD];
            int lo_quantity[ITEMS_PER_THREAD];
            int lo_discount[ITEMS_PER_THREAD];
            int lo_orderdate[ITEMS_PER_THREAD];
            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_orderdate + tile_offset, tid, tile_offset, lo_orderdate, num_tile_items);
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (tid + BLOCK_THREADS * i < num_tile_items) {
                    flags[i] = flags[i] && (db_parameter_0_valid && db::safe_gte(lo_orderdate[i], db_parameter_0));
                }
            }
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (tid + BLOCK_THREADS * i < num_tile_items) {
                    flags[i] = flags[i] && (db_parameter_1_valid && db::safe_lte(lo_orderdate[i], db_parameter_1));
                }
            }
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_discount + tile_offset, tid, tile_offset, lo_discount, num_tile_items);
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (tid + BLOCK_THREADS * i < num_tile_items) {
                    flags[i] = flags[i] && (db_parameter_2_valid && db::safe_gte(lo_discount[i], db_parameter_2));
                }
            }
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (tid + BLOCK_THREADS * i < num_tile_items) {
                    flags[i] = flags[i] && (db_parameter_3_valid && db::safe_lte(lo_discount[i], db_parameter_3));
                }
            }
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_quantity + tile_offset, tid, tile_offset, lo_quantity, num_tile_items);
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (tid + BLOCK_THREADS * i < num_tile_items) {
                    flags[i] = flags[i] && (db_parameter_4_valid && db::safe_lt(lo_quantity[i], db_parameter_4));
                }
            }
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_extendedprice + tile_offset, tid, tile_offset, items, num_tile_items);
std::int64_t local_0=db::aggregateIdentity<std::int64_t,db::AggregateOp::Sum>();
unsigned long long local_seen_0=0ULL;
for(int i=0;i<ITEMS_PER_THREAD;++i) {
if(flags[i] && tid+BLOCK_THREADS*i<num_tile_items) {
if(1) {local_0=db::AggregateMerge<std::int64_t,db::AggregateOp::Sum>{}(local_0,static_cast<std::int64_t>(db::safe_mul(items[i], lo_discount[i])));++local_seen_0;}
}}
const auto agg_0=crystal::blockReduce<std::int64_t>(local_0,db_collective_i64,threadIdx.x,blockDim.x,db::AggregateMerge<std::int64_t,db::AggregateOp::Sum>{});
const auto seen_0=crystal::blockReduce<unsigned long long>(local_seen_0,db_collective_u64,threadIdx.x,blockDim.x,crystal::Sum<unsigned long long>{});
if(tid==0) {
if(seen_0) {db::atomicAggregate<std::int64_t,db::AggregateOp::Sum>(d_result[0],static_cast<std::int64_t>(agg_0));
db::atomic_set_valid_bit(d_result_validity,0);
}
}

});
if(submitted!=pcudaSuccess) return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,pcudaGetErrorString(submitted));
return DB_KERNEL_OK;
} catch(const std::exception& exception) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,exception.what());}
catch(...) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,"unknown PCUDA launch failure");}
}
}
extern "C" const char* db_kernel_identity_v2() {return "57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b";}
extern "C" const KernelModuleDescriptorV2* db_kernel_descriptor_v2() {return &db_module_57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b::descriptor;}
extern "C" int32_t db_kernel_launch_v2(const KernelInvocationV2* call, KernelErrorV2* error) {return db_module_57da16690c7c9871a93ef1241a1685147f8738a97bfe6b45e2e210441cd2776b::launch(call,error);}
