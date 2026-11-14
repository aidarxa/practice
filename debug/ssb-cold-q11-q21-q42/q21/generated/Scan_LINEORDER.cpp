#include "jit_headers.h"
namespace db_module_7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80 {
constexpr int BLOCK_THREADS=128;
constexpr int ITEMS_PER_THREAD=4;
constexpr int TILE_SIZE=512;
static const KernelArgumentSchemaV2 schema[] = {
{1,1,3,0},
{1,1,3,0},
{1,1,3,0},
{1,1,3,0},
{1,1,3,0},
{1,1,3,0},
{1,2,3,0},
{1,1,1,0},
{1,1,1,0},
{1,1,1,0},
{1,1,1,0},
{2,2,1,0},
{2,2,1,0},
{2,2,1,0},
};
static const KernelModuleDescriptorV2 descriptor = {2,sizeof(KernelModuleDescriptorV2),DB_BACKEND_PCUDA,0,schema,14};
int32_t launch(const KernelInvocationV2* call, KernelErrorV2* error) {
if(!call) return db::kernelError(error,DB_KERNEL_ARGUMENT_ERROR,0,"null invocation");
const auto status=db::abi2::validateInvocation(db_module_7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80::descriptor,*call,error); if(status) return status;
if(call->launch.block[0]!=128 || call->launch.grid[0]>UINT32_MAX || call->launch.grid[1]!=1 || call->launch.grid[2]!=1 || call->launch.block[1]!=1 || call->launch.block[2]!=1 || call->launch.dynamic_shared_bytes)
return db::kernelError(error,DB_KERNEL_LAUNCH_ERROR,0,"incompatible 1D/static-shared module launch");
try {
int* d_d_hash_table = static_cast<int*>(call->arguments[0].data);
unsigned int* d_d_hash_table_occupied = static_cast<unsigned int*>(call->arguments[1].data);
int* d_s_hash_table = static_cast<int*>(call->arguments[2].data);
unsigned int* d_s_hash_table_occupied = static_cast<unsigned int*>(call->arguments[3].data);
int* d_p_hash_table = static_cast<int*>(call->arguments[4].data);
unsigned int* d_p_hash_table_occupied = static_cast<unsigned int*>(call->arguments[5].data);
unsigned long long* d_result = static_cast<unsigned long long*>(call->arguments[6].data);
int* d_lo_orderdate = static_cast<int*>(call->arguments[7].data);
int* d_lo_partkey = static_cast<int*>(call->arguments[8].data);
int* d_lo_revenue = static_cast<int*>(call->arguments[9].data);
int* d_lo_suppkey = static_cast<int*>(call->arguments[10].data);
const unsigned long long rows_LINEORDER = call->arguments[11].value.u64;
const unsigned long long rows_PART = call->arguments[12].value.u64;
const unsigned long long rows_SUPPLIER = call->arguments[13].value.u64;
const auto submitted=pcudaParallelFor(dim3(static_cast<unsigned>(call->launch.grid[0])),dim3(128),0,static_cast<pcudaStream_t>(call->stream),[=] {
const db::KernelIndex it;
            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tile_items = (it.get_group_linear_id() == it.get_group_range(0) - 1) ? rows_LINEORDER - tile_offset : TILE_SIZE;
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];
            int items2[ITEMS_PER_THREAD];
            int d_year[ITEMS_PER_THREAD];
            int p_brand1[ITEMS_PER_THREAD];
            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_partkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, p_brand1, flags, d_p_hash_table, rows_PART, 0, num_tile_items, d_p_hash_table_occupied);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_suppkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, d_s_hash_table, rows_SUPPLIER, 0, num_tile_items, d_s_hash_table_occupied);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_orderdate + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, d_year, flags, d_d_hash_table, 61131, 19920101, num_tile_items, d_d_hash_table_occupied);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_revenue + tile_offset, tid, tile_offset, items, num_tile_items);
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {
                    int hash = (((d_year[i] - 1992) * 1000 + (p_brand1[i] - 0))) % 7000;
                    int out = hash * 3;
                    atomicExch(&d_result[out + 0], static_cast<unsigned long long>(d_year[i]));
                    atomicExch(&d_result[out + 1], static_cast<unsigned long long>(p_brand1[i]));
db::atomicAggregate<std::int64_t,db::AggregateOp::Sum>(d_result[out + 2],static_cast<std::int64_t>(items[i]));
                }
            }

});
if(submitted!=pcudaSuccess) return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,pcudaGetErrorString(submitted));
return DB_KERNEL_OK;
} catch(const std::exception& exception) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,exception.what());}
catch(...) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,"unknown PCUDA launch failure");}
}
}
extern "C" const char* db_kernel_identity_v2() {return "7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80";}
extern "C" const KernelModuleDescriptorV2* db_kernel_descriptor_v2() {return &db_module_7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80::descriptor;}
extern "C" int32_t db_kernel_launch_v2(const KernelInvocationV2* call, KernelErrorV2* error) {return db_module_7e8a72d250a7bf38e65c64cf97c1cda348c27555352c5280b9c3eb96c43a4e80::launch(call,error);}
