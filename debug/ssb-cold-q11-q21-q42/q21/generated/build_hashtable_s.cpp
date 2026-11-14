#include "jit_headers.h"
namespace db_module_9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34 {
constexpr int BLOCK_THREADS=128;
constexpr int ITEMS_PER_THREAD=4;
constexpr int TILE_SIZE=512;
static const KernelArgumentSchemaV2 schema[] = {
{1,1,3,0},
{1,1,3,0},
{1,1,1,0},
{1,1,1,0},
{2,2,1,0},
{3,3,1,1},
};
static const KernelModuleDescriptorV2 descriptor = {2,sizeof(KernelModuleDescriptorV2),DB_BACKEND_PCUDA,0,schema,6};
int32_t launch(const KernelInvocationV2* call, KernelErrorV2* error) {
if(!call) return db::kernelError(error,DB_KERNEL_ARGUMENT_ERROR,0,"null invocation");
const auto status=db::abi2::validateInvocation(db_module_9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34::descriptor,*call,error); if(status) return status;
if(call->launch.block[0]!=128 || call->launch.grid[0]>UINT32_MAX || call->launch.grid[1]!=1 || call->launch.grid[2]!=1 || call->launch.block[1]!=1 || call->launch.block[2]!=1 || call->launch.dynamic_shared_bytes)
return db::kernelError(error,DB_KERNEL_LAUNCH_ERROR,0,"incompatible 1D/static-shared module launch");
try {
int* d_s_hash_table = static_cast<int*>(call->arguments[0].data);
unsigned int* d_s_hash_table_occupied = static_cast<unsigned int*>(call->arguments[1].data);
int* d_s_region = static_cast<int*>(call->arguments[2].data);
int* d_s_suppkey = static_cast<int*>(call->arguments[3].data);
const unsigned long long rows_SUPPLIER = call->arguments[4].value.u64;
const bool db_parameter_1_valid = call->arguments[5].valid != 0;
const std::int64_t db_parameter_1 = db_parameter_1_valid ? call->arguments[5].value.i64 : 0;
const auto submitted=pcudaParallelFor(dim3(static_cast<unsigned>(call->launch.grid[0])),dim3(128),0,static_cast<pcudaStream_t>(call->stream),[=] {
const db::KernelIndex it;
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];
            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tiles_local = (rows_SUPPLIER + TILE_SIZE - 1) / TILE_SIZE;
            int num_tile_items = TILE_SIZE;
            if (it.get_group_linear_id() == num_tiles_local - 1) {
                num_tile_items = rows_SUPPLIER - tile_offset;
            }
            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);
for(int i=0;i<ITEMS_PER_THREAD;++i) {
if(tid+BLOCK_THREADS*i<num_tile_items) flags[i] = flags[i] && (db_parameter_1_valid) && (db::safe_eq(d_s_region[(static_cast<unsigned long long>(tile_offset) + tid + BLOCK_THREADS * i)], db_parameter_1));
}
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_s_suppkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, d_s_hash_table, rows_SUPPLIER, 0, num_tile_items, d_s_hash_table_occupied);

});
if(submitted!=pcudaSuccess) return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,pcudaGetErrorString(submitted));
return DB_KERNEL_OK;
} catch(const std::exception& exception) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,exception.what());}
catch(...) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,"unknown PCUDA launch failure");}
}
}
extern "C" const char* db_kernel_identity_v2() {return "9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34";}
extern "C" const KernelModuleDescriptorV2* db_kernel_descriptor_v2() {return &db_module_9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34::descriptor;}
extern "C" int32_t db_kernel_launch_v2(const KernelInvocationV2* call, KernelErrorV2* error) {return db_module_9d6ad927889e16e23d995bc201fe6e3db691b86f27ce399f87744302be534e34::launch(call,error);}
