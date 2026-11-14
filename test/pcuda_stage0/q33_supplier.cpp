#include "q33_build.hpp"
namespace q33 { struct SupplierIdentity {}; }
extern "C" pcudaError_t q33_supplier(pcudaStream_t stream, q33::BuildArgs args) {
    return q33::build<q33::SupplierIdentity, false>(stream, args);
}
