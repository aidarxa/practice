#include "q33_build.hpp"
namespace q33 { struct CustomerIdentity {}; }
extern "C" pcudaError_t q33_customer(pcudaStream_t stream, q33::BuildArgs args) {
    return q33::build<q33::CustomerIdentity, false>(stream, args);
}
