#include "q33_build.hpp"
namespace q33 { struct DateIdentity {}; }
extern "C" pcudaError_t q33_date(pcudaStream_t stream, q33::BuildArgs args) {
    return q33::build<q33::DateIdentity, true>(stream, args);
}
