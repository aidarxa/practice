#include "typed_params.hpp"
#include <limits>

template <unsigned Identity>
pcudaError_t launch_variant(pcudaStream_t stream, const param_probe::Scalar* input,
        param_probe::Scalar* output, std::size_t count, param_probe::Parameters parameters) {
    if (!stream || !input || !output || count > std::numeric_limits<std::size_t>::max() / sizeof(*output) ||
        count / 128 + (count % 128 != 0) > std::numeric_limits<unsigned>::max())
        return pcudaErrorInvalidValue;
    if (!count) return pcudaSuccess;
    const auto grid = static_cast<unsigned>(count / 128 + (count % 128 != 0));
    return pcudaParallelFor(grid, 128, 0, stream, [=]() {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count) {
            if constexpr (Identity == param_probe::tagIdentity) {
                auto value = param_probe::pack(input[i].value.i, input[i].valid != 0);
                value = param_probe::binary(value, parameters.slots[0], false);
                value = param_probe::binary(value, parameters.slots[1], true);
                value = param_probe::binary(value, parameters.slots[2], false);
                if constexpr (param_probe::parameterCount == 4)
                    value = param_probe::binary(value, parameters.slots[3], false);
                output[i] = value;
            } else {
                output[i] = param_probe::typed<Identity>(input[i], parameters);
            }
        }
    });
}

extern "C" pcudaError_t launch_parameters(pcudaStream_t stream, const param_probe::Scalar* input,
        param_probe::Scalar* output, std::size_t count, param_probe::Parameters parameters) {
#ifdef STAGE0_RUNTIME_TAGS
    return launch_variant<param_probe::tagIdentity>(stream, input, output, count, parameters);
#else
    return launch_variant<STAGE0_TYPE_MASK>(stream, input, output, count, parameters);
#endif
}
