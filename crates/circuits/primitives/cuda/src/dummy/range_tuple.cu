#include "fp.h"
#include "launcher.cuh"
#include "primitives/histogram.cuh"

template <uint32_t RANGE_TUPLE_SIZE>
__global__ void range_tuple_dummy_tracegen(
    const uint32_t *data,
    Fp *trace,
    uint32_t *rc_count,
    size_t data_height,
    uint32_t *sizes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // We do this because of the template length of the sizes tuple, but in
    // practice a) know what tuples the chip needs (typically 2) and b) pass in
    // the sizes by value instead of through a buffer.
    uint32_t sizes_array[RANGE_TUPLE_SIZE];
    for (uint32_t i = 0; i < RANGE_TUPLE_SIZE; i++) {
        sizes_array[i] = sizes[i];
    }

    RangeTupleChecker<RANGE_TUPLE_SIZE> range_checker(rc_count, sizes);

    if (idx < data_height) {
        trace[idx] = Fp::one();
        for (uint32_t i = 0; i < RANGE_TUPLE_SIZE; i++) {
            trace[idx + data_height * (i + 1)] = Fp(data[idx * RANGE_TUPLE_SIZE + i]);
        }
        RowSlice values(trace + idx + data_height, data_height);
        range_checker.add_count(values);
    }

    range_checker.merge(rc_count);
}

extern "C" int _range_tuple_dummy_tracegen(
    const uint32_t *d_data,
    Fp *d_trace,
    uint32_t *d_rc_count,
    size_t data_height,
    uint32_t *sizes,
    size_t sizes_len
) {
    auto [grid, block] = kernel_launch_params(data_height);
    switch (sizes_len) {
    case 1: {
        range_tuple_dummy_tracegen<1>
            <<<grid, block>>>(d_data, d_trace, d_rc_count, data_height, sizes);
        break;
    }
    case 2: {
        range_tuple_dummy_tracegen<2>
            <<<grid, block>>>(d_data, d_trace, d_rc_count, data_height, sizes);
        break;
    }
    case 3: {
        range_tuple_dummy_tracegen<3>
            <<<grid, block>>>(d_data, d_trace, d_rc_count, data_height, sizes);
        break;
    }
    case 4: {
        range_tuple_dummy_tracegen<4>
            <<<grid, block>>>(d_data, d_trace, d_rc_count, data_height, sizes);
        break;
    }
    default:
        return -1;
    }
    return CHECK_KERNEL();
}
