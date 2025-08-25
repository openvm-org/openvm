#include "launcher.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/trace_access.h"

template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
__global__ void cukernel_poseidon2_tracegen(Fp *output, Fp *inputs, uint32_t n) {

    // RowMajor Input, ColMajor Output

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    poseidon2::Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> row(
        output + tid, n
    );
    RowSlice state(inputs + tid * WIDTH, 1);
    poseidon2::generate_trace_row_for_perm<
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS>(row, state);
}

extern "C" int _poseidon2_dummy_tracegen(Fp *output, Fp *inputs, uint32_t sbox_regs, uint32_t n) {

    auto [grid, block] = kernel_launch_params(n);
    switch (sbox_regs) {
    case 1:
        cukernel_poseidon2_tracegen<16, 7, 1, 4, 13><<<grid, block>>>(output, inputs, n);
        break;
    case 0:
        cukernel_poseidon2_tracegen<16, 7, 0, 4, 13><<<grid, block>>>(output, inputs, n);
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}
