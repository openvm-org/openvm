#include "poseidon2/columns.cuh"
#include "poseidon2/tracegen.cuh"
#include "specific.cuh"
#include "system/memory/controller.cuh"

using namespace poseidon2;

static const size_t WIDTH = 16;
static const size_t SBOX_DEGREE = 7;
static const size_t HALF_FULL_ROUNDS = 4;
static const size_t PARTIAL_ROUNDS = 13;

static const uint32_t NUM_INITIAL_READS = 6;
// static const uint32_t NUM_SIMPLE_ACCESSES = 7;

template <typename T, size_t SBOX_REGISTERS> struct NativePoseidon2Cols {
    Poseidon2SubCols<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> inner;

    T incorporate_row;
    T incorporate_sibling;
    T inside_row;
    T simple;

    T end_inside_row;
    T end_top_level;
    T start_top_level;

    T very_first_timestamp;
    T start_timestamp;
    T opened_element_size_inv;
    T initial_opened_index;
    T opened_base_pointer;

    T is_exhausted[CHUNK - 1];
    T specific[COL_SPECIFIC_WIDTH];
};

__device__ void mem_fill_base(
    MemoryAuxColsFactory mem_helper,
    uint32_t timestamp,
    RowSlice base_aux
) {
    uint32_t prev = base_aux[COL_INDEX(MemoryBaseAuxCols, prev_timestamp)].asUInt32();
    mem_helper.fill(base_aux, prev, timestamp);
}

template <size_t SBOX_REGISTERS> struct Poseidon2Wrapper {
    template <typename T> using Cols = NativePoseidon2Cols<T, SBOX_REGISTERS>;
    using Poseidon2Row =
        Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

    __device__ static void fill_inner(RowSlice row) {
        Poseidon2Row poseidon2_row(row);
        Fp state[WIDTH];
        {
            RowSlice inputs = poseidon2_row.inputs();
            for (size_t i = 0; i < WIDTH; ++i) {
                state[i] = inputs[i];
            }
        }
        generate_trace_row_for_perm<
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS>(poseidon2_row, RowSlice(state, 1));
    }

    __device__ static void fill_specific(RowSlice row, VariableRangeChecker range_checker) {
        RowSlice specific = row.slice_from(COL_INDEX(Cols, specific));
        MemoryAuxColsFactory mem_helper(range_checker);
        uint32_t start_timestamp = row[COL_INDEX(Cols, start_timestamp)].asUInt32();

        if (row[COL_INDEX(Cols, simple)] == Fp::one()) {
            mem_fill_base(
                mem_helper,
                start_timestamp,
                specific.slice_from(COL_INDEX(SimplePoseidonSpecificCols, read_output_pointer.base))
            );
            mem_fill_base(
                mem_helper,
                start_timestamp + 1,
                specific.slice_from(COL_INDEX(SimplePoseidonSpecificCols, read_input_pointer_1.base)
                )
            );
            mem_fill_base(
                mem_helper,
                start_timestamp + 3,
                specific.slice_from(COL_INDEX(SimplePoseidonSpecificCols, read_data_1.base))
            );
            mem_fill_base(
                mem_helper,
                start_timestamp + 4,
                specific.slice_from(COL_INDEX(SimplePoseidonSpecificCols, read_data_2.base))
            );
            mem_fill_base(
                mem_helper,
                start_timestamp + 5,
                specific.slice_from(COL_INDEX(SimplePoseidonSpecificCols, write_data_1.base))
            );
            if (specific[COL_INDEX(SimplePoseidonSpecificCols, is_compress)] == Fp::one()) {
                mem_fill_base(
                    mem_helper,
                    start_timestamp + 2,
                    specific.slice_from(
                        COL_INDEX(SimplePoseidonSpecificCols, read_input_pointer_2.base)
                    )
                );
            } else {
                mem_fill_base(
                    mem_helper,
                    start_timestamp + 6,
                    specific.slice_from(COL_INDEX(SimplePoseidonSpecificCols, write_data_2.base))
                );
            }
        } else if (row[COL_INDEX(Cols, inside_row)] == Fp::one()) {
            for (uint32_t i = 0; i < CHUNK; i++) {
                if (i > 0 && row[COL_INDEX(Cols, is_exhausted[i - 1])] == Fp::one()) {
                    break;
                } else if (specific[COL_INDEX(InsideRowSpecificCols, cells[i].is_first_in_row)] ==
                           Fp::one()) {
                    mem_fill_base(
                        mem_helper,
                        start_timestamp + (2 * i),
                        specific.slice_from(COL_INDEX(
                            InsideRowSpecificCols, cells[i].read_row_pointer_and_length.base
                        ))
                    );
                }
                mem_fill_base(
                    mem_helper,
                    start_timestamp + (2 * i) + 1,
                    specific.slice_from(COL_INDEX(InsideRowSpecificCols, cells[i].read.base))
                );
            }
        } else {
            if (row[COL_INDEX(Cols, end_top_level)] == Fp::one()) {
                uint32_t very_start_timestamp =
                    row[COL_INDEX(Cols, very_first_timestamp)].asUInt32();
                mem_fill_base(
                    mem_helper,
                    very_start_timestamp,
                    specific.slice_from(COL_INDEX(TopLevelSpecificCols, dim_base_pointer_read.base))
                );
                mem_fill_base(
                    mem_helper,
                    very_start_timestamp + 1,
                    specific.slice_from(
                        COL_INDEX(TopLevelSpecificCols, opened_base_pointer_read.base)
                    )
                );
                mem_fill_base(
                    mem_helper,
                    very_start_timestamp + 2,
                    specific.slice_from(COL_INDEX(TopLevelSpecificCols, opened_length_read.base))
                );
                mem_fill_base(
                    mem_helper,
                    very_start_timestamp + 3,
                    specific.slice_from(
                        COL_INDEX(TopLevelSpecificCols, index_base_pointer_read.base)
                    )
                );
                mem_fill_base(
                    mem_helper,
                    very_start_timestamp + 4,
                    specific.slice_from(COL_INDEX(TopLevelSpecificCols, commit_pointer_read.base))
                );
                mem_fill_base(
                    mem_helper,
                    very_start_timestamp + 5,
                    specific.slice_from(COL_INDEX(TopLevelSpecificCols, commit_read.base))
                );
            }
            if (row[COL_INDEX(Cols, incorporate_row)] == Fp::one()) {
                uint32_t end_timestamp =
                    specific[COL_INDEX(TopLevelSpecificCols, end_timestamp)].asUInt32();
                mem_fill_base(
                    mem_helper,
                    end_timestamp - 2,
                    specific.slice_from(COL_INDEX(
                        TopLevelSpecificCols, read_initial_height_or_sibling_is_on_right.base
                    ))
                );
                mem_fill_base(
                    mem_helper,
                    end_timestamp - 1,
                    specific.slice_from(COL_INDEX(TopLevelSpecificCols, read_final_height.base))
                );
            } else if (row[COL_INDEX(Cols, incorporate_sibling)] == Fp::one()) {
                mem_fill_base(
                    mem_helper,
                    start_timestamp + NUM_INITIAL_READS,
                    specific.slice_from(COL_INDEX(
                        TopLevelSpecificCols, read_initial_height_or_sibling_is_on_right.base
                    ))
                );
            }
        }
    }
};

template <size_t SBOX_REGISTERS>
__global__ void cukernel_inplace_native_poseidon2_tracegen(
    Fp *trace,
    size_t trace_height,
    size_t trace_width,
    size_t num_records,
    uint32_t *rc_buffer,
    uint32_t rc_num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, trace_height);
    Poseidon2Wrapper<SBOX_REGISTERS>::fill_inner(row);
    if (idx < num_records) {
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_specific(
            row, VariableRangeChecker(rc_buffer, rc_num_bins)
        );
    }
}

template <size_t SBOX_REGISTERS>
__global__ void cukernel_native_poseidon2_tracegen(
    Fp *trace,
    size_t trace_height,
    size_t trace_width,
    Fp *records,
    size_t num_records,
    uint32_t *rc_buffer,
    uint32_t rc_num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, trace_height);
    if (idx < num_records) {
        RowSlice record(records + idx * num_records, 1);
        for (uint32_t i = 0; i < trace_width; i++) {
            row[i] = record[i];
        }
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_inner(row);
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_specific(
            row, VariableRangeChecker(rc_buffer, rc_num_bins)
        );
    } else {
        row.fill_zero(0, trace_width);
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_inner(row);
    }
}

extern "C" int _inplace_native_poseidon2_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    size_t num_records,
    uint32_t *d_rc_buffer,
    uint32_t rc_num_bins,
    size_t sbox_regs
) {
    auto [grid, block] = kernel_launch_params(height);
    switch (sbox_regs) {
    case 1:
        assert(width == sizeof(NativePoseidon2Cols<uint8_t, 1>));
        cukernel_inplace_native_poseidon2_tracegen<1>
            <<<grid, block>>>(d_trace, height, width, num_records, d_rc_buffer, rc_num_bins);
        break;
    case 0:
        assert(width == sizeof(NativePoseidon2Cols<uint8_t, 0>));
        cukernel_inplace_native_poseidon2_tracegen<0>
            <<<grid, block>>>(d_trace, height, width, num_records, d_rc_buffer, rc_num_bins);
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

extern "C" int _native_poseidon2_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp *d_records,
    size_t num_records,
    uint32_t *d_rc_buffer,
    uint32_t rc_num_bins,
    size_t sbox_regs
) {
    auto [grid, block] = kernel_launch_params(height);
    switch (sbox_regs) {
    case 1:
        assert(width == sizeof(NativePoseidon2Cols<uint8_t, 1>));
        cukernel_native_poseidon2_tracegen<1><<<grid, block>>>(
            d_trace, height, width, d_records, num_records, d_rc_buffer, rc_num_bins
        );
        break;
    case 0:
        assert(width == sizeof(NativePoseidon2Cols<uint8_t, 0>));
        cukernel_native_poseidon2_tracegen<0><<<grid, block>>>(
            d_trace, height, width, d_records, num_records, d_rc_buffer, rc_num_bins
        );
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}
