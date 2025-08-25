#include "native/poseidon2.cuh"
#include "poseidon2-air/columns.cuh"
#include "poseidon2-air/params.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace poseidon2;

static const size_t WIDTH = 16;
static const size_t SBOX_DEGREE = Poseidon2DefaultParams::SBOX_DEGREE;
static const size_t HALF_FULL_ROUNDS = Poseidon2DefaultParams::HALF_FULL_ROUNDS;
static const size_t PARTIAL_ROUNDS = Poseidon2DefaultParams::PARTIAL_ROUNDS;

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

    __device__ static void fill_trace(
        RowSlice row,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    ) {
        if (row[COL_INDEX(Cols, simple)] == Fp::one()) {
            fill_simple_chunk(row, range_checker, timestamp_max_bits);
        } else {
            fill_verify_batch_chunk(row, range_checker, timestamp_max_bits);
        }
    }

    __device__ static void fill_simple_chunk(
        RowSlice row,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    ) {
        fill_inner(row);
        fill_specific(row, range_checker, timestamp_max_bits);
    }

    __device__ static void fill_verify_batch_chunk(
        RowSlice row,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    ) {
        Poseidon2Row first_p2_row(row);
        uint32_t num_non_inside_rows = first_p2_row.export_col()[0].asUInt32();
        RowSlice last_non_inside_row = row.shift_row(num_non_inside_rows - 1);
        Poseidon2Row last_non_inside_p2_row(last_non_inside_row);
        uint32_t total_num_rows = last_non_inside_p2_row.export_col()[0].asUInt32();

        bool first_round = true;
        Fp root[CHUNK];
        uint32_t inside_idx = num_non_inside_rows;
        uint32_t non_inside_idx = 0;
        while (inside_idx < total_num_rows || non_inside_idx < num_non_inside_rows) {
            RowSlice curr_non_inside_row = row.shift_row(non_inside_idx);
            bool incorporate_sibling =
                curr_non_inside_row[COL_INDEX(Cols, incorporate_sibling)] == Fp::one();
            if (!incorporate_sibling) {
                Fp prev_rolling_hash[WIDTH];
                // `Fp`'s constructor will set the values to 0s.
                Fp rolling_hash[WIDTH];
                do {
                    RowSlice curr_inside_row = row.shift_row(inside_idx);
                    uint32_t input_len = 0;
                    uint32_t start_timestamp_u32 =
                        curr_inside_row[COL_INDEX(Cols, start_timestamp)].asUInt32();

                    fill_specific(curr_inside_row, range_checker, timestamp_max_bits);
                    for (uint32_t i = 0; i < CHUNK; i++) {
                        if (i > 0 &&
                            curr_inside_row[COL_INDEX(Cols, is_exhausted[i - 1])] == Fp::one()) {
                            break;
                        }
                        input_len += 1;
                    }

                    Poseidon2Row poseidon2_row(curr_inside_row);
                    RowSlice inputs = poseidon2_row.inputs();

                    for (uint32_t i = 0; i < input_len; i++) {
                        rolling_hash[i] = inputs[i];
                    }
                    for (size_t i = 0; i < WIDTH; ++i) {
                        prev_rolling_hash[i] = rolling_hash[i];
                        inputs[i] = rolling_hash[i];
                    }
                    fill_inner(curr_inside_row);
                    RowSlice outputs = poseidon2_row.outputs();
                    for (size_t i = 0; i < WIDTH; ++i) {
                        rolling_hash[i] = outputs[i];
                    }
                    inside_idx += 1;
                    if (curr_inside_row[COL_INDEX(Cols, end_inside_row)] == Fp::one()) {
                        break;
                    }
                } while (true);

                {
                    RowSlice curr_non_inside_row = row.shift_row(non_inside_idx);

                    Poseidon2Row poseidon2_row(curr_non_inside_row);
                    RowSlice inputs = poseidon2_row.inputs();
                    if (first_round) {
                        for (size_t i = 0; i < WIDTH; ++i) {
                            inputs[i] = prev_rolling_hash[i];
                        }
                        first_round = false;
                    } else {
                        for (size_t i = 0; i < CHUNK; ++i) {
                            inputs[i] = root[i];
                            inputs[i + CHUNK] = rolling_hash[i];
                        }
                    }
                    fill_inner(curr_non_inside_row);
                    fill_specific(curr_non_inside_row, range_checker, timestamp_max_bits);
                    RowSlice outputs = poseidon2_row.outputs();
                    for (size_t i = 0; i < CHUNK; ++i) {
                        root[i] = outputs[i];
                    }
                    non_inside_idx += 1;
                }
            }
            if (non_inside_idx < num_non_inside_rows) {
                RowSlice curr_non_inside_row = row.shift_row(non_inside_idx);
                RowSlice curr_specific = curr_non_inside_row.slice_from(COL_INDEX(Cols, specific));
                Poseidon2Row poseidon2_row(curr_non_inside_row);
                RowSlice inputs = poseidon2_row.inputs();
                if (curr_specific[COL_INDEX(TopLevelSpecificCols, sibling_is_on_right)] ==
                    Fp::one()) {
                    for (size_t i = 0; i < CHUNK; ++i) {
                        // `sibling` is already put in inputs[..CHUNK] during execution.
                        inputs[i + CHUNK] = root[i];
                    }
                } else {
                    for (size_t i = 0; i < CHUNK; ++i) {
                        inputs[i + CHUNK] = inputs[i];
                        inputs[i] = root[i];
                    }
                }
                fill_inner(curr_non_inside_row);
                fill_specific(curr_non_inside_row, range_checker, timestamp_max_bits);
                RowSlice outputs = poseidon2_row.outputs();
                for (size_t i = 0; i < CHUNK; ++i) {
                    root[i] = outputs[i];
                }
                non_inside_idx += 1;
            }
        }
    }

    __device__ static void fill_inner(RowSlice row) {
        Poseidon2Row poseidon2_row(row);
        Fp state[WIDTH];
        {
            RowSlice inputs = poseidon2_row.inputs();
            for (size_t i = 0; i < WIDTH; ++i) {
                state[i] = inputs[i];
            }
        }
        generate_trace_row_for_perm(poseidon2_row, RowSlice(state, 1));
    }

    __device__ static void fill_specific(
        RowSlice row,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    ) {
        RowSlice specific = row.slice_from(COL_INDEX(Cols, specific));
        MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);
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
                specific.slice_from(
                    COL_INDEX(SimplePoseidonSpecificCols, read_input_pointer_1.base)
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
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, trace_height);
    Poseidon2Wrapper<SBOX_REGISTERS>::fill_inner(row);
    if (idx < num_records) {
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_specific(
            row, VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
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
    uint32_t *d_chunk_start,
    uint32_t num_chunks,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each chunk is a contiguous block of rows in the trace. Each empty row is a chunk.
    // `d_chunk_start` only contains the start indices of non-empty chunks.
    uint32_t start_idx =
        chunk_idx < num_chunks ? d_chunk_start[chunk_idx] : (num_records + chunk_idx - num_chunks);
    RowSlice row(trace + start_idx, trace_height);
    if (chunk_idx < num_chunks) {
        Fp *record = records + start_idx * trace_width;
        uint32_t chunk_height =
            (chunk_idx + 1 < num_chunks ? d_chunk_start[chunk_idx + 1] : num_records) -
            d_chunk_start[chunk_idx];
        // Transpose `record` and copy to `trace`.
        for (uint32_t r = 0; r < chunk_height; r++) {
            RowSlice curr_row = row.shift_row(r);
            for (uint32_t c = 0; c < trace_width; c++) {
                curr_row[c] = record[r * trace_width + c];
            }
        }
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_trace(
            row, VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
        );
    } else if (start_idx < trace_height) {
        row.fill_zero(0, trace_width);
        Poseidon2Wrapper<SBOX_REGISTERS>::fill_inner(row);
    }
}

extern "C" int _native_poseidon2_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp *d_records,
    size_t num_records,
    uint32_t *d_chunk_start,
    uint32_t num_chunks,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t sbox_regs,
    uint32_t timestamp_max_bits
) {
    auto [grid, block] = kernel_launch_params(height - num_records + num_chunks, 256);
    switch (sbox_regs) {
    case 1:
        assert(width == sizeof(NativePoseidon2Cols<uint8_t, 1>));
        cukernel_native_poseidon2_tracegen<1><<<grid, block>>>(
            d_trace,
            height,
            width,
            d_records,
            num_records,
            d_chunk_start,
            num_chunks,
            d_range_checker,
            range_checker_num_bins,
            timestamp_max_bits
        );
        break;
    case 0:
        assert(width == sizeof(NativePoseidon2Cols<uint8_t, 0>));
        cukernel_native_poseidon2_tracegen<0><<<grid, block>>>(
            d_trace,
            height,
            width,
            d_records,
            num_records,
            d_chunk_start,
            num_chunks,
            d_range_checker,
            range_checker_num_bins,
            timestamp_max_bits
        );
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}
