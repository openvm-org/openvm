#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/load_byte_replay.cuh"

template <typename T> struct LoadSignExtendByteCoreCols {
    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    T data_most_sig_bit;
    T read_cell_lo_byte;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadSignExtendByteCols {
    Rv64LoadByteAdapterCols<T> adapter;
    LoadSignExtendByteCoreCols<T> core;
};

struct LoadSignExtendByteCore {
    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadSignExtendByteCore(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : range_checker(range_checker), bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(
        RowSlice row, uint16_t const (&read_data)[BLOCK_FE_WIDTH], uint8_t shift
    ) {
        uint16_t read_cell = read_data[shift >> 1];
        uint16_t read_cell_bytes[2] = {
            load_sign_extend_byte_from_cell(read_cell, 0),
            load_sign_extend_byte_from_cell(read_cell, 1),
        };
        uint16_t selected_byte = read_cell_bytes[shift & 1];
        uint16_t sign_bit = selected_byte & SIGN_BYTE;

        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);
        range_checker.add_count(selected_byte - sign_bit, RV64_BYTE_BITS - 1);

        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(LoadSignExtendByteCoreCols, selector)),
            shift
        );
        COL_WRITE_VALUE(row, LoadSignExtendByteCoreCols, data_most_sig_bit, sign_bit != 0);
        COL_WRITE_VALUE(
            row, LoadSignExtendByteCoreCols, read_cell_lo_byte, read_cell_bytes[0]
        );
        COL_WRITE_ARRAY(row, LoadSignExtendByteCoreCols, read_data, read_data);
    }

    __device__ void fill_trace_row(RowSlice row, LoadByteRecord record, uint8_t shift) {
        fill_trace_row(row, record.read_data, shift);
    }
};

__global__ void rv64_load_sign_extend_byte_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadByteRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64LoadByteAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = LoadSignExtendByteCore(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendByteCols, core)),
            record.core,
            rv64_load_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_load_sign_extend_byte_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t opcode,
    uint32_t register_as,
    uint32_t memory_as,
    size_t pointer_max_bits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64LoadSignExtendByteCols<uint8_t>));
    if (idx >= num_steps) return;

    ReplayLoadByteInput input = {};
    if (!replay_load_byte(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + idx],
            opcode,
            register_as,
            memory_as,
            pointer_max_bits,
            true,
            input,
            error
        )) {
        return;
    }

    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    auto adapter = Rv64LoadByteAdapter(pointer_max_bits, checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        input.from_pc,
        input.from_timestamp,
        input.rs1_ptr,
        input.rd_ptr,
        input.needs_write,
        input.rs1_val,
        input.rs1_prev_timestamp,
        input.read_prev_timestamp,
        input.write_prev_timestamp,
        input.write_prev_data,
        input.imm,
        input.imm_sign
    );
    auto core = LoadSignExtendByteCore(checker, BitwiseOperationLookup(bitwise_lookup));
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64LoadSignExtendByteCols, core)),
        input.read_data,
        input.shift
    );
}

extern "C" int _rv64_load_sign_extend_byte_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadByteRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendByteCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_byte_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _rv64_load_sign_extend_byte_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory,
    DeviceBufferConstView<PreflightInitialWrite> d_seeds,
    DeviceBufferConstView<uint32_t> d_predecessors,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *d_error,
    uint32_t opcode,
    uint32_t register_as,
    uint32_t memory_as,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendByteCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_byte_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        d_error,
        opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
