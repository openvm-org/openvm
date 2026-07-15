#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/rvr_compact.cuh"

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

    __device__ void fill_trace_row(RowSlice row, LoadByteRecord record, uint8_t shift) {
        uint16_t read_cell = record.read_data[shift >> 1];
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
        COL_WRITE_ARRAY(row, LoadSignExtendByteCoreCols, read_data, record.read_data);
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

__global__ void rv64_load_sign_extend_byte_tracegen_compact(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const rec = records[idx];
        auto const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        Rv64LoadSignExtendRecord full;
        full.adapter = rvr_decode_alu3_load(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.read_data[i] = rvr_u16_limb(rec.c, i);
        }
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);
        auto core = LoadSignExtendByteCore(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendByteCols, core)),
            full.core,
            rv64_load_shift_amount(full.adapter)
        );
    } else {
        row.fill_zero(0, sizeof(Rv64LoadSignExtendByteCols<uint8_t>));
    }
}

extern "C" int _rv64_load_sign_extend_byte_tracegen_compact(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendByteCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_byte_tracegen_compact<<<grid, block, 0, stream>>>(
        trace,
        height,
        records,
        operand_table,
        pc_base,
        pointer_max_bits,
        range_checker,
        range_checker_num_bins,
        bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
