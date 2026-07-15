#include "riscv/cores/store.cuh"
#include "riscv/rvr_compact.cuh"

template <typename T> struct StoreDoublewordCoreCols {
    T selector[STORE_DOUBLEWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64StoreDoublewordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreDoublewordCoreCols<T> core;
};

struct StoreDoublewordCore {
    __device__ void fill_trace_row(RowSlice row, StoreRecord record, uint8_t shift) {
        assert(shift == 0);

        Encoder encoder(
            STORE_DOUBLEWORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, 0);
        row[STORE_DOUBLEWORD_SELECTOR_WIDTH] = 1;
        row.write_array(STORE_DOUBLEWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            STORE_DOUBLEWORD_SELECTOR_WIDTH + 1 + BLOCK_FE_WIDTH,
            BLOCK_FE_WIDTH,
            record.prev_data
        );
    }
};

__global__ void rv64_store_doubleword_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        StoreDoublewordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreDoublewordCols, core)),
            record.core,
            rv64_store_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreDoublewordCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_doubleword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64StoreDoublewordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_doubleword_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void rv64_store_doubleword_tracegen_compact(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const rec = records[idx];
        auto const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        Rv64StoreRecord full;
        full.adapter = rvr_decode_alu3_store(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.read_data[i] = rvr_u16_limb(rec.c, i);
            full.core.prev_data[i] = rvr_u16_limb(rec.write_prev_data, i);
        }
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);
        StoreDoublewordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreDoublewordCols, core)),
            full.core,
            rv64_store_shift_amount(full.adapter)
        );
    } else {
        row.fill_zero(0, sizeof(Rv64StoreDoublewordCols<uint8_t>));
        COL_WRITE_VALUE(row, Rv64StoreDoublewordCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_doubleword_tracegen_compact(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64StoreDoublewordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_doubleword_tracegen_compact<<<grid, block, 0, stream>>>(
        trace,
        height,
        records,
        operand_table,
        pc_base,
        pointer_max_bits,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
