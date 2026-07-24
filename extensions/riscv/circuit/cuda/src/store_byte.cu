#include "riscv/cores/store.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

template <typename T> struct StoreByteCoreCols {
    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    // Low byte of the first source register cell; the high byte is derived in the AIR.
    T read_lo_byte;
    // Low byte of the selected previous memory cell; the high byte is derived in the AIR.
    T prev_cell_lo_byte;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64StoreByteCols {
    Rv64StoreByteAdapterCols<T> adapter;
    StoreByteCoreCols<T> core;
};

struct StoreByteCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreByteCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, StoreByteRecord record, uint8_t shift) {
        uint8_t cell_shift = shift >> 1;

        uint16_t read_lo_byte = store_byte_from_cell(record.read_data[0], 0);
        uint16_t prev_cell_bytes[2] = {
            store_byte_from_cell(record.prev_data[cell_shift], 0),
            store_byte_from_cell(record.prev_data[cell_shift], 1),
        };
        bitwise_lookup.add_range(read_lo_byte, store_byte_from_cell(record.read_data[0], 1));
        bitwise_lookup.add_range(prev_cell_bytes[0], prev_cell_bytes[1]);

        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(row.slice_from(COL_INDEX(StoreByteCoreCols, selector)), shift);
        COL_WRITE_VALUE(row, StoreByteCoreCols, read_lo_byte, read_lo_byte);
        COL_WRITE_VALUE(row, StoreByteCoreCols, prev_cell_lo_byte, prev_cell_bytes[0]);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, prev_data, record.prev_data);
    }
};

__global__ void rv64_store_byte_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreByteRecord> records,
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
        auto adapter = Rv64StoreByteAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = StoreByteCore(BitwiseOperationLookup(bitwise_lookup_ptr));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreByteCols, core)),
            record.core,
            rv64_store_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreByteCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_byte_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreByteRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64StoreByteCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_byte_tracegen<<<grid, block, 0, stream>>>(
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

template <typename RecordView>
__global__ void rv64_store_byte_tracegen_compact(
    Fp *trace,
    size_t height,
    RecordView records,
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
        Rv64StoreByteRecord full;
        full.adapter = rvr_decode_alu3_store_byte(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.read_data[i] = rvr_u16_limb(rec.c, i);
            full.core.prev_data[i] = rvr_u16_limb(rec.write_prev_data, i);
        }
        auto adapter = Rv64StoreByteAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);
        auto core = StoreByteCore(BitwiseOperationLookup(bitwise_lookup_ptr));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreByteCols, core)),
            full.core,
            rv64_store_shift_amount(full.adapter)
        );
    } else {
        row.fill_zero(0, sizeof(Rv64StoreByteCols<uint8_t>));
        COL_WRITE_VALUE(row, Rv64StoreByteCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_byte_tracegen_compact(
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
#ifdef OPENVM_RVR_CUDA_G2_ONLY
    return int(cudaErrorNotSupported);
#else
    assert(width == sizeof(Rv64StoreByteCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_byte_tracegen_compact<<<grid, block, 0, stream>>>(
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
#endif
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _rv64_store_byte_tracegen_g2, Rv64StoreByteCols, rv64_store_byte_tracegen_compact,
    RvrAlu3Compact, 512, operand_table, pc_base, pointer_max_bits, range_checker,
    range_checker_num_bins, bitwise_lookup, timestamp_max_bits
)
