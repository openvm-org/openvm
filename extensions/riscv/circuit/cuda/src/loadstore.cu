#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/loadstore.cuh"

using namespace riscv;
using namespace program;

// Every (opcode, shift) pair is a selector case: 8 opcodes (LOADD..STOREB, transpiler order
// 0..=7) times 8 byte shifts. Case index = opcode * 8 + shift.
constexpr uint32_t LOADSTORE_SELECTOR_CASES = 64;
constexpr uint32_t LOADSTORE_SELECTOR_MAX_DEGREE = 2;
constexpr size_t LOADSTORE_SELECTOR_WIDTH = 10;

template <typename T, size_t NUM_CELLS> struct LoadStoreCoreCols {
    T selector[LOADSTORE_SELECTOR_WIDTH];
    /// we need to keep the degree of is_valid, is_load and the cross flags to 1
    T is_valid;
    T is_load;
    /// 1 iff this is a load spanning two blocks; multiplicity of the second block read.
    T load_cross;
    /// 1 iff this is a store spanning two blocks; multiplicity of the second block write.
    T store_cross;

    T read_data[NUM_CELLS];
    T read_data1[NUM_CELLS];
    T prev_data[NUM_CELLS];
    T prev_data1[NUM_CELLS];
    T write_data[NUM_CELLS];
    T write_data1[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadStoreCoreRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
    uint8_t read_data1[NUM_CELLS];
    uint8_t prev_data[NUM_CELLS];
    uint8_t prev_data1[NUM_CELLS];
};

// Computes [write_data, write_data1]: the new rd value (loads, write_data1 unused) or the
// new contents of the touched block(s) (stores).
template <size_t NUM_CELLS>
__device__ __forceinline__ void run_write_data(
    uint8_t (&write_data)[NUM_CELLS],
    uint8_t (&write_data1)[NUM_CELLS],
    uint8_t local_opcode,
    const uint8_t (&read_data)[NUM_CELLS],
    const uint8_t (&read_data1)[NUM_CELLS],
    const uint8_t (&prev_data)[NUM_CELLS],
    const uint8_t (&prev_data1)[NUM_CELLS],
    uint8_t shift
) {
    bool is_load = rv64_loadstore_is_load(local_opcode);
    uint32_t width = RV64_LOADSTORE_ACCESS_WIDTH[local_opcode];

    if (is_load) {
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            if (i < width) {
                size_t src = i + shift;
                write_data[i] = (src < NUM_CELLS) ? read_data[src] : read_data1[src - NUM_CELLS];
            } else {
                write_data[i] = 0u;
            }
            write_data1[i] = 0u;
        }
    } else {
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            bool in_range = (i >= shift) && (i < shift + width);
            write_data[i] = in_range ? read_data[i - shift] : prev_data[i];
            size_t global = NUM_CELLS + i;
            bool spills = (shift + width > NUM_CELLS) && (global < shift + width);
            write_data1[i] = spills ? read_data[global - shift]
                                    : ((shift + width > NUM_CELLS) ? prev_data1[i] : 0u);
        }
    }
}

template <size_t NUM_CELLS> struct LoadStoreCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadStoreCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    template <typename T> using Cols = LoadStoreCoreCols<T, NUM_CELLS>;

    __device__ void fill_trace_row(RowSlice row, LoadStoreCoreRecord<NUM_CELLS> record) {
        Encoder encoder(
            LOADSTORE_SELECTOR_CASES,
            LOADSTORE_SELECTOR_MAX_DEGREE,
            true
        );
        uint8_t shift = record.shift_amount;
        bool is_load = rv64_loadstore_is_load(record.local_opcode);
        bool crosses = shift + RV64_LOADSTORE_ACCESS_WIDTH[record.local_opcode] > NUM_CELLS;
        uint8_t write_data[NUM_CELLS] = {0};
        uint8_t write_data1[NUM_CELLS] = {0};

        COL_WRITE_VALUE(row, Cols, is_valid, 1);
        COL_WRITE_VALUE(row, Cols, is_load, is_load);
        COL_WRITE_VALUE(row, Cols, load_cross, is_load && crosses);
        COL_WRITE_VALUE(row, Cols, store_cross, !is_load && crosses);
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(Cols, selector)),
            uint32_t(record.local_opcode) * NUM_CELLS + shift
        );
        COL_WRITE_ARRAY(row, Cols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, Cols, read_data1, record.read_data1);
        COL_WRITE_ARRAY(row, Cols, prev_data, record.prev_data);
        COL_WRITE_ARRAY(row, Cols, prev_data1, record.prev_data1);
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i += 2) {
            bitwise_lookup.add_range(record.read_data[i], record.read_data[i + 1]);
            bitwise_lookup.add_range(record.prev_data[i], record.prev_data[i + 1]);
        }
        // The second-block witnesses are only live on block-spanning rows; range check them
        // exactly there (loads check read_data1, stores check prev_data1).
        if (crosses) {
            const uint8_t *checked_block1 = is_load ? record.read_data1 : record.prev_data1;
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS; i += 2) {
                bitwise_lookup.add_range(checked_block1[i], checked_block1[i + 1]);
            }
        }

        run_write_data(
            write_data,
            write_data1,
            record.local_opcode,
            record.read_data,
            record.read_data1,
            record.prev_data,
            record.prev_data1,
            shift
        );
        COL_WRITE_ARRAY(row, Cols, write_data, write_data);
        COL_WRITE_ARRAY(row, Cols, write_data1, write_data1);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct Rv64LoadStoreCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadStoreCoreCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct Rv64LoadStoreRecord {
    Rv64LoadStoreAdapterRecord adapter;
    LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS> core;
};

__global__ void rv64_load_store_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> records,
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

        auto adapter = Rv64LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = LoadStoreCore<RV64_REGISTER_NUM_LIMBS>(
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadStoreCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64LoadStoreCols<uint8_t>));
    }
}

extern "C" int _rv64_load_store_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadStoreCols<uint8_t>));
    // The per-thread register footprint of this kernel (two-block merge, six byte arrays)
    // is too large for the default 1024-thread blocks; 256 keeps the launch within the
    // SM register file for any register count.
    auto [grid, block] = kernel_launch_params(height, 256);

    rv64_load_store_tracegen<<<grid, block, 0, stream>>>(
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
