#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/loadstore.cuh"

using namespace riscv;
using namespace program;

template <typename T, size_t NUM_CELLS> struct LoadStoreCoreCols {
    T flags[4];
    /// we need to keep the degree of is_valid and is_load to 1
    T is_valid;
    T is_load;

    T read_data[NUM_CELLS];
    T prev_data[NUM_CELLS];
    /// write_data will be constrained against read_data and prev_data
    /// depending on the opcode and the shift amount
    T write_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadStoreCoreRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
    // Note: `prev_data` can be from native address space, so we need to use u32
    uint32_t prev_data[NUM_CELLS];
};

enum Rv32LoadStoreOpcode {
    LOADW,
    /// LOADBU, LOADHU are unsigned extend opcodes, implemented in the same chip with LOADW
    LOADBU,
    LOADHU,
    STOREW,
    STOREH,
    STOREB,
    /// The following are signed extend opcodes
    LOADB,
    LOADH,
};

template <size_t NUM_CELLS> struct LoadStoreCore {

    template <typename T> using Cols = LoadStoreCoreCols<T, NUM_CELLS>;

    __device__ void fill_trace_row(RowSlice row, LoadStoreCoreRecord<NUM_CELLS> record) {
        Rv32LoadStoreOpcode opcode = static_cast<Rv32LoadStoreOpcode>(record.local_opcode);

        COL_WRITE_VALUE(row, Cols, is_valid, 1);
        COL_WRITE_VALUE(
            row, Cols, is_load, (opcode == LOADW || opcode == LOADBU || opcode == LOADHU)
        );
        COL_WRITE_ARRAY(row, Cols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, Cols, prev_data, record.prev_data);

        uint8_t flags[4] = {0};
        uint32_t write_data[NUM_CELLS] = {0};
        uint8_t shift = record.shift_amount;

        switch (opcode) {
        case LOADW:
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS; i++) {
                write_data[i] = record.read_data[i];
            }
            flags[0] = 2;
            break;
        case LOADHU:
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS / 2; i++) {
                write_data[i] = record.read_data[i + shift];
            }
            switch (shift) {
            case 0:
                flags[1] = 2;
                break;
            case 2:
                flags[2] = 2;
            }
            break;
        case LOADBU:
            write_data[0] = record.read_data[shift];
            switch (shift) {
            case 0:
                flags[3] = 2;
                break;
            case 1:
                flags[0] = 1;
                break;
            case 2:
                flags[1] = 1;
                break;
            case 3:
                flags[2] = 1;
                break;
            }
            break;
        case STOREW:
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS; i++) {
                write_data[i] = record.read_data[i];
            }
            flags[3] = 1;
            break;
        case STOREH:
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS; i++) {
                if (i >= shift && i < (NUM_CELLS / 2 + shift)) {
                    write_data[i] = record.read_data[i - shift];
                } else {
                    write_data[i] = record.prev_data[i];
                }
            }
            switch (shift) {
            case 0:
                flags[0] = flags[1] = 1;
                break;
            case 2:
                flags[0] = flags[2] = 1;
                break;
            }
            break;
        case STOREB:
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS; i++) {
                write_data[i] = record.prev_data[i];
            }
            write_data[shift] = record.read_data[0];
            switch (shift) {
            case 0:
                flags[0] = flags[3] = 1;
                break;
            case 1:
                flags[1] = flags[2] = 1;
                break;
            case 2:
                flags[1] = flags[3] = 1;
                break;
            case 3:
                flags[2] = flags[3] = 1;
                break;
            }
            break;
        default:
            break;
        }

        COL_WRITE_ARRAY(row, Cols, flags, flags);
        COL_WRITE_ARRAY(row, Cols, write_data, write_data);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct Rv32LoadStoreCols {
    Rv32LoadStoreAdapterCols<T> adapter;
    LoadStoreCoreCols<T, RV32_REGISTER_NUM_LIMBS> core;
};

struct Rv32LoadStoreRecord {
    Rv32LoadStoreAdapterRecord adapter;
    LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS> core;
};

__global__ void rv32_load_store_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32LoadStoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv32LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = LoadStoreCore<RV32_REGISTER_NUM_LIMBS>();
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32LoadStoreCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv32LoadStoreCols<uint8_t>));
    }
}

extern "C" int _rv32_load_store_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(Rv32LoadStoreCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv32_load_store_tracegen<<<grid, block>>>(
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
