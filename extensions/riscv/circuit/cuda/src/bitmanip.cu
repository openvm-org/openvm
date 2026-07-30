#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/adapters/alu_reg.cuh"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/bitmanip.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct BitManipShAddCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    BitManipShAddCoreCols<T> core;
};

struct BitManipShAddRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    BitManipShAddCoreRecord core;
};

static_assert(sizeof(BitManipShAddRecord) == 60);
static_assert(offsetof(BitManipShAddRecord, core) == 40);

__global__ void rv64_bitmanip_shadd_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipShAddRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        BitManipShAddCore(VariableRangeChecker(range_ptr, range_bins))
            .fill_trace_row(row.slice_from(COL_INDEX(BitManipShAddCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_bitmanip_shadd_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipShAddRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitManipShAddCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_bitmanip_shadd_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename T> struct BitManipSlliUwCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    BitManipSlliUwCoreCols<T> core;
};

struct BitManipSlliUwRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    BitManipSlliUwCoreRecord core;
};

static_assert(sizeof(BitManipSlliUwRecord) == 44);
static_assert(offsetof(BitManipSlliUwRecord, core) == 32);

__global__ void rv64_bitmanip_slli_uw_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipSlliUwRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        BitManipSlliUwCore(VariableRangeChecker(range_ptr, range_bins))
            .fill_trace_row(row.slice_from(COL_INDEX(BitManipSlliUwCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_bitmanip_slli_uw_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipSlliUwRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitManipSlliUwCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_bitmanip_slli_uw_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename T> struct BitManipBitwiseInvCols {
    Rv64BaseAluRegAdapterCols<T> adapter;
    BitManipBitwiseInvCoreCols<T> core;
};

struct BitManipBitwiseInvRecord {
    Rv64BaseAluRegAdapterRecord adapter;
    BitManipBitwiseInvCoreRecord core;
};

__global__ void rv64_bitmanip_bitwise_inv_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipBitwiseInvRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluRegAdapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        BitManipBitwiseInvCore(BitwiseOperationLookup(bitwise_lookup_ptr))
            .fill_trace_row(row.slice_from(COL_INDEX(BitManipBitwiseInvCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_bitmanip_bitwise_inv_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipBitwiseInvRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitManipBitwiseInvCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_bitmanip_bitwise_inv_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename T> struct BitManipMinMaxCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    BitManipMinMaxCoreCols<T> core;
};

struct BitManipMinMaxRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    BitManipMinMaxCoreRecord core;
};

static_assert(sizeof(BitManipMinMaxRecord) == 60);
static_assert(offsetof(BitManipMinMaxRecord, core) == 40);

__global__ void rv64_bitmanip_min_max_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipMinMaxRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        BitManipMinMaxCore(VariableRangeChecker(range_ptr, range_bins))
            .fill_trace_row(row.slice_from(COL_INDEX(BitManipMinMaxCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_bitmanip_min_max_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipMinMaxRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitManipMinMaxCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_bitmanip_min_max_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename T> struct BitManipRegCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    BitManipRegCoreCols<T> core;
};

struct BitManipRegRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    BitManipRegCoreRecord core;
};

static_assert(sizeof(BitManipRegRecord) == 60);
static_assert(offsetof(BitManipRegRecord, core) == 40);

__global__ void rv64_bitmanip_reg_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipRegRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        BitManipRegCore().fill_trace_row(row.slice_from(COL_INDEX(BitManipRegCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_bitmanip_reg_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipRegRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitManipRegCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_bitmanip_reg_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename T> struct BitManipImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    BitManipImmCoreCols<T> core;
};

struct BitManipImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    BitManipImmCoreRecord core;
};

static_assert(sizeof(BitManipImmRecord) == 44);
static_assert(offsetof(BitManipImmRecord, core) == 32);

__global__ void rv64_bitmanip_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipImmRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        BitManipImmCore().fill_trace_row(row.slice_from(COL_INDEX(BitManipImmCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_bitmanip_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BitManipImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitManipImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_bitmanip_imm_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
