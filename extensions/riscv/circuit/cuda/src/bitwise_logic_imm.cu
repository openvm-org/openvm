#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm.cuh"
#include "riscv/cores/bitwise_logic_imm.cuh"

using namespace riscv;

// XORI/ORI/ANDI use byte limbs and the immediate-only byte ALU adapter.
using Rv64BitwiseLogicImmCoreRecord = BitwiseLogicImmCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64BitwiseLogicImmCore = BitwiseLogicImmCore<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
template <typename T>
using Rv64BitwiseLogicImmCoreCols = BitwiseLogicImmCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicImmCols {
    Rv64BaseAluImmAdapterCols<T> adapter;
    Rv64BitwiseLogicImmCoreCols<T> core;
};

struct Rv64BitwiseLogicImmRecord {
    Rv64BaseAluImmAdapterRecord adapter;
    Rv64BitwiseLogicImmCoreRecord core;
};

static_assert(sizeof(Rv64BitwiseLogicImmRecord) == 44);
static_assert(offsetof(Rv64BitwiseLogicImmRecord, core) == 32);

__global__ void bitwise_logic_imm_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BitwiseLogicImmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluImmAdapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core = Rv64BitwiseLogicImmCore(BitwiseOperationLookup(d_bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicImmCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_imm_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BitwiseLogicImmRecord> d_records,
    uint32_t *d_range_checker,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    bitwise_logic_imm_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
