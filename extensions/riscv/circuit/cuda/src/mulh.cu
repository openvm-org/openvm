#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/reg_reg_write_replay.cuh"

using namespace riscv;

template <typename T, size_t NUM_LIMBS> struct MulHCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T a_mul[NUM_LIMBS];
    T b_ext;
    T c_ext;
    T opcode_mulh_flag;
    T opcode_mulhsu_flag;
    T opcode_mulhu_flag;
};

template <size_t NUM_LIMBS> struct MulHCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

// Opcode mapping: MULH=0, MULHSU=1, MULHU=2
enum MulHOpcode { MULH = 0, MULHSU = 1, MULHU = 2 };

template <size_t NUM_LIMBS>
__device__ void run_mulh(
    MulHOpcode opcode,
    const uint32_t *x,
    const uint32_t *y,
    uint32_t *out_mulh,
    uint32_t *out_mul,
    uint32_t *out_carry,
    uint32_t &out_x_ext,
    uint32_t &out_y_ext
) {
#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        out_mul[i] = 0;
        out_carry[i] = 0;
        out_carry[NUM_LIMBS + i] = 0;
    }
#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        if (i > 0) {
            out_mul[i] = out_carry[i - 1];
        }
        for (int j = 0; j <= i; j++) {
            out_mul[i] += x[j] * y[i - j];
        }
        out_carry[i] = out_mul[i] >> RV64_BYTE_BITS;
        out_mul[i] %= (1u << RV64_BYTE_BITS);
    }

    out_x_ext = (x[NUM_LIMBS - 1] >> (RV64_BYTE_BITS - 1)) *
                (opcode == MULHU ? 0 : ((1u << RV64_BYTE_BITS) - 1));
    out_y_ext = (y[NUM_LIMBS - 1] >> (RV64_BYTE_BITS - 1)) *
                (opcode == MULH ? ((1u << RV64_BYTE_BITS) - 1) : 0);

    uint32_t x_prefix = 0;
    uint32_t y_prefix = 0;

#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        x_prefix += x[i];
        y_prefix += y[i];
        out_mulh[i] = out_carry[NUM_LIMBS + i - 1] + x_prefix * out_y_ext + y_prefix * out_x_ext;
#pragma unroll
        for (int j = i + 1; j < NUM_LIMBS; j++) {
            out_mulh[i] += x[j] * y[NUM_LIMBS + i - j];
        }
        out_carry[NUM_LIMBS + i] = out_mulh[i] >> RV64_BYTE_BITS;
        out_mulh[i] %= (1u << RV64_BYTE_BITS);
    }
}

template <size_t NUM_LIMBS> struct MulHCore {
    RangeTupleChecker<2> range_tuple;
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = MulHCoreCols<T, NUM_LIMBS>;

    __device__ MulHCore(
        uint32_t *range_tuple_ptr,
        uint32_t range_tuple_sizes[2],
        BitwiseOperationLookup bw
    )
        : range_tuple(range_tuple_ptr, range_tuple_sizes), bitwise_lookup(bw) {}

    __device__ void fill_trace_row(RowSlice row, MulHCoreRecord<NUM_LIMBS> record) {
        MulHOpcode opcode = static_cast<MulHOpcode>(record.local_opcode);

        uint32_t b[NUM_LIMBS];
        uint32_t c[NUM_LIMBS];
#pragma unroll
        for (int i = 0; i < NUM_LIMBS; i++) {
            b[i] = static_cast<uint32_t>(record.b[i]);
            c[i] = static_cast<uint32_t>(record.c[i]);
        }

        uint32_t a[NUM_LIMBS];
        uint32_t a_mul[NUM_LIMBS];
        uint32_t carry[2 * NUM_LIMBS];
        uint32_t b_ext, c_ext;

        run_mulh<NUM_LIMBS>(opcode, b, c, a, a_mul, carry, b_ext, c_ext);

#pragma unroll
        for (int i = 0; i < NUM_LIMBS; i++) {
            uint32_t aux[2] = {a_mul[i], carry[i]};
            range_tuple.add_count(aux);

            aux[0] = a[i];
            aux[1] = carry[NUM_LIMBS + i];
            range_tuple.add_count(aux);
        }

        if (opcode != MULHU) {
            uint32_t b_sign_mask = (b_ext == 0) ? 0 : (1u << (RV64_BYTE_BITS - 1));
            uint32_t c_sign_mask = (c_ext == 0) ? 0 : (1u << (RV64_BYTE_BITS - 1));

            bitwise_lookup.add_range(
                (b[NUM_LIMBS - 1] - b_sign_mask) << 1,
                (c[NUM_LIMBS - 1] - c_sign_mask) << (opcode == MULH)
            );
        }

#pragma unroll
        for (int i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_range(b[i], c[i]);
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, b);
        COL_WRITE_ARRAY(row, Cols, c, c);
        COL_WRITE_ARRAY(row, Cols, a_mul, a_mul);
        COL_WRITE_VALUE(row, Cols, b_ext, b_ext);
        COL_WRITE_VALUE(row, Cols, c_ext, c_ext);
        COL_WRITE_VALUE(row, Cols, opcode_mulh_flag, opcode == MULH);
        COL_WRITE_VALUE(row, Cols, opcode_mulhsu_flag, opcode == MULHSU);
        COL_WRITE_VALUE(row, Cols, opcode_mulhu_flag, opcode == MULHU);
    }
};

template <typename T> struct MulHCols {
    Rv64MultAdapterCols<T> adapter;
    MulHCoreCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct MulHRecord {
    Rv64MultAdapterRecord adapter;
    MulHCoreRecord<RV64_REGISTER_NUM_LIMBS> core;
};

__global__ void mulh_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<MulHRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        MulHCore<RV64_REGISTER_NUM_LIMBS> core(
            d_range_tuple_checker_ptr,
            (uint32_t[2]){range_tuple_checker_sizes.x, range_tuple_checker_sizes.y},
            BitwiseOperationLookup(d_bitwise_lookup_ptr)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(MulHCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(MulHCols<uint8_t>));
    }
}

extern "C" int _mulh_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<MulHRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(MulHCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);

    mulh_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        d_range_tuple_checker_ptr,
        range_tuple_checker_sizes,
        timestamp_max_bits
    );

    return CHECK_KERNEL();
}

__global__ void mulh_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t mulh_start, size_t mulh_count,
    size_t mulhsu_start, size_t mulhsu_count,
    size_t mulhu_start, size_t mulhu_count,
    uint32_t *error,
    uint32_t mulh_opcode, uint32_t mulhsu_opcode, uint32_t mulhu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker, size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple, uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(MulHCols<uint8_t>));
    size_t total = mulh_count + mulhsu_count + mulhu_count;
    if (idx >= total) return;

    size_t step_index;
    uint32_t expected_opcode;
    MulHOpcode local_opcode;
    if (idx < mulh_count) {
        step_index = mulh_start + idx;
        expected_opcode = mulh_opcode;
        local_opcode = MULH;
    } else if (idx < mulh_count + mulhsu_count) {
        step_index = mulhsu_start + idx - mulh_count;
        expected_opcode = mulhsu_opcode;
        local_opcode = MULHSU;
    } else {
        step_index = mulhu_start + idx - mulh_count - mulhsu_count;
        expected_opcode = mulhu_opcode;
        local_opcode = MULHU;
    }
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 641);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0) {
        preflight_set_error(error, 642);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 643);
        return;
    }
    Rv64RegRegWriteReplay replay;
    if (!replay_reg_reg_write(
            instructions[instruction_index], expected_opcode, register_address_space, from, to,
            step, memory, seeds, predecessors, replay, error, 644
        )) return;

    uint32_t b[RV64_REGISTER_NUM_LIMBS];
    uint32_t c[RV64_REGISTER_NUM_LIMBS];
    uint32_t expected[RV64_REGISTER_NUM_LIMBS];
    uint32_t low[RV64_REGISTER_NUM_LIMBS];
    uint32_t carry[2 * RV64_REGISTER_NUM_LIMBS];
    uint32_t b_ext, c_ext;
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        b[i] = replay.rs1[i];
        c[i] = replay.rs2[i];
    }
    run_mulh<RV64_REGISTER_NUM_LIMBS>(
        local_opcode, b, c, expected, low, carry, b_ext, c_ext
    );
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        if (replay.result[i] != expected[i]) {
            preflight_set_error(error, 649);
            return;
        }
    }
    MulHCoreRecord<RV64_REGISTER_NUM_LIMBS> core_record{};
    core_record.local_opcode = static_cast<uint8_t>(local_opcode);
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        core_record.b[i] = replay.rs1[i];
        core_record.c[i] = replay.rs2[i];
    }
    Rv64MultAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(row, replay_mult_adapter_record(replay));
    MulHCore<RV64_REGISTER_NUM_LIMBS> core(
        range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y},
        BitwiseOperationLookup(bitwise_lookup)
    );
    core.fill_trace_row(row.slice_from(COL_INDEX(MulHCols, core)), core_record);
}

extern "C" int _mulh_replay_tracegen(
    Fp *trace, size_t height, size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions, uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t mulh_start, size_t mulh_count,
    size_t mulhsu_start, size_t mulhsu_count,
    size_t mulhu_start, size_t mulhu_count,
    uint32_t *error,
    uint32_t mulh_opcode, uint32_t mulhsu_opcode, uint32_t mulhu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker, size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple, uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits, cudaStream_t stream
) {
    assert(width == sizeof(MulHCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(mulh_start <= steps.len() && mulh_count <= steps.len() - mulh_start);
    assert(mulhsu_start <= steps.len() && mulhsu_count <= steps.len() - mulhsu_start);
    assert(mulhu_start <= steps.len() && mulhu_count <= steps.len() - mulhu_start);
    assert(mulh_count <= SIZE_MAX - mulhsu_count);
    assert(mulh_count + mulhsu_count <= SIZE_MAX - mulhu_count);
    assert(height >= mulh_count + mulhsu_count + mulhu_count);
    auto [grid, block] = kernel_launch_params(height, 512);
    mulh_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, instructions, pc_base, program_log, memory, seeds, predecessors, steps,
        mulh_start, mulh_count, mulhsu_start, mulhsu_count, mulhu_start, mulhu_count, error,
        mulh_opcode, mulhsu_opcode, mulhu_opcode, register_address_space, range_checker,
        range_checker_bins, bitwise_lookup, range_tuple, range_tuple_sizes, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
