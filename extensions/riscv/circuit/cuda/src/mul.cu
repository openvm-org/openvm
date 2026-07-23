#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/mul.cuh"
#include "riscv/reg_reg_write_replay.cuh"

using namespace riscv;

// Concrete type aliases for 64-bit
using Rv64MultiplicationCoreRecord = MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64MultiplicationCore = MultiplicationCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv64MultiplicationCoreCols = MultiplicationCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64MultiplicationCols {
    Rv64MultAdapterCols<T> adapter;
    Rv64MultiplicationCoreCols<T> core;
};

struct Rv64MultiplicationRecord {
    Rv64MultAdapterRecord adapter;
    Rv64MultiplicationCoreRecord core;
};

__global__ void mul_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
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

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr);
        Rv64MultiplicationCore core(range_tuple_checker, bitwise_lookup);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MultiplicationCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MultiplicationCols<uint8_t>));
    }
}

extern "C" int _mul_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64MultiplicationCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    mul_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void mul_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64MultiplicationCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 601);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0) {
        preflight_set_error(error, 602);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 603);
        return;
    }

    Rv64RegRegWriteReplay replay;
    if (!replay_reg_reg_write(
            instructions[instruction_index],
            opcode,
            register_address_space,
            from,
            to,
            step,
            memory,
            seeds,
            predecessors,
            replay,
            error,
            604
        )) {
        return;
    }
    uint8_t expected[RV64_REGISTER_NUM_LIMBS];
    uint32_t carry[RV64_REGISTER_NUM_LIMBS];
    run_mul<RV64_REGISTER_NUM_LIMBS>(replay.rs1, replay.rs2, expected, carry);
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        if (replay.result[i] != expected[i]) {
            preflight_set_error(error, 609);
            return;
        }
    }

    Rv64MultiplicationCoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        core_record.b[i] = replay.rs1[i];
        core_record.c[i] = replay.rs2[i];
    }
    Rv64MultAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(row, replay_mult_adapter_record(replay));
    Rv64MultiplicationCore core(
        RangeTupleChecker<2>(
            range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        ),
        BitwiseOperationLookup(bitwise_lookup)
    );
    core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MultiplicationCols, core)), core_record);
}

extern "C" int _mul_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64MultiplicationCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len());
    assert(num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    mul_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        error,
        opcode,
        register_address_space,
        range_checker,
        range_checker_bins,
        bitwise_lookup,
        range_tuple,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
