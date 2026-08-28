#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/vec_heap.cuh"
#include "riscv-adapters/vec_heap_branch_u16.cuh"
#include "riscv-adapters/vec_heap_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "riscv/cores/beq.cuh"
#include "riscv/cores/bitwise_logic.cuh"
#include "riscv/cores/blt.cuh"
#include "riscv/cores/less_than.cuh"
#include "riscv/cores/mul.cuh"
#include "riscv/cores/shift_logical.cuh"
#include "riscv/cores/shift_right_arithmetic.cuh"
#include "arch/rvr/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

constexpr size_t INT256_NUM_U8_LIMBS = 32;
constexpr size_t INT256_NUM_MEMORY_BLOCKS = INT256_NUM_U8_LIMBS / MEMORY_BLOCK_BYTES;
constexpr size_t INT256_NUM_U16_LIMBS = INT256_NUM_U8_LIMBS / sizeof(uint16_t);
constexpr size_t NUM_READS = 2;

using BitwiseLogic256Core = BitwiseLogicCore<INT256_NUM_U8_LIMBS>;
template <typename T>
using BitwiseLogic256CoreCols = BitwiseLogicCoreCols<T, INT256_NUM_U8_LIMBS>;
using BranchEqual256Core = BranchEqualCore<INT256_NUM_U16_LIMBS>;
template <typename T>
using BranchEqual256CoreCols = BranchEqualCoreCols<T, INT256_NUM_U16_LIMBS>;
using BranchEqual256CoreRecord = BranchEqualCoreRecord<INT256_NUM_U16_LIMBS>;
using LessThan256Core = LessThanCore<INT256_NUM_U16_LIMBS, U16_BITS>;
template <typename T>
using LessThan256CoreCols = LessThanCoreCols<T, INT256_NUM_U16_LIMBS, U16_BITS>;
using Multiplication256Core = MultiplicationCore<INT256_NUM_U8_LIMBS>;
using Multiplication256CoreRecord = MultiplicationCoreRecord<INT256_NUM_U8_LIMBS>;
template <typename T>
using Multiplication256CoreCols = MultiplicationCoreCols<T, INT256_NUM_U8_LIMBS>;
using ShiftRightArithmetic256Core =
    ShiftRightArithmeticCore<INT256_NUM_U16_LIMBS, U16_BITS>;
template <typename T>
using ShiftRightArithmetic256CoreCols =
    ShiftRightArithmeticCoreCols<T, INT256_NUM_U16_LIMBS, U16_BITS>;
using ShiftLogical256Core = ShiftLogicalCore<INT256_NUM_U16_LIMBS, U16_BITS>;
template <typename T>
using ShiftLogical256CoreCols = ShiftLogicalCoreCols<T, INT256_NUM_U16_LIMBS, U16_BITS>;
using BranchLessThan256Core = BranchLessThanCore<INT256_NUM_U16_LIMBS, U16_BITS>;
template <typename T>
using BranchLessThan256CoreCols =
    BranchLessThanCoreCols<T, INT256_NUM_U16_LIMBS, U16_BITS>;
using BranchLessThan256CoreRecord =
    BranchLessThanCoreRecord<INT256_NUM_U16_LIMBS, U16_BITS>;
using AddSub256Core = AddSubCore<INT256_NUM_U16_LIMBS, U16_BITS, true>;
template <typename T>
using AddSub256CoreCols = AddSubCoreCols<T, INT256_NUM_U16_LIMBS>;

using VecHeapAdapter256 =
    VecHeapAdapter<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
template <typename T>
using VecHeapAdapter256Cols = VecHeapAdapterCols<
    T,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS>;
using VecHeapAdapter256Record = VecHeapAdapterRecord<
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS>;
using VecHeapBranchU16Adapter256 =
    VecHeapBranchU16Adapter<NUM_READS, INT256_NUM_MEMORY_BLOCKS>;
template <typename T>
using VecHeapBranchU16Adapter256Cols =
    VecHeapBranchU16AdapterCols<T, NUM_READS, INT256_NUM_MEMORY_BLOCKS>;
using VecHeapBranchU16Adapter256Record =
    VecHeapBranchU16AdapterRecord<NUM_READS, INT256_NUM_MEMORY_BLOCKS>;
using VecHeapU16Adapter256 =
    VecHeapU16Adapter<NUM_READS, INT256_NUM_MEMORY_BLOCKS, INT256_NUM_MEMORY_BLOCKS>;
template <typename T>
using VecHeapU16Adapter256Cols = VecHeapU16AdapterCols<
    T,
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS>;
using VecHeapU16Adapter256Record = VecHeapU16AdapterRecord<
    NUM_READS,
    INT256_NUM_MEMORY_BLOCKS,
    INT256_NUM_MEMORY_BLOCKS>;

template <typename T> struct AddSub256Cols {
    VecHeapU16Adapter256Cols<T> adapter;
    AddSub256CoreCols<T> core;
};
template <typename T> struct BitwiseLogic256Cols {
    VecHeapAdapter256Cols<T> adapter;
    BitwiseLogic256CoreCols<T> core;
};
template <typename T> struct LessThan256Cols {
    VecHeapU16Adapter256Cols<T> adapter;
    LessThan256CoreCols<T> core;
};
template <typename T> struct ShiftLogical256Cols {
    VecHeapU16Adapter256Cols<T> adapter;
    ShiftLogical256CoreCols<T> core;
};
template <typename T> struct ShiftRightArithmetic256Cols {
    VecHeapU16Adapter256Cols<T> adapter;
    ShiftRightArithmetic256CoreCols<T> core;
};
template <typename T> struct Multiplication256Cols {
    VecHeapAdapter256Cols<T> adapter;
    Multiplication256CoreCols<T> core;
};
template <typename T> struct BranchEqual256Cols {
    VecHeapBranchU16Adapter256Cols<T> adapter;
    BranchEqual256CoreCols<T> core;
};
template <typename T> struct BranchLessThan256Cols {
    VecHeapBranchU16Adapter256Cols<T> adapter;
    BranchLessThan256CoreCols<T> core;
};

// Checkpoint replay consumes the shared, chronology-resolved memory log
// directly. No Int256 record buffer is materialized on either host or device.

namespace {

static constexpr uint32_t INT256_REPLAY_BAD_STEP = 401;
static constexpr uint32_t INT256_REPLAY_BAD_INSTRUCTION = 402;
static constexpr uint32_t INT256_REPLAY_BAD_EVENT = 403;
static constexpr uint32_t INT256_REPLAY_BAD_PREDECESSOR = 404;
static constexpr uint32_t INT256_REPLAY_BAD_RESULT = 405;
static constexpr uint32_t INT256_REPLAY_BAD_BRANCH = 406;
static constexpr uint32_t INT256_REPLAY_FIELD_ORDER = 2013265921u;

struct Int256ReplayRow {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t to_pc;
    uint32_t rs_ptrs[NUM_READS];
    uint32_t rd_ptr;
    uint32_t rs_vals[NUM_READS];
    uint32_t rd_val;
    uint32_t rs_prev_timestamps[NUM_READS];
    uint32_t rd_prev_timestamp;
    uint32_t read_prev_timestamps[NUM_READS][INT256_NUM_MEMORY_BLOCKS];
    uint32_t write_prev_timestamps[INT256_NUM_MEMORY_BLOCKS];
    uint16_t reads[NUM_READS][INT256_NUM_MEMORY_BLOCKS][BLOCK_FE_WIDTH];
    uint16_t writes[INT256_NUM_MEMORY_BLOCKS][BLOCK_FE_WIDTH];
    uint16_t write_previous[INT256_NUM_MEMORY_BLOCKS][BLOCK_FE_WIDTH];
    uint32_t immediate;
    uint8_t local_opcode;
};

struct Int256ReplayInputs {
    DeviceBufferConstView<RvrReplayInstruction> instructions;
    uint32_t pc_base;
    DeviceBufferConstView<PreflightProgramEvent> program_log;
    DeviceBufferConstView<PreflightMemoryEvent> memory;
    DeviceBufferConstView<PreflightInitialWrite> seeds;
    DeviceBufferConstView<uint32_t> predecessors;
    DeviceBufferConstView<RvrReplayStep> steps;
    size_t step_start;
    size_t num_steps;
    uint32_t opcode_base;
    uint32_t register_address_space;
    uint32_t memory_address_space;
    uint32_t pointer_max_bits;
    uint32_t *error;
};

__device__ __forceinline__ uint64_t int256_block_u64(
    uint16_t const (&limbs)[BLOCK_FE_WIDTH]
) {
    uint64_t value = 0;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) value |= uint64_t(limbs[i]) << (16 * i);
    return value;
}

__host__ __device__ constexpr bool int256_pointer_range(
    uint64_t pointer,
    uint32_t pointer_max_bits
) {
    uint64_t end = pointer + INT256_NUM_U8_LIMBS;
    if ((pointer & 7u) != 0 || end < pointer || pointer_max_bits > 32) return false;
    return end <= (uint64_t(1) << pointer_max_bits);
}

static_assert(int256_pointer_range((uint64_t(1) << 32) - 32, 32));
static_assert(!int256_pointer_range(2, 32));
static_assert(!int256_pointer_range(4, 32));
static_assert(!int256_pointer_range(6, 32));
static_assert(!int256_pointer_range((uint64_t(1) << 32) - 31, 32));
static_assert(!int256_pointer_range((uint64_t(1) << 32) - 30, 32));
static_assert(!int256_pointer_range(0, 33));

__host__ __device__ constexpr bool int256_sequential_pc(uint32_t from_pc, uint32_t to_pc) {
    return from_pc <= UINT32_MAX - program::DEFAULT_PC_STEP &&
           to_pc == from_pc + program::DEFAULT_PC_STEP;
}

static_assert(int256_sequential_pc(0, program::DEFAULT_PC_STEP));
static_assert(!int256_sequential_pc(UINT32_MAX - 3, 0));

__device__ __forceinline__ bool int256_expected_event(
    PreflightMemoryEvent const &event,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t pointer,
    bool is_write
) {
    return event.timestamp == timestamp && preflight_address_space(event) == address_space &&
           event.pointer == pointer && preflight_is_write(event) == is_write;
}

__device__ bool load_int256_replay_row(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t expected_opcode_base,
    uint8_t first_local_opcode,
    uint8_t num_local_opcodes,
    uint32_t register_address_space,
    uint32_t memory_address_space,
    uint32_t pointer_max_bits,
    bool has_write,
    ReplayPcEffect pc_effect,
    Int256ReplayRow &out,
    uint32_t *error
) {
    uint32_t timestamp_delta = has_write ? 15u : 10u;
    ReplayProgramTransition transition;
    ReplayProgramTransitionError transition_error = resolve_replay_program_transition(
        instructions,
        pc_base,
        program_log,
        step.program_index,
        timestamp_delta,
        pc_effect,
        transition
    );
    if (transition_error != ReplayProgramTransitionError::None) {
        uint32_t error_code =
            transition_error == ReplayProgramTransitionError::MissingInstruction
                ? INT256_REPLAY_BAD_INSTRUCTION
                : INT256_REPLAY_BAD_STEP;
        preflight_set_error(error, error_code);
        return false;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t local_opcode = instruction.words[0] - expected_opcode_base;
    if (instruction.words[0] < expected_opcode_base || local_opcode < first_local_opcode ||
        local_opcode - first_local_opcode >= num_local_opcodes ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != memory_address_space || instruction.words[6] != 0 ||
        instruction.words[7] != 0 || (!has_write && instruction.words[3] >= INT256_REPLAY_FIELD_ORDER)) {
        preflight_set_error(error, INT256_REPLAY_BAD_INSTRUCTION);
        return false;
    }

    uint32_t rs_ptrs[NUM_READS] = {
        has_write ? instruction.words[2] : instruction.words[1],
        has_write ? instruction.words[3] : instruction.words[2],
    };
    uint32_t rd_ptr = has_write ? instruction.words[1] : 0;
    if (!replay_canonical_register_pointer(rs_ptrs[0]) ||
        !replay_canonical_register_pointer(rs_ptrs[1]) ||
        (has_write && !replay_canonical_register_pointer(rd_ptr))) {
        preflight_set_error(error, INT256_REPLAY_BAD_INSTRUCTION);
        return false;
    }

    size_t event_count = has_write ? 15 : 10;
    size_t event_start = step.memory_start;
    if (event_start > memory.len() || event_count > memory.len() - event_start ||
        memory.len() != predecessors.len()) {
        preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
        return false;
    }
    size_t next_event = event_start + event_count;
    if (next_event < memory.len() && memory[next_event].timestamp < to.timestamp) {
        preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
        return false;
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.to_pc = to.pc;
    out.local_opcode = static_cast<uint8_t>(local_opcode);
    out.immediate = has_write ? 0 : instruction.words[3];
    out.rd_ptr = rd_ptr;
#pragma unroll
    for (size_t i = 0; i < NUM_READS; i++) out.rs_ptrs[i] = rs_ptrs[i];

    size_t cursor = event_start;
#pragma unroll
    for (size_t i = 0; i < NUM_READS; i++, cursor++) {
        auto const &event = memory[cursor];
        if (!int256_expected_event(
                event, from.timestamp + i, register_address_space, rs_ptrs[i] / 2, false
            )) {
            preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
            return false;
        }
        ReplayPreviousValue previous;
        if (!replay_previous_value(cursor, event, predecessors[cursor], memory, seeds, previous)) {
            preflight_set_error(error, INT256_REPLAY_BAD_PREDECESSOR);
            return false;
        }
        uint64_t value = int256_block_u64(event.value);
        if ((value >> 32) != 0 || !int256_pointer_range(value, pointer_max_bits)) {
            preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
            return false;
        }
        out.rs_vals[i] = static_cast<uint32_t>(value);
        out.rs_prev_timestamps[i] = previous.timestamp;
    }
    if (has_write) {
        auto const &event = memory[cursor];
        if (!int256_expected_event(
                event, from.timestamp + 2, register_address_space, rd_ptr / 2, false
            )) {
            preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
            return false;
        }
        ReplayPreviousValue previous;
        if (!replay_previous_value(cursor, event, predecessors[cursor], memory, seeds, previous)) {
            preflight_set_error(error, INT256_REPLAY_BAD_PREDECESSOR);
            return false;
        }
        uint64_t value = int256_block_u64(event.value);
        if ((value >> 32) != 0 || !int256_pointer_range(value, pointer_max_bits)) {
            preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
            return false;
        }
        out.rd_val = static_cast<uint32_t>(value);
        out.rd_prev_timestamp = previous.timestamp;
        cursor++;
    }

#pragma unroll
    for (size_t read = 0; read < NUM_READS; read++) {
#pragma unroll
        for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++, cursor++) {
            auto const &event = memory[cursor];
            uint32_t timestamp = from.timestamp + (has_write ? 3u : 2u) +
                                 read * INT256_NUM_MEMORY_BLOCKS + block;
            uint32_t pointer = (out.rs_vals[read] + block * MEMORY_BLOCK_BYTES) / 2;
            if (!int256_expected_event(
                    event, timestamp, memory_address_space, pointer, false
                )) {
                preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
                return false;
            }
            ReplayPreviousValue previous;
            if (!replay_previous_value(
                    cursor, event, predecessors[cursor], memory, seeds, previous
                )) {
                preflight_set_error(error, INT256_REPLAY_BAD_PREDECESSOR);
                return false;
            }
            out.read_prev_timestamps[read][block] = previous.timestamp;
#pragma unroll
            for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
                out.reads[read][block][limb] = event.value[limb];
            }
        }
    }
    if (has_write) {
#pragma unroll
        for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++, cursor++) {
            auto const &event = memory[cursor];
            uint32_t timestamp = from.timestamp + 11u + block;
            uint32_t pointer = (out.rd_val + block * MEMORY_BLOCK_BYTES) / 2;
            if (!int256_expected_event(event, timestamp, memory_address_space, pointer, true)) {
                preflight_set_error(error, INT256_REPLAY_BAD_EVENT);
                return false;
            }
            ReplayPreviousValue previous;
            if (!replay_previous_value(
                    cursor, event, predecessors[cursor], memory, seeds, previous
                )) {
                preflight_set_error(error, INT256_REPLAY_BAD_PREDECESSOR);
                return false;
            }
            out.write_prev_timestamps[block] = previous.timestamp;
#pragma unroll
            for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
                out.writes[block][limb] = event.value[limb];
                out.write_previous[block][limb] = previous.value[limb];
            }
        }
    }
    return true;
}

__device__ __forceinline__ void int256_flatten_u16_reads(
    Int256ReplayRow const &replay,
    uint16_t (&b)[INT256_NUM_U16_LIMBS],
    uint16_t (&c)[INT256_NUM_U16_LIMBS]
) {
#pragma unroll
    for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
#pragma unroll
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            b[block * BLOCK_FE_WIDTH + limb] = replay.reads[0][block][limb];
            c[block * BLOCK_FE_WIDTH + limb] = replay.reads[1][block][limb];
        }
    }
}

__device__ __forceinline__ void int256_flatten_u8_reads(
    Int256ReplayRow const &replay,
    uint8_t (&b)[INT256_NUM_U8_LIMBS],
    uint8_t (&c)[INT256_NUM_U8_LIMBS]
) {
#pragma unroll
    for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
#pragma unroll
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            size_t byte = block * MEMORY_BLOCK_BYTES + 2 * limb;
            b[byte] = static_cast<uint8_t>(replay.reads[0][block][limb]);
            b[byte + 1] = static_cast<uint8_t>(replay.reads[0][block][limb] >> 8);
            c[byte] = static_cast<uint8_t>(replay.reads[1][block][limb]);
            c[byte + 1] = static_cast<uint8_t>(replay.reads[1][block][limb] >> 8);
        }
    }
}

template <typename Record>
__device__ void int256_fill_u16_adapter_record(
    Int256ReplayRow const &replay,
    Record &record
) {
    record.from_pc = replay.from_pc;
    record.from_timestamp = replay.from_timestamp;
    record.rd_ptr = replay.rd_ptr;
    record.rd_val = replay.rd_val;
    record.rd_read_aux.prev_timestamp = replay.rd_prev_timestamp;
#pragma unroll
    for (size_t read = 0; read < NUM_READS; read++) {
        record.rs_ptrs[read] = replay.rs_ptrs[read];
        record.rs_vals[read] = replay.rs_vals[read];
        record.rs_read_aux[read].prev_timestamp = replay.rs_prev_timestamps[read];
#pragma unroll
        for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
            record.reads_aux[read][block].prev_timestamp =
                replay.read_prev_timestamps[read][block];
        }
    }
#pragma unroll
    for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
        record.writes_aux[block].prev_timestamp = replay.write_prev_timestamps[block];
#pragma unroll
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            record.writes_aux[block].prev_data[limb] = replay.write_previous[block][limb];
        }
    }
}

template <typename Record>
__device__ void int256_fill_u8_adapter_record(
    Int256ReplayRow const &replay,
    Record &record
) {
    record.from_pc = replay.from_pc;
    record.from_timestamp = replay.from_timestamp;
    record.rd_ptr = replay.rd_ptr;
    record.rd_val = replay.rd_val;
    record.rd_read_aux.prev_timestamp = replay.rd_prev_timestamp;
#pragma unroll
    for (size_t read = 0; read < NUM_READS; read++) {
        record.rs_ptrs[read] = replay.rs_ptrs[read];
        record.rs_vals[read] = replay.rs_vals[read];
        record.rs_read_aux[read].prev_timestamp = replay.rs_prev_timestamps[read];
#pragma unroll
        for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
            record.reads_aux[read][block].prev_timestamp =
                replay.read_prev_timestamps[read][block];
        }
    }
#pragma unroll
    for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
        record.writes_aux[block].prev_timestamp = replay.write_prev_timestamps[block];
#pragma unroll
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            record.writes_aux[block].prev_data[2 * limb] =
                static_cast<uint8_t>(replay.write_previous[block][limb]);
            record.writes_aux[block].prev_data[2 * limb + 1] =
                static_cast<uint8_t>(replay.write_previous[block][limb] >> 8);
        }
    }
}

template <typename Record>
__device__ void int256_fill_branch_adapter_record(
    Int256ReplayRow const &replay,
    Record &record
) {
    record.from_pc = replay.from_pc;
    record.from_timestamp = replay.from_timestamp;
#pragma unroll
    for (size_t read = 0; read < NUM_READS; read++) {
        record.rs_ptr[read] = replay.rs_ptrs[read];
        record.rs_vals[read] = replay.rs_vals[read];
        record.rs_read_aux[read].prev_timestamp = replay.rs_prev_timestamps[read];
#pragma unroll
        for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
            record.reads_aux[read][block].prev_timestamp =
                replay.read_prev_timestamps[read][block];
        }
    }
}

__device__ bool int256_u16_write_matches(
    Int256ReplayRow const &replay,
    uint16_t const (&expected)[INT256_NUM_U16_LIMBS]
) {
#pragma unroll
    for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
#pragma unroll
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            if (replay.writes[block][limb] != expected[block * BLOCK_FE_WIDTH + limb]) {
                return false;
            }
        }
    }
    return true;
}

__device__ bool int256_u8_write_matches(
    Int256ReplayRow const &replay,
    uint8_t const (&expected)[INT256_NUM_U8_LIMBS]
) {
#pragma unroll
    for (size_t block = 0; block < INT256_NUM_MEMORY_BLOCKS; block++) {
#pragma unroll
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            size_t byte = block * MEMORY_BLOCK_BYTES + 2 * limb;
            uint16_t packed = uint16_t(expected[byte]) | (uint16_t(expected[byte + 1]) << 8);
            if (replay.writes[block][limb] != packed) return false;
        }
    }
    return true;
}

// Taken targets must stay inside the implemented PC address space on an aligned slot
// (mirrors the CPU trace filler's `checked_branch_target`).
__device__ __forceinline__ bool int256_valid_branch_target(
    uint32_t pc, uint32_t immediate, uint32_t to_pc
) {
    return replay_branch_target_in_bounds(pc, immediate) &&
           to_pc == replay_taken_branch_pc(pc, immediate);
}

__global__ void add_sub256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t
);
__global__ void bitwise_logic256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t *, uint32_t
);
__global__ void less_than256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t
);
__global__ void shift_logical256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t
);
__global__ void shift_right_arithmetic256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t
);
__global__ void branch_equal256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t
);
__global__ void branch_less_than256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t
);
__global__ void multiplication256_replay_tracegen(
    Fp *, size_t, Int256ReplayInputs, uint32_t *, size_t, uint32_t *, uint32_t *, uint2,
    uint32_t
);

} // namespace

static Int256ReplayInputs int256_replay_inputs(
    DeviceBufferConstView<RvrReplayInstruction>,
    uint32_t,
    DeviceBufferConstView<PreflightProgramEvent>,
    DeviceBufferConstView<PreflightMemoryEvent>,
    DeviceBufferConstView<PreflightInitialWrite>,
    DeviceBufferConstView<uint32_t>,
    DeviceBufferConstView<RvrReplayStep>,
    size_t,
    size_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t *
);

extern "C" int _int256_u16_replay_tracegen(
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
    uint32_t opcode_base,
    uint32_t register_address_space,
    uint32_t memory_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t kind,
    cudaStream_t stream
) {
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len() && num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto inputs = int256_replay_inputs(
        instructions, pc_base, program_log, memory, seeds, predecessors, steps, step_start,
        num_steps, opcode_base, register_address_space, memory_address_space, pointer_max_bits,
        error
    );
    auto [grid, block] = kernel_launch_params(height, 256);
    switch (kind) {
    case 0:
        assert(width == sizeof(LessThan256Cols<uint8_t>));
        less_than256_replay_tracegen<<<grid, block, 0, stream>>>(
            trace, height, inputs, range_checker, range_checker_bins, timestamp_max_bits
        );
        break;
    case 1:
        assert(width == sizeof(ShiftLogical256Cols<uint8_t>));
        shift_logical256_replay_tracegen<<<grid, block, 0, stream>>>(
            trace, height, inputs, range_checker, range_checker_bins, timestamp_max_bits
        );
        break;
    case 2:
        assert(width == sizeof(ShiftRightArithmetic256Cols<uint8_t>));
        shift_right_arithmetic256_replay_tracegen<<<grid, block, 0, stream>>>(
            trace, height, inputs, range_checker, range_checker_bins, timestamp_max_bits
        );
        break;
    case 3:
        assert(width == sizeof(BranchEqual256Cols<uint8_t>));
        branch_equal256_replay_tracegen<<<grid, block, 0, stream>>>(
            trace, height, inputs, range_checker, range_checker_bins, timestamp_max_bits
        );
        break;
    case 4:
        assert(width == sizeof(BranchLessThan256Cols<uint8_t>));
        branch_less_than256_replay_tracegen<<<grid, block, 0, stream>>>(
            trace, height, inputs, range_checker, range_checker_bins, timestamp_max_bits
        );
        break;
    default: return 1;
    }
    return CHECK_KERNEL();
}

extern "C" int _multiplication256_replay_tracegen(
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
    uint32_t opcode_base,
    uint32_t register_address_space,
    uint32_t memory_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple,
    uint2 range_tuple_sizes,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Multiplication256Cols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len() && num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto inputs = int256_replay_inputs(
        instructions, pc_base, program_log, memory, seeds, predecessors, steps, step_start,
        num_steps, opcode_base, register_address_space, memory_address_space, pointer_max_bits,
        error
    );
    auto [grid, block] = kernel_launch_params(height, 256);
    multiplication256_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, inputs, range_checker, range_checker_bins, bitwise_lookup, range_tuple,
        range_tuple_sizes, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

static Int256ReplayInputs int256_replay_inputs(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t opcode_base,
    uint32_t register_address_space,
    uint32_t memory_address_space,
    uint32_t pointer_max_bits,
    uint32_t *error
) {
    return Int256ReplayInputs{
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        opcode_base,
        register_address_space,
        memory_address_space,
        pointer_max_bits,
        error,
    };
}

extern "C" int _add_sub256_replay_tracegen(
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
    uint32_t opcode_base,
    uint32_t register_address_space,
    uint32_t memory_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(AddSub256Cols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len() && num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto inputs = int256_replay_inputs(
        instructions, pc_base, program_log, memory, seeds, predecessors, steps, step_start,
        num_steps, opcode_base, register_address_space, memory_address_space, pointer_max_bits,
        error
    );
    auto [grid, block] = kernel_launch_params(height, 256);
    add_sub256_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, inputs, range_checker, range_checker_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _bitwise_logic256_replay_tracegen(
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
    uint32_t opcode_base,
    uint32_t register_address_space,
    uint32_t memory_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BitwiseLogic256Cols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len() && num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto inputs = int256_replay_inputs(
        instructions, pc_base, program_log, memory, seeds, predecessors, steps, step_start,
        num_steps, opcode_base, register_address_space, memory_address_space, pointer_max_bits,
        error
    );
    auto [grid, block] = kernel_launch_params(height, 256);
    bitwise_logic256_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, inputs, range_checker, range_checker_bins, bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

namespace {

__device__ bool int256_replay_load_alu(
    Int256ReplayInputs const &inputs,
    size_t row_index,
    uint8_t first_local_opcode,
    uint8_t num_local_opcodes,
    Int256ReplayRow &replay
) {
    size_t step_index = inputs.step_start + row_index;
    if (step_index >= inputs.steps.len()) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_STEP);
        return false;
    }
    return load_int256_replay_row(
        inputs.instructions,
        inputs.pc_base,
        inputs.program_log,
        inputs.memory,
        inputs.seeds,
        inputs.predecessors,
        inputs.steps[step_index],
        inputs.opcode_base,
        first_local_opcode,
        num_local_opcodes,
        inputs.register_address_space,
        inputs.memory_address_space,
        inputs.pointer_max_bits,
        true,
        ReplayPcEffect::Sequential,
        replay,
        inputs.error
    );
}

__device__ bool int256_replay_load_branch(
    Int256ReplayInputs const &inputs,
    size_t row_index,
    uint8_t num_local_opcodes,
    Int256ReplayRow &replay
) {
    size_t step_index = inputs.step_start + row_index;
    if (step_index >= inputs.steps.len()) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_STEP);
        return false;
    }
    return load_int256_replay_row(
        inputs.instructions,
        inputs.pc_base,
        inputs.program_log,
        inputs.memory,
        inputs.seeds,
        inputs.predecessors,
        inputs.steps[step_index],
        inputs.opcode_base,
        0,
        num_local_opcodes,
        inputs.register_address_space,
        inputs.memory_address_space,
        inputs.pointer_max_bits,
        false,
        ReplayPcEffect::Dynamic,
        replay,
        inputs.error
    );
}

__global__ void add_sub256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(AddSub256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_alu(inputs, idx, 0, 2, replay)) return;
    uint16_t b[INT256_NUM_U16_LIMBS], c[INT256_NUM_U16_LIMBS];
    uint16_t expected[INT256_NUM_U16_LIMBS];
    uint32_t carry[INT256_NUM_U16_LIMBS];
    int256_flatten_u16_reads(replay, b, c);
    if (replay.local_opcode == 0) {
        run_add<INT256_NUM_U16_LIMBS, U16_BITS>(b, c, expected, carry);
    } else {
        run_sub<INT256_NUM_U16_LIMBS, U16_BITS>(b, c, expected, carry);
    }
    if (!int256_u16_write_matches(replay, expected)) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_RESULT);
        return;
    }
    VecHeapU16Adapter256Record adapter_record{};
    int256_fill_u16_adapter_record(replay, adapter_record);
    VecHeapU16Adapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    AddSub256Core core(VariableRangeChecker(range_checker, range_checker_bins));
    core.fill_trace_row(row.slice_from(COL_INDEX(AddSub256Cols, core)), b, c, replay.local_opcode);
}

__global__ void bitwise_logic256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(BitwiseLogic256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_alu(inputs, idx, 2, 3, replay)) return;
    uint8_t b[INT256_NUM_U8_LIMBS], c[INT256_NUM_U8_LIMBS];
    uint8_t expected[INT256_NUM_U8_LIMBS];
    int256_flatten_u8_reads(replay, b, c);
    if (replay.local_opcode == 2) {
        run_xor<INT256_NUM_U8_LIMBS>(b, c, expected);
    } else if (replay.local_opcode == 3) {
        run_or<INT256_NUM_U8_LIMBS>(b, c, expected);
    } else {
        run_and<INT256_NUM_U8_LIMBS>(b, c, expected);
    }
    if (!int256_u8_write_matches(replay, expected)) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_RESULT);
        return;
    }
    VecHeapAdapter256Record adapter_record{};
    int256_fill_u8_adapter_record(replay, adapter_record);
    VecHeapAdapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    BitwiseLogic256Core core{BitwiseOperationLookup(bitwise_lookup)};
    core.fill_trace_row(
        row.slice_from(COL_INDEX(BitwiseLogic256Cols, core)), b, c, replay.local_opcode
    );
}

__global__ void less_than256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(LessThan256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_alu(inputs, idx, 0, 2, replay)) return;
    uint16_t b[INT256_NUM_U16_LIMBS], c[INT256_NUM_U16_LIMBS];
    uint16_t expected[INT256_NUM_U16_LIMBS] = {0};
    int256_flatten_u16_reads(replay, b, c);
    expected[0] = run_less_than<INT256_NUM_U16_LIMBS, U16_BITS>(
                      replay.local_opcode == 0, b, c
    ).cmp_result;
    if (!int256_u16_write_matches(replay, expected)) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_RESULT);
        return;
    }
    VecHeapU16Adapter256Record adapter_record{};
    int256_fill_u16_adapter_record(replay, adapter_record);
    VecHeapU16Adapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    LessThan256Core core{VariableRangeChecker(range_checker, range_checker_bins)};
    core.fill_trace_row(
        row.slice_from(COL_INDEX(LessThan256Cols, core)), b, c, replay.local_opcode
    );
}

__global__ void shift_logical256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(ShiftLogical256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_alu(inputs, idx, 0, 2, replay)) return;
    uint16_t b[INT256_NUM_U16_LIMBS], c[INT256_NUM_U16_LIMBS];
    uint16_t expected[INT256_NUM_U16_LIMBS];
    size_t limb_shift, bit_shift;
    int256_flatten_u16_reads(replay, b, c);
    if (replay.local_opcode == 0) {
        run_shift_left<INT256_NUM_U16_LIMBS, U16_BITS>(
            b, c, expected, limb_shift, bit_shift
        );
    } else {
        run_shift_right_logical<INT256_NUM_U16_LIMBS, U16_BITS>(
            b, c, expected, limb_shift, bit_shift
        );
    }
    if (!int256_u16_write_matches(replay, expected)) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_RESULT);
        return;
    }
    VecHeapU16Adapter256Record adapter_record{};
    int256_fill_u16_adapter_record(replay, adapter_record);
    VecHeapU16Adapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    ShiftLogical256Core core(VariableRangeChecker(range_checker, range_checker_bins));
    core.fill_trace_row(
        row.slice_from(COL_INDEX(ShiftLogical256Cols, core)), b, c, replay.local_opcode
    );
}

__global__ void shift_right_arithmetic256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(ShiftRightArithmetic256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_alu(inputs, idx, 2, 1, replay)) return;
    uint16_t b[INT256_NUM_U16_LIMBS], c[INT256_NUM_U16_LIMBS];
    uint16_t expected[INT256_NUM_U16_LIMBS];
    size_t limb_shift, bit_shift;
    int256_flatten_u16_reads(replay, b, c);
    run_shift_right_arithmetic<INT256_NUM_U16_LIMBS, U16_BITS>(
        b, c, expected, limb_shift, bit_shift
    );
    if (!int256_u16_write_matches(replay, expected)) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_RESULT);
        return;
    }
    VecHeapU16Adapter256Record adapter_record{};
    int256_fill_u16_adapter_record(replay, adapter_record);
    VecHeapU16Adapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    ShiftRightArithmetic256Core core(VariableRangeChecker(range_checker, range_checker_bins));
    core.fill_trace_row(row.slice_from(COL_INDEX(ShiftRightArithmetic256Cols, core)), b, c);
}

__global__ void multiplication256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
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
    row.fill_zero(0, sizeof(Multiplication256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_alu(inputs, idx, 0, 1, replay)) return;
    uint8_t b[INT256_NUM_U8_LIMBS], c[INT256_NUM_U8_LIMBS];
    uint8_t expected[INT256_NUM_U8_LIMBS];
    uint32_t carry[INT256_NUM_U8_LIMBS];
    int256_flatten_u8_reads(replay, b, c);
    run_mul<INT256_NUM_U8_LIMBS>(b, c, expected, carry);
    if (!int256_u8_write_matches(replay, expected)) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_RESULT);
        return;
    }
    VecHeapAdapter256Record adapter_record{};
    int256_fill_u8_adapter_record(replay, adapter_record);
    VecHeapAdapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    RangeTupleChecker<2> range_tuple_checker(
        range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
    );
    Multiplication256Core core(
        range_tuple_checker, BitwiseOperationLookup(bitwise_lookup)
    );
    Multiplication256CoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < INT256_NUM_U8_LIMBS; i++) {
        core_record.b[i] = b[i];
        core_record.c[i] = c[i];
    }
    core.fill_trace_row(row.slice_from(COL_INDEX(Multiplication256Cols, core)), core_record);
}

__global__ void branch_equal256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(BranchEqual256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_branch(inputs, idx, 2, replay)) return;
    uint16_t a[INT256_NUM_U16_LIMBS], b[INT256_NUM_U16_LIMBS];
    int256_flatten_u16_reads(replay, a, b);
    bool equal = true;
#pragma unroll
    for (size_t i = 0; i < INT256_NUM_U16_LIMBS; i++) equal &= a[i] == b[i];
    bool take = replay.local_opcode == 0 ? equal : !equal;
    bool valid_pc =
        take ? int256_valid_branch_target(replay.from_pc, replay.immediate, replay.to_pc)
             : int256_sequential_pc(replay.from_pc, replay.to_pc);
    if (!valid_pc) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_BRANCH);
        return;
    }
    VecHeapBranchU16Adapter256Record adapter_record{};
    int256_fill_branch_adapter_record(replay, adapter_record);
    VecHeapBranchU16Adapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    BranchEqual256CoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < INT256_NUM_U16_LIMBS; i++) {
        core_record.a[i] = a[i];
        core_record.b[i] = b[i];
    }
    core_record.imm = replay.immediate;
    core_record.local_opcode = replay.local_opcode;
    BranchEqual256Core core;
    core.fill_trace_row(row.slice_from(COL_INDEX(BranchEqual256Cols, core)), core_record);
}

__device__ bool int256_branch_less_than(
    uint16_t const (&a)[INT256_NUM_U16_LIMBS],
    uint16_t const (&b)[INT256_NUM_U16_LIMBS],
    uint8_t local_opcode
) {
    bool signed_op = local_opcode == 0 || local_opcode == 2;
    bool ge_op = local_opcode >= 2;
    bool less = false;
#pragma unroll
    for (int i = INT256_NUM_U16_LIMBS - 1; i >= 0; i--) {
        if (a[i] == b[i]) continue;
        if (signed_op && i == INT256_NUM_U16_LIMBS - 1) {
            bool a_sign = (a[i] >> (U16_BITS - 1)) != 0;
            bool b_sign = (b[i] >> (U16_BITS - 1)) != 0;
            less = a_sign != b_sign ? a_sign : a[i] < b[i];
        } else {
            less = a[i] < b[i];
        }
        break;
    }
    return ge_op ? !less : less;
}

__global__ void branch_less_than256_replay_tracegen(
    Fp *trace,
    size_t height,
    Int256ReplayInputs inputs,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(BranchLessThan256Cols<uint8_t>));
    if (idx >= inputs.num_steps) return;
    Int256ReplayRow replay{};
    if (!int256_replay_load_branch(inputs, idx, 4, replay)) return;
    uint16_t a[INT256_NUM_U16_LIMBS], b[INT256_NUM_U16_LIMBS];
    int256_flatten_u16_reads(replay, a, b);
    bool take = int256_branch_less_than(a, b, replay.local_opcode);
    bool valid_pc =
        take ? int256_valid_branch_target(replay.from_pc, replay.immediate, replay.to_pc)
             : int256_sequential_pc(replay.from_pc, replay.to_pc);
    if (!valid_pc) {
        preflight_set_error(inputs.error, INT256_REPLAY_BAD_BRANCH);
        return;
    }
    VecHeapBranchU16Adapter256Record adapter_record{};
    int256_fill_branch_adapter_record(replay, adapter_record);
    VecHeapBranchU16Adapter256 adapter(
        inputs.pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_bins),
        timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    BranchLessThan256CoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < INT256_NUM_U16_LIMBS; i++) {
        core_record.a[i] = a[i];
        core_record.b[i] = b[i];
    }
    core_record.imm = replay.immediate;
    core_record.local_opcode = replay.local_opcode;
    BranchLessThan256Core core{VariableRangeChecker(range_checker, range_checker_bins)};
    core.fill_trace_row(
        row.slice_from(COL_INDEX(BranchLessThan256Cols, core)), core_record
    );
}

} // namespace
