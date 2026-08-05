#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/reveal_replay.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace program;
using namespace riscv;

static constexpr uint32_t REVEAL_NUM_SHIFTS = MEMORY_BLOCK_BYTES;
static constexpr size_t REVEAL_SHIFT_SELECTOR_WIDTH = 3;
static constexpr uint32_t REVEAL_SHIFT_SELECTOR_MAX_DEGREE = 2;

__device__ inline Encoder reveal_shift_encoder() {
    return Encoder(
        REVEAL_NUM_SHIFTS,
        REVEAL_SHIFT_SELECTOR_MAX_DEGREE,
        true,
        REVEAL_SHIFT_SELECTOR_WIDTH
    );
}

template <typename T> struct RevealAdapterCols {
    ExecutionState<T> from_state;
    T base_ptr;
    T base_data[PTR_U16_LIMBS];
    MemoryReadAuxCols<T> base_aux_cols;
    T src_ptr;
    MemoryReadAuxCols<T> src_aux_cols;
    T imm;
    T imm_sign;
    T reveal_ptr_low_limb;
    T reveal_ptr_carry;
    MemoryBaseAuxCols<T> write_base_aux[2];
};

struct RevealAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ RevealAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t src_ptr,
        uint32_t base_ptr,
        uint32_t base_value,
        uint32_t base_prev_timestamp,
        uint32_t src_prev_timestamp,
        uint32_t write0_prev_timestamp,
        uint32_t write1_prev_timestamp,
        uint16_t imm,
        uint8_t imm_sign
    ) {
        COL_WRITE_VALUE(row, RevealAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(row, RevealAdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, RevealAdapterCols, base_ptr, base_ptr);

        Fp base_data[PTR_U16_LIMBS];
        ptr_to_u16_limbs(base_data, base_value);
        COL_WRITE_ARRAY(row, RevealAdapterCols, base_data, base_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(RevealAdapterCols, base_aux_cols)),
            base_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(RevealAdapterCols, src_aux_cols)),
            src_prev_timestamp,
            from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(RevealAdapterCols, write_base_aux[0])),
            write0_prev_timestamp,
            from_timestamp + 2
        );
        bool crosses = write1_prev_timestamp != UINT32_MAX;
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(RevealAdapterCols, write_base_aux[1])),
                write1_prev_timestamp,
                from_timestamp + 3
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(RevealAdapterCols, write_base_aux[1]))
            );
        }

        COL_WRITE_VALUE(row, RevealAdapterCols, src_ptr, src_ptr);
        COL_WRITE_VALUE(row, RevealAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, RevealAdapterCols, imm_sign, imm_sign);

        uint32_t ptr = base_value + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, RevealAdapterCols, reveal_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        uint32_t aligned_limb = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        uint32_t block1_low_sum = aligned_limb + uint32_t(MEMORY_BLOCK_BYTES);
        bool carry = crosses && block1_low_sum == (1u << U16_BITS);
        COL_WRITE_VALUE(row, RevealAdapterCols, reveal_ptr_carry, carry);
        if (crosses) {
            range_checker.add_count(
                (block1_low_sum - (uint32_t(carry) << U16_BITS)) >> 3,
                U16_BITS - 3
            );
        }
        if (carry) {
            range_checker.add_count(ptr_limbs[1] + carry, pointer_max_bits - U16_BITS);
        }
    }
};

template <typename T> struct RevealCoreCols {
    T selector[REVEAL_SHIFT_SELECTOR_WIDTH];
    T src_data[BLOCK_FE_WIDTH];
    T prev_data[2][BLOCK_FE_WIDTH];
    T src_lo_bytes[REVEAL_ACCESS_WIDTH / U16_CELL_SIZE];
    T prev_bound_bytes[2];
};

struct RevealCore {
    using Cols = RevealCoreCols<uint8_t>;
    static constexpr size_t NUM_VALUE_CELLS = REVEAL_ACCESS_WIDTH / U16_CELL_SIZE;

    BitwiseOperationLookup bitwise_lookup;

    __device__ RevealCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint16_t const (&src_data)[BLOCK_FE_WIDTH],
        uint16_t const (&prev_data)[2][BLOCK_FE_WIDTH],
        uint8_t shift
    ) {
        Encoder encoder = reveal_shift_encoder();
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, src_data), BLOCK_FE_WIDTH, src_data);
        row.write_array(offsetof(Cols, prev_data), 2 * BLOCK_FE_WIDTH, &prev_data[0][0]);

        uint16_t src_lo_bytes[NUM_VALUE_CELLS] = {};
        uint16_t prev_bound_bytes[2] = {};
        if (shift & 1) {
            for (size_t i = 0; i < NUM_VALUE_CELLS; i++) {
                src_lo_bytes[i] = src_data[i] & BYTE_MASK;
                bitwise_lookup.add_range(src_lo_bytes[i], src_data[i] >> BYTE_BITS);
            }
            for (size_t which = 0; which < 2; which++) {
                size_t cell_index = (shift >> 1) + which * NUM_VALUE_CELLS;
                uint16_t cell = prev_data[cell_index / BLOCK_FE_WIDTH]
                                         [cell_index % BLOCK_FE_WIDTH];
                uint16_t lo = cell & BYTE_MASK;
                uint16_t hi = cell >> BYTE_BITS;
                bitwise_lookup.add_range(lo, hi);
                prev_bound_bytes[which] = which == 0 ? lo : hi;
            }
        }
        row.write_array(offsetof(Cols, src_lo_bytes), NUM_VALUE_CELLS, src_lo_bytes);
        row.write_array(offsetof(Cols, prev_bound_bytes), 2, prev_bound_bytes);
    }
};

template <typename T> struct RevealCols {
    RevealAdapterCols<T> adapter;
    RevealCoreCols<T> core;
};

#include "../rvr/src/reveal.inc.cuh"
