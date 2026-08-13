#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/reveal_replay.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace program;
using namespace riscv;

template <typename T> struct RevealCols {
    T is_valid;
    ExecutionState<T> from_state;
    T base_ptr;
    T base_ptr_limbs[PTR_U16_LIMBS];
    MemoryReadAuxCols<T> base_aux;
    T src_ptr;
    T src_bytes[REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> src_aux;
    T imm;
    T imm_sign;
    T dst_ptr_low_limb;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux[2];
};

struct Reveal {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    __device__ Reveal(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          bitwise_lookup(bitwise_lookup), mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, ReplayRevealInput const &input) {
        COL_WRITE_VALUE(row, RevealCols, is_valid, 1);
        COL_WRITE_PC(row, RevealCols, from_state.pc, input.from_pc);
        COL_WRITE_VALUE(row, RevealCols, from_state.timestamp, input.from_timestamp);
        COL_WRITE_VALUE(row, RevealCols, base_ptr, input.base_ptr);

        uint32_t base_ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(base_ptr_limbs, input.base_value);
        COL_WRITE_ARRAY(row, RevealCols, base_ptr_limbs, base_ptr_limbs);
        mem_helper.fill(
            row.slice_from(COL_INDEX(RevealCols, base_aux)),
            input.base_prev_timestamp,
            input.from_timestamp
        );

        COL_WRITE_VALUE(row, RevealCols, src_ptr, input.src_ptr);
        COL_WRITE_ARRAY(row, RevealCols, src_bytes, input.src_bytes);
        mem_helper.fill(
            row.slice_from(COL_INDEX(RevealCols, src_aux)),
            input.src_prev_timestamp,
            input.from_timestamp + 1
        );
        COL_WRITE_VALUE(row, RevealCols, imm, input.imm);
        COL_WRITE_VALUE(row, RevealCols, imm_sign, input.imm_sign);

        uint32_t dst_ptr = input.base_value + uint32_t(input.imm) +
                           uint32_t(input.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t dst_ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(dst_ptr_limbs, dst_ptr);
        COL_WRITE_VALUE(row, RevealCols, dst_ptr_low_limb, dst_ptr_limbs[0]);
        range_checker.add_count(dst_ptr_limbs[0] >> 3, U16_BITS - 3);
        range_checker.add_count(dst_ptr_limbs[1], pointer_max_bits - U16_BITS);
        for (size_t i = 0; i < REGISTER_NUM_LIMBS; i += 2) {
            bitwise_lookup.add_range(input.src_bytes[i], input.src_bytes[i + 1]);
        }

        for (size_t block = 0; block < 2; block++) {
            COL_WRITE_ARRAY(
                row,
                RevealCols,
                write_aux[block].prev_data,
                input.write_prev_data[block]
            );
            mem_helper.fill(
                row.slice_from(COL_INDEX(RevealCols, write_aux[block])),
                input.write_prev_timestamp[block],
                input.from_timestamp + 2 + block
            );
        }
    }
};

#include "../rvr/src/reveal.inc.cuh"
