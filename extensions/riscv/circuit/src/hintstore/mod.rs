use std::borrow::Borrow;

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS};
use openvm_riscv_transpiler::{
    HintStoreOpcode::{HINT_BUFFER, HINT_STORED},
    MAX_HINT_BUFFER_DWORDS, MAX_HINT_BUFFER_DWORDS_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::adapters::{
    byte_ptr_to_u16_ptr, expand_to_block, ptr_bound_from_high_u16_expr, u16_limbs_to_ptr, PTR_BITS,
    PTR_U16_LIMBS, U16_BITS,
};

mod execution;

pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

// `rem_words` is bounded by `2^MAX_HINT_BUFFER_DWORDS_BITS` (= 2^10), so one u16
// column carries all information. The upper register cells are hardcoded to zero
// in the memory bus interaction.
const _: () = assert!(
    MAX_HINT_BUFFER_DWORDS_BITS <= U16_BITS,
    "MAX_HINT_BUFFER_DWORDS_BITS must fit in one u16 cell"
);
// Scale factor for rem_words range checks.
const REM_WORDS_SHIFT: usize = U16_BITS - MAX_HINT_BUFFER_DWORDS_BITS;

#[inline]
fn validate_hint_buffer_num_words(pc: u32, num_words: u64) -> Result<u16, ExecutionError> {
    if num_words.wrapping_sub(1) >= MAX_HINT_BUFFER_DWORDS as u64 {
        return Err(if num_words == 0 {
            ExecutionError::HintBufferZeroWords { pc }
        } else {
            ExecutionError::HintBufferTooLarge {
                pc,
                num_words,
                max_hint_buffer_words: MAX_HINT_BUFFER_DWORDS as u64,
            }
        });
    }
    Ok(num_words as u16)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct HintStoreCols<T> {
    // common
    pub is_single: T,
    pub is_buffer: T,
    /// Low u16 cell of the 8-byte RV64 register that holds `rem_words`.
    /// `rem_words` is bounded by `2^MAX_HINT_BUFFER_DWORDS_BITS` (= 2^10), so the
    /// upper register cells are known to be zero and are not materialized as columns.
    pub rem_words: T,

    pub from_state: ExecutionState<T>,
    pub mem_ptr_ptr: T,
    /// Low 32 bits of the 8-byte RV64 register that holds `mem_ptr`. `mem_ptr` is a
    /// u32 memory address, so the upper 4 bytes are known to be zero and are hardcoded
    /// in the memory bus interaction rather than materialized as columns.
    pub mem_ptr_limbs: [T; PTR_U16_LIMBS],
    pub mem_ptr_aux_cols: MemoryReadAuxCols<T>,

    pub write_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
    /// One hint word packed as u16 cells.
    pub data: [T; BLOCK_FE_WIDTH],

    // only buffer
    pub is_buffer_start: T,
    pub num_words_ptr: T,
    pub num_words_aux_cols: MemoryReadAuxCols<T>,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(HintStoreCols<u8>)]
pub struct HintStoreAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for HintStoreAir {
    fn width(&self) -> usize {
        HintStoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for HintStoreAir {}
impl<F: Field> PartitionedBaseAir<F> for HintStoreAir {}

impl<AB: InteractionBuilder> Air<AB> for HintStoreAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let local_cols: &HintStoreCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1).unwrap();
        let next_cols: &HintStoreCols<AB::Var> = (*next).borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        builder.assert_bool(local_cols.is_single);
        builder.assert_bool(local_cols.is_buffer);
        builder.assert_bool(local_cols.is_buffer_start);
        builder
            .when(local_cols.is_buffer_start)
            .assert_one(local_cols.is_buffer);
        builder.assert_bool(local_cols.is_single + local_cols.is_buffer);

        let is_valid = local_cols.is_single + local_cols.is_buffer;
        let is_start = local_cols.is_single + local_cols.is_buffer_start;
        // `is_end` is false iff the next row is a buffer row that is not buffer start
        // This is boolean because is_buffer_start == 1 => is_buffer == 1
        // Note: every non-valid row has `is_end == 1`
        let is_end = not::<AB::Expr>(next_cols.is_buffer) + next_cols.is_buffer_start;

        let rem_words: AB::Expr = local_cols.rem_words.into();
        let next_rem_words: AB::Expr = next_cols.rem_words.into();

        let mem_ptr: AB::Expr = u16_limbs_to_ptr(&local_cols.mem_ptr_limbs);
        let next_mem_ptr: AB::Expr = u16_limbs_to_ptr(&next_cols.mem_ptr_limbs);

        // Constrain that if local is invalid, then the next state is invalid as well
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_valid.clone()))
            .assert_zero(next_cols.is_single + next_cols.is_buffer);

        // Constrain that when we start a buffer, the is_buffer_start is set to 1
        builder
            .when(local_cols.is_single)
            .assert_one(is_end.clone());
        builder
            .when_first_row()
            .assert_one(not::<AB::Expr>(local_cols.is_buffer) + local_cols.is_buffer_start);

        // read mem_ptr
        let mem_ptr_data: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_block(&local_cols.mem_ptr_limbs);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.mem_ptr_ptr),
                ),
                mem_ptr_data,
                timestamp_pp(),
                &local_cols.mem_ptr_aux_cols,
            )
            .eval(builder, is_start.clone());

        // read num_words
        let num_words_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            local_cols.rem_words.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.num_words_ptr),
                ),
                num_words_data,
                timestamp_pp(),
                &local_cols.num_words_aux_cols,
            )
            .eval(builder, local_cols.is_buffer_start);

        // write hint
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(MEMORY_AS),
                    byte_ptr_to_u16_ptr::<AB>(mem_ptr.clone()),
                ),
                local_cols.data.map(Into::into),
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, is_valid.clone());
        let expected_opcode = (local_cols.is_single
            * AB::F::from_usize(HINT_STORED as usize + self.offset))
            + (local_cols.is_buffer * AB::F::from_usize(HINT_BUFFER as usize + self.offset));

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode,
                [
                    local_cols.is_buffer * (local_cols.num_words_ptr),
                    local_cols.mem_ptr_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(MEMORY_AS),
                ],
                local_cols.from_state,
                rem_words.clone() * AB::F::from_usize(timestamp_delta),
            )
            .eval(builder, is_start.clone());

        assert!(
            (U16_BITS..=PTR_BITS).contains(&self.pointer_max_bits),
            "pointer_max_bits must fit in the low 32-bit mem_ptr view"
        );

        // Preventing mem_ptr overflow: mem_ptr < 2^pointer_max_bits.
        self.range_bus
            .range_check(
                ptr_bound_from_high_u16_expr(
                    local_cols.mem_ptr_limbs[PTR_U16_LIMBS - 1],
                    self.pointer_max_bits,
                ),
                U16_BITS,
            )
            .eval(builder, is_start.clone());
        // Preventing rem_words overflow: rem_words < 2^MAX_HINT_BUFFER_DWORDS_BITS.
        self.range_bus
            .range_check(
                local_cols.rem_words * AB::F::from_usize(1 << REM_WORDS_SHIFT),
                U16_BITS,
            )
            .eval(builder, is_start.clone());

        // buffer transition
        // `is_end` implies that the next row belongs to a new instruction,
        // which could be one of empty, hint_single, or hint_buffer
        // Constrains that when the current row is not empty and `is_end == 1`, then `rem_words` is
        // 1
        builder
            .when(is_valid)
            .when(is_end.clone())
            .assert_one(rem_words.clone());

        let mut when_buffer_transition = builder.when(not::<AB::Expr>(is_end.clone()));
        // Notes on `rem_words`: we constrain that `rem_words` doesn't overflow when we first read
        // it and that on each row it decreases by one (below). We also constrain that when
        // the current instruction ends then `rem_words` is 1. However, we don't constrain
        // that when `rem_words` is 1 then we have to end the current instruction.
        // The only way to exploit this if we to do some multiple of `p` number of additional
        // illegal `buffer` rows where `p` is the modulus of `F`. However, when doing `p`
        // additional `buffer` rows we will always increment `mem_ptr` to an illegal memory address
        // at some point, which prevents this exploit.
        when_buffer_transition.assert_one(rem_words.clone() - next_rem_words.clone());
        // Note: we only care about the composed `next_mem_ptr` and not the individual limbs:
        // the limbs do not need to be in the range, they can be anything that makes
        // `next_mem_ptr` correct -- this is just a way to avoid another column for `mem_ptr`.
        // The constraint we care about is `next.mem_ptr == local.mem_ptr + 8`. Since we increment
        // by `8` each time, any out of bounds memory access will be rejected by the memory bus
        // before we overflow the field.
        when_buffer_transition.assert_eq(
            next_mem_ptr.clone() - mem_ptr.clone(),
            AB::F::from_usize(REGISTER_NUM_LIMBS),
        );
        when_buffer_transition.assert_eq(
            timestamp + AB::F::from_usize(timestamp_delta),
            next_cols.from_state.timestamp,
        );
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct HintStoreExecutor {
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct HintStoreFiller {
    pointer_max_bits: usize,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

pub type HintStoreChip<F> = VmChipWrapper<F, HintStoreFiller>;
