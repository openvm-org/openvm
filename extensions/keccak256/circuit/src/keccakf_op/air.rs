use std::{borrow::Borrow, iter};

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::{offline_checker::MemoryBridge, MemoryAddress},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus, ColumnsAir,
};
use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::expand_to_rv64_block;
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::keccakf_op::columns::{KeccakfOpCols, BUFFER_PTR_NUM_LIMBS, NUM_KECCAKF_OP_COLS};

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(KeccakfOpCols<u8>)]
pub struct KeccakfOpAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Kept for parity with the rest of the keccak256 extension's bus wiring; the chip
    /// no longer emits any 8-bit bitwise-lookup messages now that `buffer_ptr` is stored
    /// as u16 cells (the high-cell range check moved to `range_bus` to support 16-bit
    /// cells).
    #[allow(dead_code)]
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Direct bus with keccakf pre- or post-state. Bus message is
    /// ```text
    /// is_post || timestamp || state_u16_limbs
    /// ```
    pub keccakf_state_bus: PermutationCheckBus,
    /// Used to range-check the u16 high cell of `buffer_ptr` after scaling.
    pub range_bus: VariableRangeCheckerBus,
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for KeccakfOpAir {}
impl<F> PartitionedBaseAir<F> for KeccakfOpAir {}
impl<F> BaseAir<F> for KeccakfOpAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_OP_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfOpAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0).unwrap();
        let local: &KeccakfOpCols<_> = (*local).borrow();

        let is_valid = local.is_valid;
        builder.assert_bool(is_valid);

        let start_timestamp = local.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            start_timestamp + AB::F::from_usize(timestamp_delta - 1)
        };
        // ======== Read `rd` =========
        let rd_ptr = local.rd_ptr;
        // Build full BLOCK_FE_WIDTH (4) cell data array; `buffer_ptr_limbs` already
        // covers the low 32 bits of the 8-byte RV64 register as 2-byte cells, so it
        // matches the bus message shape directly. The upper 32 bits of the register
        // are hardcoded to zero by `expand_to_rv64_block`.
        let buffer_ptr_data: [AB::Expr; BLOCK_FE_WIDTH] =
            expand_to_rv64_block(&local.buffer_ptr_limbs);
        self.memory_bridge
            .read_4(
                MemoryAddress::new(AB::F::from_u32(RV64_REGISTER_AS), rd_ptr),
                buffer_ptr_data,
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);

        // Range check that buffer_ptr < 2^ptr_max_bits. `buffer_ptr_limbs[1]` is the
        // high u16 cell (covering bits [16, 32)); scaling by `1 << (32 - ptr_max_bits)`
        // and range-checking the result to 16 bits forces the cell into
        // `[0, 2^(ptr_max_bits - 16))`.
        assert!(
            (16..=32).contains(&self.ptr_max_bits),
            "ptr_max_bits must be in [16, 32] for the buffer_ptr range check"
        );
        self.range_bus
            .range_check(
                local.buffer_ptr_limbs[BUFFER_PTR_NUM_LIMBS - 1]
                    * AB::F::from_usize(1 << (32 - self.ptr_max_bits)),
                16,
            )
            .eval(builder, is_valid);
        // Now it is safe to cast buffer_ptr to F: compose the 2 u16 cells with base 2^16.
        let mut buffer_ptr = AB::Expr::ZERO;
        for i in (0..BUFFER_PTR_NUM_LIMBS).rev() {
            buffer_ptr = buffer_ptr * AB::F::from_u32(1 << 16) + local.buffer_ptr_limbs[i];
        }

        // ======== Constrain new writes of `buffer` to memory =========
        // Note: the post-state u16 cells are NOT individually range-checked. Each cell
        // is used only twice: once on the keccakf_state_bus (which constrains the value
        // to a canonical u16 produced by the keccakf periphery) and once on the memory
        // bus (where AS2 is u16-celled so the consumer also sees the value as a u16).
        // Any field-element value that matches the packed u16 satisfies both buses
        // identically.
        // NOTE: we use the _next_ row's `buffer` as the pre-state
        for (word_idx, (prev_word, post_word, base_aux)) in izip!(
            local.preimage.chunks_exact(BLOCK_FE_WIDTH),
            local.postimage.chunks_exact(BLOCK_FE_WIDTH),
            local.buffer_word_aux
        )
        .enumerate()
        {
            // Safety:
            // - we range checked that buffer_ptr < 2^ptr_max_bits but not that buffer_ptr +
            //   KECCAK_WIDTH_BYTES is in range.
            // - the previous range check implies `buffer_ptr + KECCAK_WIDTH_BYTES` does not
            //   overflow the field `F` hence it is safe to consider `ptr` as a field element.
            // - the memory_bridge.write at `ptr` consists of a receive on memory bus at a previous
            //   timestamp. The only way this bus interaction could balance is if there was already
            //   a previous valid write at `ptr`. Assuming the invariant that all previous memory
            //   accesses are valid and timestamp always moves forward, the new write to `ptr` must
            //   be valid as well.
            // Memory is byte-addressed; each chunk of `BLOCK_FE_WIDTH` u16 cells covers
            // `MEMORY_BLOCK_BYTES` bytes, so we step by `MEMORY_BLOCK_BYTES`.
            let ptr = buffer_ptr.clone() + AB::F::from_usize(word_idx * MEMORY_BLOCK_BYTES);
            let prev_data: [AB::Expr; BLOCK_FE_WIDTH] =
                std::array::from_fn(|i| prev_word[i].into());
            let data: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|i| post_word[i].into());
            self.memory_bridge
                .write_4_with_prev(
                    MemoryAddress::new(AB::F::from_u32(RV64_MEMORY_AS), ptr),
                    data,
                    prev_data,
                    timestamp_pp(),
                    &base_aux,
                )
                .eval(builder, is_valid);
        }

        // ======== Execution bus =========
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_usize(KeccakfOpcode::KECCAKF as usize + self.offset),
                [
                    rd_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::from_u32(RV64_MEMORY_AS),
                ],
                ExecutionState::new(local.pc, local.timestamp),
                AB::F::from_usize(timestamp_delta),
            )
            .eval(builder, is_valid);

        // ======== KeccakF State Interaction =======
        // Now we actually constrain that the pre- and post- buffer values are valid, by
        // doing a permutation check with the KeccakFPeripheryAir. The chip stores
        // `preimage` / `postimage` directly as u16 cells (matching the periphery's u16
        // limb shape), so the bus payload uses the columns as-is.
        //
        // We use two interactions bound with the same timestamp to avoid having a really large
        // message length.
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([AB::Expr::ZERO, local.timestamp.into()])
                .chain(local.preimage.iter().copied().map(Into::into)),
            is_valid,
        );
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([AB::Expr::ONE, local.timestamp.into()])
                .chain(local.postimage.iter().copied().map(Into::into)),
            is_valid,
        );
    }
}
