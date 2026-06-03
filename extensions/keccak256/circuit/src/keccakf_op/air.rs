use std::{borrow::Borrow, iter};

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxInput},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerBus, ColumnsAir, U16_BITS};
use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr, expand_to_rv64_block, ptr_bound_from_high_u16_expr, u16_limbs_to_ptr,
    RV64_PTR_U16_LIMBS,
};
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::keccakf_op::columns::{KeccakfOpCols, NUM_KECCAKF_OP_COLS};

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(KeccakfOpCols<u8>)]
pub struct KeccakfOpAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Direct bus with keccakf pre- or post-state. Bus message is
    /// ```text
    /// is_post || timestamp || state_u16_limbs
    /// ```
    pub keccakf_state_bus: PermutationCheckBus,
    /// Range-checks the u16 high cell of `buffer_ptr` after scaling.
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
        // Register read: low 32 bits as u16 cells, zero-extended to one memory block.
        let buffer_ptr_data: [AB::Expr; BLOCK_FE_WIDTH] =
            expand_to_rv64_block(&local.buffer_ptr_limbs);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(rd_ptr),
                ),
                buffer_ptr_data,
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);

        self.range_bus
            .range_check(
                ptr_bound_from_high_u16_expr::<AB::Expr, _>(
                    local.buffer_ptr_limbs[RV64_PTR_U16_LIMBS - 1],
                    self.ptr_max_bits,
                ),
                U16_BITS,
            )
            .eval(builder, is_valid);
        let buffer_ptr = u16_limbs_to_ptr(&local.buffer_ptr_limbs);

        // ======== Constrain new writes of `buffer` to memory =========
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
            let ptr = buffer_ptr.clone() + AB::F::from_usize(word_idx * MEMORY_BLOCK_BYTES);
            let prev_data: [AB::Expr; BLOCK_FE_WIDTH] =
                std::array::from_fn(|i| prev_word[i].into());
            let data: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|i| post_word[i].into());
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::F::from_u32(RV64_MEMORY_AS),
                        byte_ptr_to_u16_ptr::<AB>(ptr),
                    ),
                    data,
                    timestamp_pp(),
                    MemoryWriteAuxInput::from_prev_data_exprs(&base_aux, prev_data),
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
        // Now we actually constrain that the pre- and post-buffer values are valid, by doing a
        // permutation check with the KeccakFPeripheryAir. The state columns are already u16
        // limbs, matching the keccakf periphery bus.
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
