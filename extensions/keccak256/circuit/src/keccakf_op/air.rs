use std::{borrow::Borrow, iter};

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxInput},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerBus, ColumnsAir};
use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::{
    eval_add_const_u16_limbs, eval_byte_ptr_limbs_to_u16_cell_ptr_limbs, expand_to_rv64_block,
    reg_byte_ptr_to_cell_ptr_limbs,
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
                    // Register byte pointers are small: `rd_ptr / 2` in the low cell limb.
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(rd_ptr),
                ),
                buffer_ptr_data,
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);

        // Convert the base `buffer` *byte* pointer to base AS-native u16 *cell* pointer limbs.
        let buffer_byte_limbs: [AB::Expr; 2] =
            std::array::from_fn(|i| local.buffer_ptr_limbs[i].into());
        let buffer_base_cell_ptr = eval_byte_ptr_limbs_to_u16_cell_ptr_limbs::<AB>(
            builder,
            self.range_bus,
            buffer_byte_limbs,
            local.buffer_cell_carry,
            self.ptr_max_bits,
            is_valid.into(),
        );
        // Cell-pointer stride (in u16 cells) between consecutive heap blocks.
        let cell_ptr_block_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;

        // ======== Constrain new writes of `buffer` to memory =========
        // Keccak state and memory both consume these values as packed u16 cells.
        for (word_idx, (prev_word, post_word, base_aux, add_carry)) in izip!(
            local.preimage.chunks_exact(BLOCK_FE_WIDTH),
            local.postimage.chunks_exact(BLOCK_FE_WIDTH),
            local.buffer_word_aux,
            local.buffer_word_add_carry
        )
        .enumerate()
        {
            // Safety:
            // - `buffer_base_cell_ptr` is range-checked to be a canonical cell pointer below
            //   `2^cell_max_bits`, and each `eval_add_const_u16_limbs` range-checks the new low
            //   limb, so the per-block cell pointer is canonical.
            // - the memory_bridge.write at this cell pointer consists of a receive on memory bus at
            //   a previous timestamp. The only way this bus interaction could balance is if there
            //   was already a previous valid write there. Assuming the invariant that all previous
            //   memory accesses are valid and timestamp always moves forward, the new write must be
            //   valid as well.
            let block_cell_ptr = eval_add_const_u16_limbs::<AB>(
                builder,
                self.range_bus,
                buffer_base_cell_ptr.clone(),
                word_idx as u32 * cell_ptr_block_stride,
                add_carry,
                is_valid.into(),
            );
            let prev_data: [AB::Expr; BLOCK_FE_WIDTH] =
                std::array::from_fn(|i| prev_word[i].into());
            let data: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|i| post_word[i].into());
            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::F::from_u32(RV64_MEMORY_AS), block_cell_ptr),
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
