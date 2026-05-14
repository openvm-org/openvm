use std::{borrow::Borrow, iter};

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::{pack_u8_for_bus, MemoryBridge},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, utils::compose, ColumnsAir,
};
use openvm_instructions::riscv::{
    RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::expand_to_rv64_register;
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
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Direct bus with keccakf pre- or post-state. Bus message is
    /// ```text
    /// is_post || timestamp || state_u16_limbs
    /// ```
    pub keccakf_state_bus: PermutationCheckBus,
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
        // Build full 8-element data array with upper 4 limbs hardcoded to zero
        let buffer_ptr_limbs: [AB::Expr; RV64_REGISTER_NUM_LIMBS] =
            expand_to_rv64_register(&local.buffer_ptr_limbs);
        self.memory_bridge
            .read_4(
                MemoryAddress::new(AB::F::from_u32(RV64_REGISTER_AS), rd_ptr),
                pack_u8_for_bus::<AB>(&buffer_ptr_limbs),
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);

        // Range check that buffer_ptr_limbs fits in [0, 2^ptr_max_bits) as u32
        {
            assert!(self.ptr_max_bits >= RV64_CELL_BITS * (RV64_WORD_NUM_LIMBS - 1));
            let limb_shift =
                AB::F::from_usize(1 << (RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.ptr_max_bits));
            let msb = local.buffer_ptr_limbs[RV64_WORD_NUM_LIMBS - 1];
            self.bitwise_lookup_bus
                .send_range(msb * limb_shift, msb * limb_shift)
                .eval(builder, is_valid);
        }
        // u8-range-check the non-MSB buffer_ptr_limbs (read from rs1 register).
        // After the bus pack pairs share a packed field element; without
        // local u8 checks the prover could re-split values between adjacent limbs.
        // The MSB already has a tighter range via the limb_shift check above; we
        // pair limb[2] with limb[3] here for an additional (redundant on the MSB
        // but tight on limb[2]) u8 check.
        self.bitwise_lookup_bus
            .send_range(local.buffer_ptr_limbs[0], local.buffer_ptr_limbs[1])
            .eval(builder, is_valid);
        self.bitwise_lookup_bus
            .send_range(local.buffer_ptr_limbs[2], local.buffer_ptr_limbs[3])
            .eval(builder, is_valid);
        // Now it is safe to cast buffer_ptr to F
        let buffer_ptr: AB::Expr = compose(&local.buffer_ptr_limbs[..], RV64_CELL_BITS);

        // ======== Constrain new writes of `buffer` to memory =========
        // Note: the post-state byte pairs are NOT individually byte-range-checked. Each pair
        // (post_word[2k], post_word[2k+1]) is used only in packed `pair[0] + 256·pair[1]` form:
        // once on the keccakf_state_bus (which constrains the packed value to a canonical u16
        // produced by the keccakf periphery) and once on the memory bus (where AS2 is u16-celled
        // so the consumer also sees the packed value, never the individual byte). Any byte-pair
        // decomposition that matches the packed u16 satisfies both buses identically.
        // NOTE: we use the _next_ row's `buffer` as the pre-state
        for (word_idx, (prev_word, post_word, base_aux)) in izip!(
            local.preimage.chunks_exact(MEMORY_BLOCK_BYTES),
            local.postimage.chunks_exact(MEMORY_BLOCK_BYTES),
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
            let prev_data: [AB::Expr; MEMORY_BLOCK_BYTES] =
                std::array::from_fn(|i| prev_word[i].into());
            // post_word consists of bytes due to range checks above
            let data: [AB::Expr; MEMORY_BLOCK_BYTES] = std::array::from_fn(|i| post_word[i].into());
            self.memory_bridge
                .write_4_with_prev(
                    MemoryAddress::new(AB::F::from_u32(RV64_MEMORY_AS), ptr),
                    pack_u8_for_bus::<AB>(&data),
                    pack_u8_for_bus::<AB>(&prev_data),
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
        // Now we actually constrain that the pre- and post- buffer values are valid, but doing a
        // permutation check with the KeccakFPeripheryAir. We compose two u8 into a u16
        // since the keccakf periphery air uses u16 limbs
        //
        // We use two interactions bound with the same timestamp to avoid having a really large
        // message length.
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([AB::Expr::ZERO, local.timestamp.into()])
                .chain(
                    local
                        .preimage
                        .chunks(2)
                        .map(|pair| pair[0] + pair[1] * AB::F::from_u32(256)),
                ),
            is_valid,
        );
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([AB::Expr::ONE, local.timestamp.into()])
                .chain(
                    local
                        .postimage
                        .chunks(2)
                        .map(|pair| pair[0] + pair[1] * AB::F::from_u32(256)),
                ),
            is_valid,
        );
    }
}
