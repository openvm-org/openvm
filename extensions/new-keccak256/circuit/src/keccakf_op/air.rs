use std::{
    borrow::Borrow,
    iter::{self},
};

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, utils::not};
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_rv32im_circuit::adapters::abstract_compose;
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::{
    keccakf_op::columns::{KeccakfOpCols, NUM_KECCAKF_OP_COLS},
    KECCAK_WORD_SIZE,
};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfOpAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Direct bus with keccakf pre- or post-state. Bus message is `is_post || timestamp ||
    /// state_u16_limbs`
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

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &KeccakfOpCols<_> = (*local).borrow();
        let next: &KeccakfOpCols<_> = (*next).borrow();

        let is_valid = local.is_valid;
        let is_after_valid = local.is_after_valid;
        builder.assert_bool(is_valid);
        builder.assert_bool(is_after_valid);
        // Only one of `is_valid` or `is_after_valid` can be true at a time
        builder.assert_zero(is_valid * is_after_valid);
        // Two row design: we always use two rows for handling of a single instruction execution
        // - We could do this without `is_after_valid`, but we wanted to avoid using as many `next`
        //   (rotation) columns
        builder.when(is_valid).assert_zero(next.is_valid);
        builder.when(is_valid).assert_one(next.is_after_valid);
        // Ensure that truly dummy rows have `is_valid = is_after_valid = 0`
        builder
            .when_transition()
            .when(not(is_valid))
            .assert_zero(next.is_after_valid);
        builder.when_first_row().assert_zero(is_after_valid);
        // Lemma: is_after_valid = 1 if and only if it is not the first row and the previous row has
        // is_valid = 1
        // Proof:
        // - (=>) is_after_valid = 0 on first row, so not first row. The previous row is not last.
        //   Hence
        // if previous row is not valid, current row must have `is_after_valid = 0`. Therefore
        // previous row must have `is_valid = 1`.
        // - (<=) direct constraint
        builder
            .when(is_valid)
            .assert_eq(local.timestamp, next.timestamp);

        let start_timestamp = local.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            start_timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };
        // ======== Read `rd` =========
        let rd_ptr = local.rd_ptr;
        let buffer_ptr_limbs = local.buffer_ptr_limbs;
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), rd_ptr),
                buffer_ptr_limbs,
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);
        // Range check that buffer_ptr_limbs fits in [0, 2^ptr_max_bits) as u32
        {
            assert!(self.ptr_max_bits >= RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1));
            let limb_shift = AB::F::from_canonical_usize(
                1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
            );
            let need_range_check = [
                buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
                buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            ];
            for pair in need_range_check.chunks_exact(2) {
                self.bitwise_lookup_bus
                    .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                    .eval(builder, is_valid);
            }
        }
        // Now it is safe to cast buffer_ptr to F
        let buffer_ptr: AB::Expr = abstract_compose(local.buffer_ptr_limbs);

        // ======== Constrain new writes of `buffer` to memory =========
        // NOTE: we use the _next_ row's `buffer` as the post-state
        for (word_idx, (prev_word, post_word, base_aux)) in izip!(
            local.buffer.chunks_exact(KECCAK_WORD_SIZE),
            next.buffer.chunks_exact(KECCAK_WORD_SIZE),
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
            let ptr = buffer_ptr.clone() + AB::F::from_canonical_usize(word_idx * KECCAK_WORD_SIZE);
            let prev_data: &[_; KECCAK_WORD_SIZE] = prev_word.try_into().unwrap();
            // We justify that post_word consists of bytes below
            let data: &[_; KECCAK_WORD_SIZE] = post_word.try_into().unwrap();
            let write_aux = MemoryWriteAuxCols {
                base: base_aux,
                prev_data: *prev_data,
            };
            self.memory_bridge.write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_MEMORY_AS), ptr),
                *data,
                timestamp_pp(),
                &write_aux,
            );
        }

        // ======== Execution bus =========
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(KeccakfOpcode::KECCAKF as usize + self.offset),
                [
                    rd_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(local.pc, local.timestamp),
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, is_valid);

        // ======== KeccakF State Interaction =======
        // Now we actually constrain that the pre- and post- buffer values are valid, but doing a
        // permutation check with the KeccakFPeripheryAir. We compose two u8 into a u16
        // since the keccakf periphery air uses u16 limbs
        // - `is_valid + is_after_valid` is boolean.
        // - is_valid = 1 => pre-state
        // - is_after_valid = 1 => post-state
        // - timestamp is the same on the two adjacent rows
        self.keccakf_state_bus.send(
            builder,
            iter::empty()
                .chain([is_after_valid.into(), local.timestamp.into()])
                .chain(
                    local
                        .buffer
                        .chunks(2)
                        .map(|pair| pair[0] * pair[1] * AB::F::from_canonical_u32(256)),
                ),
            is_valid + is_after_valid,
        );

        // We know that the pre-state buffer consists of bytes due to the invariant of Address Space
        // 2 in memory. The keccakf_state_bus guarantees that the post-state consists of
        // u16, but we still need to constrain that each pair actually consists of bytes.
        // NOTE[jpw]: this can be removed if AS2 cells are changed to u16s
        for pair in local.buffer.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, is_after_valid);
        }
    }
}
