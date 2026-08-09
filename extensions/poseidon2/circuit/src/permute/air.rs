use std::borrow::Borrow;

use itertools::{izip, Itertools};
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, ColumnsAir};
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_poseidon2_transpiler::Poseidon2Opcode;
use openvm_rv32im_circuit::adapters::abstract_compose;
use openvm_stark_backend::{
    interaction::{InteractionBuilder, LookupBus},
    p3_air::{Air, BaseAir},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::{
    canonicity::{CanonicitySubAir, F_NUM_BYTES},
    permute::columns::{Poseidon2PermuteOpCols, NUM_POSEIDON2_PERMUTE_OP_COLS},
    POSEIDON2_WORD_SIZE,
};

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Poseidon2PermuteOpCols<u8>)]
pub struct Poseidon2PermuteAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Direct bus with the periphery chip. Bus message is `input_words || output_words`, where
    /// each entry is a composed 4-byte word.
    pub poseidon2_bus: LookupBus,
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for Poseidon2PermuteAir {}
impl<F> PartitionedBaseAir<F> for Poseidon2PermuteAir {}
impl<F> BaseAir<F> for Poseidon2PermuteAir {
    fn width(&self) -> usize {
        NUM_POSEIDON2_PERMUTE_OP_COLS
    }
}

/// The canonicity sub-AIR works on the 4-byte little-endian decomposition of a field element, which
/// must line up exactly with how `postimage` words are composed for the poseidon2 bus.
const _: () = assert!(POSEIDON2_WORD_SIZE == F_NUM_BYTES);

impl<AB: InteractionBuilder> Air<AB> for Poseidon2PermuteAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0).unwrap();
        let local: &Poseidon2PermuteOpCols<_> = (*local).borrow();

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
        let buffer_ptr_limbs = local.buffer_ptr_limbs;
        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_u32(RV32_REGISTER_AS), rd_ptr),
                buffer_ptr_limbs,
                timestamp_pp(),
                &local.rd_aux,
            )
            .eval(builder, is_valid);
        // Range check that buffer_ptr_limbs fits in [0, 2^ptr_max_bits) as u32
        {
            assert!(self.ptr_max_bits >= RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1));
            let limb_shift = AB::F::from_usize(
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

        // ======== Constrain that post-state consists of canonical bytes =========
        // We know that the pre-state buffer consists of bytes due to the invariant of Address Space
        // 2 in memory. The post-state bytes, however, are prover-chosen witness values that we
        // write back to memory, so they need to be fully constrained here.
        //
        // First, each cell must actually be a byte for the word composition below to be valid.
        // NOTE: this can be removed if AS2 cells are changed to u16s
        for pair in local.postimage.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0], pair[1])
                .eval(builder, is_valid);
        }
        // Second, the byte decomposition must be _canonical_. Byte range checks alone are not
        // enough: `poseidon2_bus` only pins down the composed field element `y`, and the bytes
        // encoding the integer `y + F::ORDER_U32` compose to that same `y` (the sum is always
        // `< 2^32` since `y < p < 2^31`). Without this check a malicious prover could write those
        // alternate bytes to `[buffer_ptr:POSEIDON2_STATE_BYTES]_2` and still verify, so the
        // guest-visible result of `PERMUTE` would not be uniquely determined.
        let canonicity_rcs = izip!(
            local.postimage.chunks_exact(POSEIDON2_WORD_SIZE),
            local.postimage_canonicity_aux
        )
        .map(|(word, aux)| CanonicitySubAir.assert_canonicity(builder, word, &aux, is_valid.into()))
        .collect_vec();
        for rc_pair in canonicity_rcs.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(rc_pair[0].clone(), rc_pair[1].clone())
                .eval(builder, is_valid);
        }

        // ======== Constrain new writes of `buffer` to memory =========
        for (word_idx, (prev_word, post_word, base_aux)) in izip!(
            local.preimage.chunks_exact(POSEIDON2_WORD_SIZE),
            local.postimage.chunks_exact(POSEIDON2_WORD_SIZE),
            local.buffer_word_aux
        )
        .enumerate()
        {
            // Safety:
            // - we range checked that buffer_ptr < 2^ptr_max_bits but not that buffer_ptr +
            //   POSEIDON2_STATE_BYTES is in range.
            // - the previous range check implies `buffer_ptr + POSEIDON2_STATE_BYTES` does not
            //   overflow the field `F` hence it is safe to consider `ptr` as a field element.
            // - the memory_bridge.write at `ptr` consists of a receive on memory bus at a previous
            //   timestamp. The only way this bus interaction could balance is if there was already
            //   a previous valid write at `ptr`. Assuming the invariant that all previous memory
            //   accesses are valid and timestamp always moves forward, the new write to `ptr` must
            //   be valid as well.
            let ptr = buffer_ptr.clone() + AB::F::from_usize(word_idx * POSEIDON2_WORD_SIZE);
            let prev_data: &[_; POSEIDON2_WORD_SIZE] = prev_word.try_into().unwrap();
            // post_word is a canonical byte decomposition due to the checks above
            let data: &[_; POSEIDON2_WORD_SIZE] = post_word.try_into().unwrap();
            let write_aux = MemoryWriteAuxCols {
                base: base_aux,
                prev_data: *prev_data,
            };
            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::F::from_u32(RV32_MEMORY_AS), ptr),
                    *data,
                    timestamp_pp(),
                    &write_aux,
                )
                .eval(builder, is_valid);
        }

        // ======== Execution bus =========
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_usize(Poseidon2Opcode::PERMUTE as usize + self.offset),
                [
                    rd_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::from_u32(RV32_REGISTER_AS),
                    AB::Expr::from_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(local.pc, local.timestamp),
                AB::F::from_usize(timestamp_delta),
            )
            .eval(builder, is_valid);

        // ======== Poseidon2 State Interaction =======
        // Now we actually constrain that the pre- and post- buffer values are valid by looking up
        // the composed `[input || output]` words on the periphery's direct bus. The pre-state is
        // whatever the guest left in memory and is interpreted mod `p` (matching the host), while
        // the post-state bytes are pinned to the unique canonical encoding of the output words.
        self.poseidon2_bus.lookup_key(
            builder,
            local
                .preimage
                .chunks_exact(POSEIDON2_WORD_SIZE)
                .chain(local.postimage.chunks_exact(POSEIDON2_WORD_SIZE))
                .map(|word| {
                    word[0]
                        + word[1] * AB::F::from_u32(256)
                        + word[2] * AB::F::from_u32(65536)
                        + word[3] * AB::F::from_u32(1 << 24)
                }),
            is_valid,
        );
    }
}
