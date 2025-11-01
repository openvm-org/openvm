use core::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{Poseidon2Bus, Poseidon2BusMessage, TranscriptBus, TranscriptBusMessage},
    transcript::poseidon2::{CHUNK, POSEIDON2_WIDTH},
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct TranscriptCols<T> {
    pub proof_idx: T,
    pub is_proof_start: T,

    pub tidx: T,
    /// Indicator for sample/observe.
    pub is_sample: T,
    /// 0/1 indicators for the positions that we are absorbing/squeezing (i.e., are in
    /// the transcript). Constrained to be "decreasing".
    pub mask: [T; CHUNK],
    /// The lookup counts. Must be zero when corresponding mask is zero; otherwise unconstrained.
    pub lookup: [T; CHUNK],

    /// inidicator, whether the state is permutation from previous row's state
    pub permuted: T,
    /// The poseidon2 state.
    pub prev_state: [T; POSEIDON2_WIDTH],
    pub post_state: [T; POSEIDON2_WIDTH],
}

pub struct TranscriptAir {
    pub transcript_bus: TranscriptBus,
    pub poseidon2_bus: Poseidon2Bus,
}

impl<F: Field> BaseAir<F> for TranscriptAir {
    fn width(&self) -> usize {
        TranscriptCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for TranscriptAir {}
impl<F: Field> PartitionedBaseAir<F> for TranscriptAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for TranscriptAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TranscriptCols<AB::Var> = (*local).borrow();
        let next: &TranscriptCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////
        let is_valid = local.mask[0];

        builder.when(local.is_proof_start).assert_zero(local.tidx);
        builder.when(local.is_proof_start).assert_one(is_valid);
        builder.assert_bool(local.is_sample);

        // Initial state constraints
        for i in 0..CHUNK {
            builder
                .when(local.is_proof_start)
                .assert_eq(local.prev_state[i + CHUNK], AB::Expr::ZERO);

            builder
                .when(local.is_proof_start * (AB::Expr::ONE - local.mask[i]))
                .assert_eq(local.prev_state[i], AB::Expr::ZERO);
        }

        let mut count = AB::Expr::ZERO;
        let local_next_same_proof = next.mask[0] * (AB::Expr::ONE - next.is_proof_start);
        for i in 0..CHUNK {
            builder.assert_bool(local.mask[i]);
            count += local.mask[i].into();

            let skip = local.mask[i] - AB::Expr::ONE;
            if i < CHUNK - 1 {
                // if mask[i] = 0, then mask[i+1] = 0
                builder.when(skip.clone()).assert_zero(local.mask[i + 1]);
            }
            builder.when(skip).assert_zero(local.lookup[i]);

            // The state after permutation of this round, should check against next round's input
            // if next.mask[i] = 0 --> i-th not touched --> it should stay the same (if next is
            // valid)
            builder
                .when((AB::Expr::ONE - next.mask[i]) * local_next_same_proof.clone())
                .assert_eq(local.post_state[i], next.prev_state[i]);
            // When it's squeeze(sample), the state always remains the same
            builder
                .when(next.is_sample * local_next_same_proof.clone())
                .assert_eq(local.post_state[i], next.prev_state[i]);

            // The capacity part should always be the same
            builder
                .when(local_next_same_proof.clone()) // if next is valid
                .assert_eq(local.post_state[i + CHUNK], next.prev_state[i + CHUNK]);
        }

        builder
            .when(local_next_same_proof) // if next is valid
            .assert_eq(next.tidx, local.tidx + count);

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////
        for i in 0..CHUNK {
            // When absorb, it's normal order (0 -> RATE)
            let observe_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                value: local.prev_state[i].into(),
                is_sample: AB::Expr::ZERO,
            };
            // When squeeze, it's reverse RATE -> 0, so i means RATE - 1 - i
            let sample_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                value: local.prev_state[CHUNK - 1 - i].into(),
                is_sample: AB::Expr::ONE,
            };
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                observe_message,
                local.lookup[i] * (AB::Expr::ONE - local.is_sample),
            );
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                sample_message,
                local.lookup[i] * local.is_sample,
            );
        }

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2BusMessage {
                input: local.prev_state,
                output: local.post_state,
            },
            local.permuted,
        )
    }
}
