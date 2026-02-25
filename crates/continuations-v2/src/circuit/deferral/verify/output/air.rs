use std::borrow::Borrow;

use itertools::{fold, izip};
use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    AlignedBorrow,
};
use openvm_stark_backend::{interaction::InteractionBuilder, PartitionedBaseAir};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    prelude::DIGEST_SIZE,
    primitives::bus::{RangeCheckerBus, RangeCheckerBusMessage},
};

use crate::circuit::{
    deferral::verify::bus::{OutputCommitBus, OutputCommitMessage, OutputValBus, OutputValMessage},
    root::digests_to_poseidon2_input,
};

pub(crate) const F_NUM_BYTES: usize = 4;
pub(crate) const VALS_IN_DIGEST: usize = exact_div_or_panic(DIGEST_SIZE, F_NUM_BYTES);

const fn exact_div_or_panic(a: usize, b: usize) -> usize {
    assert!(b != 0 && a % b == 0, "non-exact division");
    a / b
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralOutputCommitCols<F> {
    pub is_valid: F,
    pub is_first: F,

    pub state: [F; DIGEST_SIZE],
    pub next_bytes: [F; DIGEST_SIZE],

    pub next_f: [F; VALS_IN_DIGEST],
    pub next_f_idx: F,
}

#[derive(Debug)]
pub struct DeferralOutputCommitAir {
    pub poseidon2_bus: Poseidon2CompressBus,
    pub range_bus: RangeCheckerBus,
    pub output_val_bus: OutputValBus,
    pub output_commit_bus: OutputCommitBus,
}

impl<F> BaseAir<F> for DeferralOutputCommitAir {
    fn width(&self) -> usize {
        DeferralOutputCommitCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralOutputCommitAir {}
impl<F> PartitionedBaseAir<F> for DeferralOutputCommitAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DeferralOutputCommitAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("row 0 present");
        let next = main.row_slice(1).expect("row 1 present");

        let local: &DeferralOutputCommitCols<AB::Var> = (*local).borrow();
        let next: &DeferralOutputCommitCols<AB::Var> = (*next).borrow();

        /*
         * Base constraints to ensure that all the valid rows are at the beginning,
         * and that there is at least one valid and one invalid row. The latter is
         * important because we send the output commit on the first invalid row.
         */
        builder.assert_bool(local.is_valid);
        builder.when_first_row().assert_one(local.is_valid);
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);
        builder.when_last_row().assert_zero(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        /*
         * On valid rows we want to receive the next VALS_IN_DIGEST output values
         * and constrain that next_bytes is their byte decomposition. To this end
         * we also constrain that next_f_idx increments by VALS_IN_DIGEST.
         */
        for (byte_decomp, next_f) in izip!(local.next_bytes.chunks(F_NUM_BYTES), local.next_f) {
            let composed_f = fold(
                byte_decomp.iter().enumerate(),
                AB::Expr::ZERO,
                |acc, (i, byte)| acc + (AB::Expr::from_usize(1 << (i * 8)) * byte.clone().into()),
            );
            builder.when(local.is_valid).assert_eq(composed_f, next_f);
        }

        builder.when_first_row().assert_zero(local.next_f_idx);
        builder
            .when_transition()
            .assert_eq(local.next_f_idx + AB::Expr::ONE, next.next_f_idx);

        for byte in local.next_bytes {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: byte.into(),
                    max_bits: AB::Expr::from_u8(8),
                },
                local.is_valid,
            );
        }

        self.output_val_bus.receive(
            builder,
            OutputValMessage {
                values: local.next_f,
                idx: local.next_f_idx,
            },
            local.is_valid,
        );

        /*
         * Compute the output commit and send it on the first invalid row. Note
         * that we do not do a hash on the first row.
         */
        assert_array_eq(&mut builder.when_first_row(), local.next_bytes, next.state);

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.state, local.next_bytes),
                output: next.state,
            },
            local.is_valid * not(local.is_first),
        );

        self.output_commit_bus.send(
            builder,
            OutputCommitMessage { commit: next.state },
            local.is_valid * not(next.is_valid),
        );
    }
}
