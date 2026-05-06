use std::{array::from_fn, borrow::Borrow};

use itertools::{fold, Itertools};
use openvm_circuit_primitives::{utils::assert_array_eq, AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_continuations::utils::digests_to_poseidon2_input;
use openvm_deferral_circuit::canonicity::{CanonicityAuxCols, CanonicitySubAir};
use openvm_recursion_circuit::{
    bus::{Poseidon2PermuteBus, Poseidon2PermuteMessage},
    prelude::DIGEST_SIZE,
    primitives::bus::{RangeCheckerBus, RangeCheckerBusMessage},
};
use openvm_stark_backend::{interaction::InteractionBuilder, PartitionedBaseAir};
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;

use crate::bus::{OutputCommitBus, OutputCommitMessage, OutputValBus, OutputValMessage};

pub(crate) const F_NUM_BYTES: usize = 4;
pub(crate) const VALS_IN_DIGEST: usize = exact_div_or_panic(DIGEST_SIZE, F_NUM_BYTES);

const fn exact_div_or_panic(a: usize, b: usize) -> usize {
    assert!(b != 0 && a.is_multiple_of(b), "non-exact division");
    a / b
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DeferralOutputCommitCols<F> {
    pub is_valid: F,
    pub is_first: F,
    pub row_idx: F,
    pub output_len: F,

    pub input_vals: [F; DIGEST_SIZE],
    pub res_left: [F; DIGEST_SIZE],
    pub res_right: [F; DIGEST_SIZE],

    pub canonicity_aux: [CanonicityAuxCols<F>; VALS_IN_DIGEST],
}

#[derive(Debug, ColumnsAir)]
#[columns_via(DeferralOutputCommitCols<F>)]

pub struct DeferralOutputCommitAir {
    pub poseidon2_bus: Poseidon2PermuteBus,
    pub range_bus: RangeCheckerBus,
    pub output_val_bus: OutputValBus,
    pub output_commit_bus: OutputCommitBus,

    pub def_idx: usize,
}

impl<F> BaseAir<F> for DeferralOutputCommitAir {
    fn width(&self) -> usize {
        DeferralOutputCommitCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralOutputCommitAir {}
impl<F> PartitionedBaseAir<F> for DeferralOutputCommitAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DeferralOutputCommitAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("row 0 present");
        let next = main.row_slice(1).expect("row 1 present");

        let local: &DeferralOutputCommitCols<AB::Var> = (*local).borrow();
        let next: &DeferralOutputCommitCols<AB::Var> = (*next).borrow();

        let is_transition = next.is_valid - next.is_first;
        let is_last = local.is_valid - is_transition.clone();

        /*
         * Base constraints to ensure that all the valid rows are at the beginning,
         * and that there is at least one valid row. Additionally, row_idx starts at
         * 0 and increments.
         */
        builder.assert_bool(local.is_valid);
        builder.when_first_row().assert_one(local.is_valid);
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_eq(local.row_idx + AB::Expr::ONE, next.row_idx);

        /*
         * On the first row, input_vals should be [def_idx, output_len, 0, ...].
         * We constrain def_idx against a constant, and output_len against the
         * last valid row_idx.
         */
        let mut initial_state = [AB::Expr::ZERO; DIGEST_SIZE];
        initial_state[0] = AB::Expr::from_usize(self.def_idx);
        initial_state[1] = local.output_len.into();

        assert_array_eq(
            &mut builder.when_first_row(),
            local.input_vals,
            initial_state,
        );

        builder
            .when(is_transition.clone())
            .assert_eq(local.output_len, next.output_len);
        builder.when(is_last.clone()).assert_eq(
            local.output_len,
            local.row_idx * AB::Expr::from_usize(DIGEST_SIZE),
        );

        /*
         * On valid rows non-first we want to receive the next VALS_IN_DIGEST values
         * and constrain that input_vals is their byte decomposition.
         */
        let next_f: [_; VALS_IN_DIGEST] = local
            .input_vals
            .chunks(F_NUM_BYTES)
            .map(|c| {
                fold(c.iter().enumerate(), AB::Expr::ZERO, |acc, (i, byte)| {
                    acc + (AB::Expr::from_usize(1 << (i * 8)) * (*byte).into())
                })
            })
            .collect_array()
            .unwrap();

        for byte in local.input_vals {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: byte.into(),
                    max_bits: AB::Expr::from_u8(8),
                },
                local.is_valid - local.is_first,
            );
        }

        self.output_val_bus.receive(
            builder,
            OutputValMessage {
                values: next_f,
                idx: local.row_idx - AB::Expr::ONE,
            },
            local.is_valid - local.is_first,
        );

        /*
         * For each output value we need to constraint the canonicity of the byte
         * decomposition.
         */
        let rcs = local
            .input_vals
            .chunks(F_NUM_BYTES)
            .zip(local.canonicity_aux)
            .map(|(x, aux)| {
                CanonicitySubAir.assert_canonicity(
                    builder,
                    x,
                    &aux,
                    local.is_valid - local.is_first,
                )
            })
            .collect_vec();

        for rc in rcs {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: rc,
                    max_bits: AB::Expr::from_u8(8),
                },
                local.is_valid - local.is_first,
            );
        }

        /*
         * Compute the output commit and send it on the last row. We sponge hash each
         * valid input_vals and take the left child of the last row.
         */
        let next_capacity = from_fn(|i| is_transition.clone() * local.res_right[i]);
        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: digests_to_poseidon2_input(next.input_vals.map(Into::into), next_capacity),
                output: digests_to_poseidon2_input(next.res_left, next.res_right).map(Into::into),
            },
            next.is_valid,
        );

        self.output_commit_bus.send(
            builder,
            OutputCommitMessage {
                commit: local.res_left,
            },
            is_last,
        );
    }
}
