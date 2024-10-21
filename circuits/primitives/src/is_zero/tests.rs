use std::borrow::{Borrow, BorrowMut};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use test_case::test_case;

use super::{IsZeroAir, IsZeroIo};
use crate::{SubAir, TraceSubRowGenerator};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsZeroCols<T> {
    pub x: T,
    pub out: T,
    pub inv: T,
}

impl<F: Field> BaseAirWithPublicValues<F> for IsZeroAir {}
impl<F: Field> PartitionedBaseAir<F> for IsZeroAir {}
impl<F: Field> BaseAir<F> for IsZeroAir {
    fn width(&self) -> usize {
        IsZeroCols::<F>::width()
    }
}
impl<AB: AirBuilder> Air<AB> for IsZeroAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &IsZeroCols<_> = (*local).borrow();
        let io = IsZeroIo::new(local.x.into(), local.out.into());

        SubAir::eval(self, builder, (io, local.inv));
    }
}

pub struct IsZeroChip<F> {
    x: Vec<F>,
}

impl<F: Field> IsZeroChip<F> {
    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        let air = IsZeroAir;
        assert!(self.x.len().is_power_of_two());
        let width = IsZeroCols::<F>::width();
        let mut rows = vec![F::zero(); width * self.x.len()];
        rows.par_chunks_mut(width).zip(self.x).for_each(|(row, x)| {
            let row: &mut IsZeroCols<F> = row.borrow_mut();
            row.x = x;
            row.out = F::from_bool(x.is_zero());
            air.generate_subrow(x, &mut row.inv);
        });

        RowMajorMatrix::new(rows, width)
    }
}

#[test_case(97 ; "97 => 0")]
#[test_case(0 ; "0 => 1")]
fn test_single_is_zero(x: u32) {
    let chip = IsZeroChip {
        x: vec![BabyBear::from_canonical_u32(x)],
    };
    let trace = chip.generate_trace();

    assert_eq!(trace.get(0, 1), AbstractField::from_bool(x == 0));

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![IsZeroAir], vec![trace])
        .expect("Verification failed");
}

#[test_case([0, 1, 2, 7], [1, 0, 0, 0] ; "0, 1, 2, 7 => 1, 0, 0, 0")]
#[test_case([97, 23, 179, 0], [0, 0, 0, 1] ; "97, 23, 179, 0 => 0, 0, 0, 1")]
fn test_vec_is_zero(x_vec: [u32; 4], expected: [u32; 4]) {
    let x_vec = x_vec
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsZeroChip { x: x_vec };

    let trace = chip.generate_trace();

    for (i, value) in expected.iter().enumerate() {
        assert_eq!(
            trace.values[3 * i + 1],
            AbstractField::from_canonical_u32(*value)
        );
    }

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![IsZeroAir], vec![trace])
        .expect("Verification failed");
}

#[test_case(97 ; "97 => 0")]
#[test_case(0 ; "0 => 1")]
fn test_single_is_zero_fail(x: u32) {
    let x = AbstractField::from_canonical_u32(x);
    let chip = IsZeroChip { x: vec![x] };

    let mut trace = chip.generate_trace();
    trace.values[1] = BabyBear::one() - trace.values[1];

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![IsZeroAir],
            vec![trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test_case([1, 2, 7, 0], [0, 0, 0, 1] ; "1, 2, 7, 0 => 0, 0, 0, 1")]
#[test_case([97, 0, 179, 0], [0, 1, 0, 1] ; "97, 0, 179, 0 => 0, 1, 0, 1")]
fn test_vec_is_zero_fail(x_vec: [u32; 4], expected: [u32; 4]) {
    let x_vec: Vec<BabyBear> = x_vec
        .into_iter()
        .map(BabyBear::from_canonical_u32)
        .collect();

    let chip = IsZeroChip { x: x_vec };

    let mut trace = chip.generate_trace();

    disable_debug_builder();
    for (i, _value) in expected.iter().enumerate() {
        trace.row_mut(i)[1] = BabyBear::one() - trace.row_mut(i)[1];
        assert_eq!(
            BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
                any_rap_arc_vec![IsZeroAir],
                vec![trace.clone()]
            )
            .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        trace.row_mut(i)[1] = BabyBear::one() - trace.row_mut(i)[1];
    }
}
