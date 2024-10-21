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
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use test_case::test_matrix;

use super::{IsEqualAir, IsEqualIo};
use crate::{SubAir, TraceSubRowGenerator};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqualCols<T> {
    pub x: T,
    pub y: T,
    pub out: T,
    pub inv: T,
}

impl<F: Field> BaseAirWithPublicValues<F> for IsEqualAir {}
impl<F: Field> PartitionedBaseAir<F> for IsEqualAir {}
impl<F: Field> BaseAir<F> for IsEqualAir {
    fn width(&self) -> usize {
        IsEqualCols::<F>::width()
    }
}
impl<AB: AirBuilder> Air<AB> for IsEqualAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &IsEqualCols<_> = (*local).borrow();
        let io = IsEqualIo::new(
            local.x.into(),
            local.y.into(),
            local.out.into(),
            AB::Expr::one(),
        );

        SubAir::eval(self, builder, (io, local.inv));
    }
}

pub struct IsEqualChip<F> {
    pairs: Vec<(F, F)>,
}

impl<F: Field> IsEqualChip<F> {
    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        let air = IsEqualAir;
        assert!(self.pairs.len().is_power_of_two());
        let width = IsEqualCols::<F>::width();
        let mut rows = vec![F::zero(); width * self.pairs.len()];
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row: &mut IsEqualCols<F> = row.borrow_mut();
                row.x = x;
                row.y = y;
                row.out = F::from_bool(x == y);
                air.generate_subrow((x, y), &mut row.inv);
            });

        RowMajorMatrix::new(rows, width)
    }
}

#[test_matrix(
    [0,97,127],
    [0,23,97]
)]
fn test_single_is_equal(x: u32, y: u32) {
    let x = AbstractField::from_canonical_u32(x);
    let y = AbstractField::from_canonical_u32(y);

    let chip = IsEqualChip {
        pairs: vec![(x, y)],
    };

    let trace = chip.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![IsEqualAir], vec![trace])
        .expect("Verification failed");
}

#[test_matrix(
    [0,97,127],
    [0,23,97]
)]
fn test_single_is_zero_fail(x: u32, y: u32) {
    let x = AbstractField::from_canonical_u32(x);
    let y = AbstractField::from_canonical_u32(y);

    let chip = IsEqualChip {
        pairs: vec![(x, y)],
    };

    let mut trace = chip.generate_trace();
    trace.values[2] = if trace.values[2] == AbstractField::one() {
        AbstractField::zero()
    } else {
        AbstractField::one()
    };

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![IsEqualAir],
            vec![trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}
