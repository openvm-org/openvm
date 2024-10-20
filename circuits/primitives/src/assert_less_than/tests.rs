use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{
    prover::USE_DEBUG_BUILDER,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    verifier::VerificationError,
};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use derive_new::new;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};

use super::*;
use crate::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};

// We only create an Air for testing purposes

// repr(C) is needed to make sure that the compiler does not reorder the fields
// we assume the order of the fields when using borrow or borrow_mut
#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug, new)]
pub struct AssertLessThanCols<T, const AUX_LEN: usize> {
    pub x: T,
    pub y: T,
    pub aux: LessThanAuxCols<T, AUX_LEN>,
}

#[derive(Clone, Copy)]
pub struct AssertLtTestAir<const AUX_LEN: usize>(pub AssertLessThanAir);

impl<F: Field, const AUX_LEN: usize> BaseAirWithPublicValues<F> for AssertLtTestAir<AUX_LEN> {}
impl<F: Field, const AUX_LEN: usize> PartitionedBaseAir<F> for AssertLtTestAir<AUX_LEN> {}
impl<F: Field, const AUX_LEN: usize> BaseAir<F> for AssertLtTestAir<AUX_LEN> {
    fn width(&self) -> usize {
        AssertLessThanCols::<F, AUX_LEN>::width()
    }
}
impl<AB: InteractionBuilder, const AUX_LEN: usize> Air<AB> for AssertLtTestAir<AUX_LEN> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &AssertLessThanCols<_, AUX_LEN> = (*local).borrow();

        let io = AssertLessThanIo::new(local.x, local.y, AB::F::one());
        self.0.eval(builder, (io, &local.aux.lower_decomp));
    }
}

#[derive(Clone)]
pub struct AssertLessThanChip<const AUX_LEN: usize> {
    pub air: AssertLtTestAir<AUX_LEN>,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl<const AUX_LEN: usize> AssertLessThanChip<AUX_LEN> {
    pub fn generate_trace<F: Field>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        assert!(pairs.len().is_power_of_two());
        let width: usize = AssertLessThanCols::<F, AUX_LEN>::width();

        let mut rows = vec![F::zero(); width * pairs.len()];
        for (row, (x, y)) in rows.chunks_mut(width).zip(pairs) {
            let row: &mut AssertLessThanCols<F, AUX_LEN> = row.borrow_mut();
            row.x = F::from_canonical_u32(x);
            row.y = F::from_canonical_u32(y);
            self.air
                .0
                .generate_subrow((&self.range_checker, x, y), &mut row.aux.lower_decomp);
        }

        RowMajorMatrix::new(rows, width)
    }
}

impl<const AUX_LEN: usize> AssertLessThanChip<AUX_LEN> {
    pub fn new(
        bus: VariableRangeCheckerBus,
        max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        Self {
            air: AssertLtTestAir(AssertLessThanAir::new(bus, max_bits)),
            range_checker,
        }
    }
}

#[test]
fn test_borrow_mut_roundtrip() {
    const AUX_LEN: usize = 2; // number of auxilliary columns is two

    let num_cols = AssertLessThanCols::<usize, AUX_LEN>::width();
    let mut all_cols = (0..num_cols).collect::<Vec<usize>>();

    let lt_cols: &mut AssertLessThanCols<_, AUX_LEN> = all_cols[..].borrow_mut();

    lt_cols.x = 2;
    lt_cols.y = 8;
    lt_cols.aux.lower_decomp[0] = 1;
    lt_cols.aux.lower_decomp[1] = 0;

    assert_eq!(all_cols[0], 2);
    assert_eq!(all_cols[1], 8);
    assert_eq!(all_cols[2], 1);
    assert_eq!(all_cols[3], 0);
}

#[test]
fn test_assert_less_than_chip_lt() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (0, 1), (28, 120), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![chip.air, chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 4;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let trace = chip.generate_trace(vec![(14321, 26883), (0, 1), (28, 120), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![chip.air, chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_assert_less_than_negative_1() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 2;

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let mut trace = chip.generate_trace(vec![(28, 29)]);
    let range_trace = chip.range_checker.generate_trace();

    // Make the trace invalid
    trace.values.swap(0, 1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![chip.air, chip.range_checker.air],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_assert_less_than_negative_2() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    const AUX_LEN: usize = 4;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    let chip = AssertLessThanChip::<AUX_LEN>::new(bus, max_bits, range_checker);
    let mut trace = chip.generate_trace(vec![(28, 29)]);
    let range_trace = chip.range_checker.generate_trace();

    // Make the trace invalid
    trace.values[2] = AbstractField::from_canonical_u64(1 << decomp as u64);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![chip.air, chip.range_checker.air],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
