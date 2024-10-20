use std::sync::Arc;

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::DenseMatrix;

use super::{super::is_less_than::IsLessThanChip, columns::IsLessThanCols};
use crate::{
    is_less_than::IsLessThanAir,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};

impl IsLessThanChip {
    pub fn generate_trace<F: Field>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        let width: usize = IsLessThanCols::<F>::width(&self.air);

        let mut rows_concat = vec![F::zero(); width * pairs.len()];
        for (i, (x, y)) in pairs.iter().enumerate() {
            let mut lt_cols =
                IsLessThanColsMut::<F>::from_slice(&mut rows_concat[i * width..(i + 1) * width]);

            self.air
                .generate_trace_row(*x, *y, &self.range_checker, &mut lt_cols);
        }

        RowMajorMatrix::new(rows_concat, width)
    }
}

fn get_tester_is_lt_chip() -> IsLessThanChip {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    IsLessThanChip::new(max_bits, range_checker)
}

#[test]
fn test_flatten_fromslice_roundtrip() {
    let lt_air = IsLessThanAir::new(VariableRangeCheckerBus::new(0, 8), 16);

    let num_cols = IsLessThanCols::<usize>::width(&lt_air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = IsLessThanCols::<usize>::from_slice(&all_cols);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_is_less_than_chip_lt() {
    let chip = get_tester_is_lt_chip();
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![chip.air, chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let chip = get_tester_is_lt_chip();
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![chip.air, chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_negative() {
    let chip = get_tester_is_lt_chip();
    let mut trace = chip.generate_trace(vec![(446, 553)]);
    let range_trace = chip.range_checker.generate_trace();

    trace.values[2] = AbstractField::from_canonical_u64(0);

    disable_debug_builder();
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
