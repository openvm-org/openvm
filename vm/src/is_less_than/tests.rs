use std::sync::Arc;

use afs_primitives::{
    is_less_than::columns::IsLessThanIoCols, var_range::VariableRangeCheckerChip,
};
use ax_sdk::{
    config::baby_bear_blake3::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use rand::Rng;

use super::IsLessThanChip;

const RANGE_CHECK_BUS: usize = 1;
const LESS_THAN_BUS: usize = 2;
const DECOMP: usize = 10;

#[test]
fn test_less_than() {
    let mut rng = create_seeded_rng();
    let range_checker = Arc::new(VariableRangeCheckerChip::new(RANGE_CHECK_BUS, 1 << DECOMP));
    let mut chip: IsLessThanChip<BabyBear> =
        IsLessThanChip::new(LESS_THAN_BUS, 10, DECOMP, range_checker.clone());

    let x = rng.gen_range(1..=100);
    let y = rng.gen_range(1..=100);
    let less_than: u32 = if x < y { 1 } else { 0 };
    let x_b = BabyBear::from_canonical_u32(x);
    let y_b = BabyBear::from_canonical_u32(y);

    chip.compare((x_b, y_b));

    let range_trace: DenseMatrix<BabyBear> = range_checker.generate_trace();
    let trace = chip.generate_trace();

    let dummy_trace = RowMajorMatrix::new(
        vec![
            BabyBear::one(), // 0-th col is count for Dummy air.
            x_b,
            y_b,
            BabyBear::from_canonical_u32(less_than),
        ],
        IsLessThanIoCols::<BabyBear>::width() + 1,
    );
    let dummy_air =
        DummyInteractionAir::new(IsLessThanIoCols::<BabyBear>::width(), true, LESS_THAN_BUS);

    run_simple_test_no_pis(
        vec![&chip.air, &range_checker.air, &dummy_air],
        vec![trace.clone(), range_trace.clone(), dummy_trace.clone()],
    )
    .expect("test failed")
}
