use crate::flat_hash::FlatHashChip;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

#[test]
fn test_single_is_zero() {
    let chip = FlatHashChip::<10, 3>::new(10, 4, 5, 2, 3, 0, 0);

    let mut rng = create_seeded_rng();
    let x = (0..chip.page_height)
        .map(|_| {
            (0..chip.page_width)
                .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..100)))
                .collect()
        })
        .collect();

    let trace = chip.generate_trace(x);

    // assert_eq!(trace.values[1], AbstractField::from_canonical_u32(0));

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}
