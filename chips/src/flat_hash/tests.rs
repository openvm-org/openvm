use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::{config::baby_bear_poseidon2::run_simple_test, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use super::PageController;

#[test]
fn test_single_is_zero() {
    let chip = PageController::new(8, 4, 5, 2, 3, 0, 0);

    let mut rng = create_seeded_rng();
    let x = (0..chip.air.page_height)
        .map(|_| {
            (0..chip.air.page_width)
                // .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..100)))
                .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..3)))
                .collect()
        })
        .collect::<Vec<Vec<BabyBear>>>();

    let pageread_trace = chip.generate_trace(x);
    let hash_chip = chip.hash_chip.lock();
    let hash_chip_trace = hash_chip.generate_cached_trace();

    let all_chips: Vec<&dyn AnyRap<_>> = vec![&chip.air, &hash_chip.air];

    let all_traces = vec![pageread_trace.clone(), hash_chip_trace.clone()];

    let pis = pageread_trace
        .values
        .iter()
        .rev()
        .take(chip.air.hash_width)
        .rev()
        .take(chip.air.digest_width)
        .cloned()
        .collect::<Vec<_>>();
    let all_pis = vec![pis, vec![]];

    run_simple_test(all_chips, all_traces, all_pis).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    let chip = PageController::new(8, 4, 5, 2, 3, 0, 0);

    let mut rng = create_seeded_rng();
    let x = (0..chip.air.page_height)
        .map(|_| {
            (0..chip.air.page_width)
                // .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..100)))
                .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..3)))
                .collect()
        })
        .collect::<Vec<Vec<BabyBear>>>();

    let mut pageread_trace = chip.generate_trace(x);
    pageread_trace.values[0] = BabyBear::one();

    let hash_chip = chip.hash_chip.lock();
    let hash_chip_trace = hash_chip.generate_cached_trace();

    let all_chips: Vec<&dyn AnyRap<_>> = vec![&chip.air, &hash_chip.air];

    let all_traces = vec![pageread_trace.clone(), hash_chip_trace.clone()];

    let pis = pageread_trace
        .values
        .iter()
        .rev()
        .take(chip.air.hash_width)
        .rev()
        .take(chip.air.digest_width)
        .cloned()
        .collect::<Vec<_>>();
    let all_pis = vec![pis, vec![]];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test(all_chips, all_traces, all_pis),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected constraint to fail"
    );
}
