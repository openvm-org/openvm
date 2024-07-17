use crate::dummy_hash::DummyHashAir;
use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::{config::baby_bear_poseidon2::run_simple_test, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use super::PageController;

#[test]
fn test_single_is_zero() {
    let chip = PageController::new(10, 9, 5, 2, 3, 0, 1);
    let correct_height = chip.air.page_height.next_power_of_two();

    let mut rng = create_seeded_rng();
    let x = (0..chip.air.page_height)
        .map(|_| {
            (0..chip.air.page_width)
                .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..3)))
                .collect()
        })
        .collect::<Vec<Vec<BabyBear>>>();

    let requester = DummyInteractionAir::new(chip.air.page_width, true, chip.air.bus_index);

    let dummy_trace = RowMajorMatrix::new(
        x.iter()
            .flat_map(|row| [BabyBear::one()].into_iter().chain(row.iter().cloned()))
            .chain(
                std::iter::repeat(BabyBear::zero())
                    .take((chip.air.page_width + 1) * (correct_height - chip.air.page_height)),
            )
            .collect(),
        chip.air.page_width + 1,
    );

    let pageread_trace = chip.generate_trace(x.clone());

    let hash_chip_trace = {
        let hash_chip = chip.hash_chip.lock();
        hash_chip.generate_trace()
    };

    let hash_chip_air = chip.hash_chip.lock();
    let all_airs: Vec<&dyn AnyRap<_>> = vec![&chip.air, &hash_chip_air.air, &requester];

    let all_traces = vec![
        pageread_trace.clone(),
        hash_chip_trace.clone(),
        dummy_trace.clone(),
    ];

    let mut expected_anses = vec![BabyBear::zero(); chip.air.hash_width];

    expected_anses = x
        .iter()
        .flat_map(|row| row.chunks(chip.air.hash_rate))
        .fold(expected_anses, |state, chunk| {
            DummyHashAir::hash(state, chunk.to_vec())
        })[..chip.air.digest_width]
        .to_vec();

    let all_pis = vec![expected_anses, vec![], vec![]];

    run_simple_test(all_airs, all_traces, all_pis).expect("Verification failed");
}

#[test]
fn test_single_is_zero_fail() {
    let chip = PageController::new(10, 3, 5, 2, 3, 0, 1);
    let correct_height = chip.air.page_height.next_power_of_two();

    let mut rng = create_seeded_rng();
    let x = (0..chip.air.page_height)
        .map(|_| {
            (0..chip.air.page_width)
                .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..3)))
                .collect()
        })
        .collect::<Vec<Vec<BabyBear>>>();

    let requester = DummyInteractionAir::new(chip.air.page_width, true, chip.air.bus_index);
    let dummy_trace = RowMajorMatrix::new(
        x.iter()
            .flat_map(|row| [BabyBear::one()].into_iter().chain(row.iter().cloned()))
            .chain(
                std::iter::repeat(BabyBear::zero())
                    .take((chip.air.page_width + 1) * (correct_height - chip.air.page_height)),
            )
            .collect(),
        chip.air.page_width + 1,
    );

    let pageread_trace = chip.generate_trace(x.clone());
    let hash_chip = chip.hash_chip.lock();
    let hash_chip_trace = hash_chip.generate_trace();

    let all_airs: Vec<&dyn AnyRap<_>> = vec![&chip.air, &hash_chip.air, &requester];

    let all_traces = vec![
        pageread_trace.clone(),
        hash_chip_trace.clone(),
        dummy_trace.clone(),
    ];

    let mut expected_anses = vec![BabyBear::zero(); chip.air.hash_width];

    expected_anses = x
        .iter()
        .flat_map(|row| row.chunks(chip.air.hash_rate))
        .fold(expected_anses, |state, chunk| {
            DummyHashAir::hash(state, chunk.to_vec())
        })[..chip.air.digest_width]
        .to_vec();

    let all_pis = vec![expected_anses, vec![], vec![]];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    for _ in 0..20 {
        let row_index = rng.gen_range(0..chip.air.page_height);
        let column_index = rng.gen_range(1..chip.air.page_width);
        let mut all_traces_clone = all_traces.clone();
        all_traces_clone[0].row_mut(row_index)[column_index] +=
            BabyBear::from_canonical_u32(rng.gen_range(2..100));
        assert_eq!(
            run_simple_test(all_airs.clone(), all_traces_clone, all_pis.clone()),
            Err(VerificationError::NonZeroCumulativeSum),
            "Expected constraint to fail"
        );
    }
}
