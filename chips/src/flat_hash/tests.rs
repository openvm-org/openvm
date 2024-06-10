use crate::flat_hash::FlatHashChip;

use afs_stark_backend::prover::USE_DEBUG_BUILDER;
use afs_stark_backend::rap::AnyRap;
use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::{config::baby_bear_poseidon2::run_simple_test, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use crate::dummy_hash::DummyHashChip;

#[test]
fn test_single_is_zero() {
    let chip = FlatHashChip::new(8, 4, 5, 2, 3, 0, 1);
    let num_hashchips = chip.page_width / chip.hash_rate;

    let mut rng = create_seeded_rng();
    let x = (0..chip.page_height)
        .map(|_| {
            (0..chip.page_width)
                .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..100)))
                .collect()
        })
        .collect::<Vec<Vec<BabyBear>>>();

    let mut hash_chip_states = vec![];
    let mut hash_chip_slices = vec![];

    let mut state = vec![BabyBear::zero(); chip.hash_width];

    for row in &x {
        for width in 0..num_hashchips {
            hash_chip_states.push(state.clone());
            let slice = row[width * chip.hash_rate..(width + 1) * chip.hash_rate].to_vec();
            hash_chip_slices.push(slice.clone());
            state = DummyHashChip::request(&chip.hash_chip, state, slice);
        }
    }

    let hash_chip_trace = chip
        .hash_chip
        .generate_trace(hash_chip_states.clone(), hash_chip_slices.clone());

    let trace = chip.generate_trace(x);
    let all_chips: Vec<&dyn AnyRap<_>> = vec![&chip, &chip.hash_chip];

    let all_traces = vec![trace, hash_chip_trace];

    let pis = state
        .iter()
        .take(chip.digest_width)
        .cloned()
        .collect::<Vec<_>>();
    let all_pis = vec![pis, vec![]];

    run_simple_test(all_chips, all_traces, all_pis).expect("Verification failed");
}

// #[test]
// fn test_single_is_zero_fail() {
//     let chip = FlatHashChip::new(10, 4, 5, 2, 3, 0, 1);
//     let num_hashchips = chip.page_width / chip.hash_rate;

//     let hash_chips = (0..num_hashchips)
//         .map(|_| DummyHashChip::new(0, chip.hash_rate, chip.hash_width))
//         .collect::<Vec<_>>();

//     let mut rng = create_seeded_rng();
//     let x = (0..chip.page_height)
//         .map(|_| {
//             (0..chip.page_width)
//                 .map(|_| BabyBear::from_canonical_u32(rng.gen_range(0..100)))
//                 .collect()
//         })
//         .collect::<Vec<Vec<BabyBear>>>();

//     let mut hash_chip_states = vec![vec![]; num_hashchips];
//     let mut hash_chip_slices = vec![vec![]; num_hashchips];

//     let mut state = vec![BabyBear::zero(); chip.hash_width];

//     for height in 0..chip.page_height {
//         for width in 0..num_hashchips {
//             hash_chip_states[width].push(state.clone());
//             let slice = x[height][width * chip.hash_rate..(width + 1) * chip.hash_rate].to_vec();
//             hash_chip_slices[width].push(slice.clone());
//             state = DummyHashChip::request(&hash_chips[width], state, slice);
//         }
//     }

//     let hash_chip_traces = hash_chips
//         .iter()
//         .enumerate()
//         .map(|(i, hash_chip)| {
//             hash_chip.generate_trace(hash_chip_states[i].clone(), hash_chip_slices[i].clone())
//         })
//         .collect::<Vec<_>>();

//     let trace = chip.generate_trace(x);
//     // let hash_trace = chip.hash_chip.generate_trace(x);
//     let mut all_chips: Vec<&dyn AnyRap<_>> = vec![&chip];
//     all_chips.extend(hash_chips.iter().map(|hc| hc as &dyn AnyRap<_>));

//     let mut all_traces = vec![trace];
//     all_traces.extend(hash_chip_traces);

//     let pis = state
//         .iter()
//         .take(chip.digest_width)
//         .cloned()
//         .collect::<Vec<_>>();
//     let mut all_pis = vec![pis];
//     all_pis.extend(vec![vec![]; num_hashchips]);

//     // assert_eq!(trace.values[1], AbstractField::from_canonical_u32(0));

//     run_simple_test(all_chips, all_traces, all_pis).expect("Verification failed");

//     let mut trace = chip.generate_trace(x);
//     trace.values[1] = AbstractField::from_canonical_u32(1);

//     // assert_eq!(trace.values[1], AbstractField::from_canonical_u32(0));

//     USE_DEBUG_BUILDER.with(|debug| {
//         *debug.lock().unwrap() = false;
//     });
//     assert_eq!(
//         run_simple_test_no_pis(vec![&chip], vec![trace]),
//         Err(VerificationError::NonZeroCumulativeSum),
//         "Expected constraint to fail"
//     );
// }
