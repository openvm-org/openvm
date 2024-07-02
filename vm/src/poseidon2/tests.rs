use crate::poseidon2::Poseidon2Air;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::config::{
    baby_bear_poseidon2::{engine_from_perm, random_perm},
    fri_params::fri_params_with_80_bits_of_security,
};
use afs_test_utils::engine::StarkEngine;
use afs_test_utils::interaction::dummy_interaction_air::DummyInteractionAir;
use afs_test_utils::utils::create_seeded_rng;
use criterion::black_box;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::Permutation;
use p3_util::log2_strict_usize;
use rand::Rng;
use rand::RngCore;
use zkhash::fields::babybear::FpBabyBear as HorizenBabyBear;
use zkhash::poseidon2::poseidon2::Poseidon2 as HorizenPoseidon2;
use zkhash::poseidon2::poseidon2_instance_babybear::POSEIDON2_BABYBEAR_16_PARAMS;

#[test]
fn test_poseidon2() {
    // config
    let num_rows = 1 << 4;
    let num_ext_rounds = 8;
    let num_int_rounds = 13;

    // random constants, state generation
    let mut rng = create_seeded_rng();
    let external_constants: Vec<[BabyBear; 16]> = (0..num_ext_rounds)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();
    let internal_constants: Vec<BabyBear> = (0..num_int_rounds)
        .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
        .collect();
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();

    // air and trace generation
    let poseidon2_air = Poseidon2Air::<16, BabyBear>::new(
        external_constants.clone(),
        internal_constants.clone(),
        0,
    );
    let mut poseidon2_trace = poseidon2_air.generate_trace(states.clone());
    let mut outputs = states.clone();
    let poseidon2: Poseidon2<
        BabyBear,
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear,
        16,
        7,
    > = Poseidon2::new(
        num_ext_rounds,
        external_constants.clone(),
        Poseidon2ExternalMatrixGeneral,
        num_int_rounds,
        internal_constants.clone(),
        DiffusionMatrixBabyBear,
    );
    for output in outputs.iter_mut() {
        poseidon2.permute_mut(output);
    }

    // dummy interaction air and trace generation
    let page_requester = DummyInteractionAir::new(2 * 16, true, poseidon2_air.bus_index);
    let dummy_trace = RowMajorMatrix::new(
        states
            .into_iter()
            .zip(outputs.iter())
            .flat_map(|(state, output)| {
                [BabyBear::one()]
                    .into_iter()
                    .chain(state.to_vec())
                    .chain(output.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect(),
        2 * 16 + 1,
    );

    let traces = vec![poseidon2_trace.clone(), dummy_trace.clone()];

    // engine generation
    let max_trace_height = traces.iter().map(|trace| trace.height()).max().unwrap();
    let max_log_degree = log2_strict_usize(max_trace_height);
    let perm = random_perm();
    let fri_params = fri_params_with_80_bits_of_security()[1];
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    // positive test
    engine
        .run_simple_test(
            vec![&poseidon2_air, &page_requester],
            traces,
            vec![vec![]; 2],
        )
        .expect("Verification failed");

    // negative test
    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    for _ in 0..10 {
        let width = rng.gen_range(0..poseidon2_air.get_width());
        let height = rng.gen_range(0..num_rows);
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut(height)[width] += rand;
        assert_eq!(
            engine.run_simple_test(
                vec![&poseidon2_air, &page_requester],
                vec![poseidon2_trace.clone(), dummy_trace.clone()],
                vec![vec![]; 2],
            ),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut(height)[width] -= rand;
    }
}

// #[test]
// fn test_horizen_poseidon2() {
//     let mut rng = create_seeded_rng();
//     let instance = HorizenPoseidon2::new(&POSEIDON2_BABYBEAR_16_PARAMS);
//     let u32state = (0..16)
//         .map(|_| rng.gen_range(1..=1 << 27))
//         .collect::<Vec<_>>();
//     let horizen_state: Vec<HorizenBabyBear> =
//         u32state.into_iter().map(HorizenBabyBear::from).collect();
//     let result = instance.permutation(black_box(&horizen_state));
//     println!("{:?}", result);
// }
