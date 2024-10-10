use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ark_ff::PrimeField as _;
use ax_sdk::{
    any_rap_box_vec,
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::StarkEngine,
    interaction::dummy_interaction_air::DummyInteractionAir,
    utils::create_seeded_rng,
};
use p3_baby_bear::{
    BabyBear, DiffusionMatrixBabyBear, POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY,
};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::Permutation;
use p3_util::log2_strict_usize;
use rand::{
    distributions::{Distribution, Standard},
    Rng, RngCore, SeedableRng,
};
use rand_xoshiro::Xoroshiro128Plus;
use zkhash::{
    fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::{
        poseidon2::Poseidon2 as HorizenPoseidon2,
        poseidon2_instance_babybear::POSEIDON2_BABYBEAR_16_PARAMS,
    },
};

use super::{HL_BABYBEAR_EXT_CONST_16, HL_BABYBEAR_INT_CONST_16, HL_MDS_MAT_4, MDS_MAT_4};
use crate::poseidon2::Poseidon2Air;

#[test]
fn test_poseidon2_default() {
    // config
    let num_rows = 1 << 4;
    let num_ext_rounds = 8;
    let num_int_rounds = 13;

    // random constants, state generation
    let mut rng = create_seeded_rng();
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();

    // air and trace generation
    let poseidon2_air = Poseidon2Air::<16, BabyBear>::default(); // max constraint degree = 7

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
        HL_BABYBEAR_EXT_CONST_16.to_vec(),
        Poseidon2ExternalMatrixGeneral,
        num_int_rounds,
        HL_BABYBEAR_INT_CONST_16.to_vec(),
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
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3); // max constraint degree = 7 requires log blowup = 3
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    // positive test
    engine
        .run_simple_test_impl(
            any_rap_box_vec![poseidon2_air.clone(), page_requester],
            traces,
            vec![vec![]; 2],
        )
        .expect("Verification failed");

    // negative test
    disable_debug_builder();
    for _ in 0..10 {
        let width = rng.gen_range(0..poseidon2_air.get_width());
        let height = rng.gen_range(0..num_rows);
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut(height)[width] += rand;
        assert_eq!(
            engine
                .run_simple_test_impl(
                    any_rap_box_vec![poseidon2_air.clone(), page_requester],
                    vec![poseidon2_trace.clone(), dummy_trace.clone()],
                    vec![vec![]; 2],
                )
                .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut(height)[width] -= rand;
    }
}

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
        MDS_MAT_4,
        POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY,
        BabyBear::from_wrapped_u64(1u64 << 32).inverse(), // 943718400
        3,
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
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    let engine = engine_from_perm(perm, max_log_degree, fri_params);

    // positive test
    engine
        .run_simple_test_impl(
            any_rap_box_vec![poseidon2_air.clone(), page_requester],
            traces,
            vec![vec![]; 2],
        )
        .expect("Verification failed");

    // negative test
    disable_debug_builder();
    for _ in 0..10 {
        let width = rng.gen_range(0..poseidon2_air.get_width());
        let height = rng.gen_range(0..num_rows);
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut(height)[width] += rand;
        assert_eq!(
            engine
                .run_simple_test_impl(
                    any_rap_box_vec![poseidon2_air.clone(), page_requester],
                    vec![poseidon2_trace.clone(), dummy_trace.clone()],
                    vec![vec![]; 2],
                )
                .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut(height)[width] -= rand;
    }
}

#[test]
fn test_horizen_poseidon2() {
    let horizen_permut = HorizenPoseidon2::new(&POSEIDON2_BABYBEAR_16_PARAMS);
    let mut rng = create_seeded_rng();
    let (external_round_constants, internal_round_constants, horizen_int_diag) =
        Poseidon2Air::<16, BabyBear>::horizen_round_consts_16();
    let mut air_permut = Poseidon2Air::<16, BabyBear>::new(
        external_round_constants,
        internal_round_constants,
        HL_MDS_MAT_4,
        horizen_int_diag,
        BabyBear::one(),
        3,
        0,
    );
    let u32state = (0..16)
        .map(|_| rng.gen_range(1..=1 << 27))
        .collect::<Vec<_>>();
    let horizen_state: Vec<HorizenBabyBear> =
        u32state.into_iter().map(HorizenBabyBear::from).collect();
    let p3_state: [BabyBear; 16] = horizen_state
        .iter()
        .copied()
        .map(Poseidon2Air::<16, BabyBear>::horizen_to_p3)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let air_result: Vec<BabyBear> = air_permut.request_trace(&[p3_state])[0].clone();
    let horizen_result = horizen_permut.permutation(&horizen_state);
    let air_u32_result = air_result
        .iter()
        .map(BabyBear::as_canonical_u32)
        .collect::<Vec<_>>();
    let horizen_u32_result = horizen_result
        .into_iter()
        .map(|elem| elem.into_bigint().0[0] as u32)
        .collect::<Vec<_>>();
    assert_eq!(air_u32_result, horizen_u32_result);
}

#[test]
fn test_poseidon2_air_xoshiro()
where
    Standard: Distribution<BabyBear>,
{
    let mut rng = Xoroshiro128Plus::seed_from_u64(1);

    let external_constants: Vec<[BabyBear; 16]> = (0..8).map(|_| rng.gen()).collect();
    let internal_constants: Vec<BabyBear> = (0..13).map(|_| rng.gen()).collect();

    let mut poseidon2air = Poseidon2Air::<16, BabyBear>::new(
        external_constants.clone(),
        internal_constants.clone(),
        MDS_MAT_4,
        POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY,
        BabyBear::from_wrapped_u64(1u64 << 32).inverse(), // 943718400
        3,
        0,
    );
    let input: [BabyBear; 16] = [
        894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831, 120589055,
        19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920, 1783754913,
    ]
    .map(BabyBear::from_canonical_u32);

    let result = poseidon2air.request_trace(&[input]);

    let expected: [BabyBear; 16] = [
        512585766, 975869435, 1921378527, 1238606951, 899635794, 132650430, 1426417547, 1734425242,
        57415409, 67173027, 1535042492, 1318033394, 1070659233, 17258943, 856719028, 1500534995,
    ]
    .map(BabyBear::from_canonical_u32);

    assert_eq!(result[0], expected)
}
