use std::{array::from_fn, sync::Arc};

use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::FieldAlgebra, utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use p3_poseidon2::ExternalLayerConstants;
use rand::{rngs::StdRng, Rng, RngCore};
#[cfg(feature = "cuda")]
use {
    crate::cuda_abi::poseidon2,
    openvm_cuda_backend::{
        base::DeviceMatrix, data_transporter::assert_eq_host_and_device_matrix, types::F,
    },
    openvm_cuda_common::copy::MemCopyH2D as _,
    openvm_stark_backend::p3_field::PrimeField32,
};

use super::{Poseidon2Config, Poseidon2Constants, Poseidon2SubChip};
use crate::BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS;

fn run_poseidon2_subchip_test(subchip: Arc<Poseidon2SubChip<BabyBear, 0>>, rng: &mut StdRng) {
    // random state and trace generation
    let num_rows = 1 << 4;
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();
    let mut poseidon2_trace = subchip.generate_trace(states.clone());

    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3); // max constraint degree = 7 requires log blowup = 3
    let engine = BabyBearPoseidon2Engine::new(fri_params);

    // positive test
    engine
        .run_simple_test_impl(
            vec![subchip.air.clone()],
            vec![poseidon2_trace.clone()],
            vec![vec![]],
        )
        .expect("Verification failed");

    // negative test
    disable_debug_builder();
    for _ in 0..10 {
        let rand_idx = rng.gen_range(0..subchip.air.width());
        let rand_inc = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut((1 << 4) - 1)[rand_idx] += rand_inc;
        assert_eq!(
            engine
                .run_simple_test_impl(
                    vec![subchip.air.clone()],
                    vec![poseidon2_trace.clone()],
                    vec![vec![]],
                )
                .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut((1 << 4) - 1)[rand_idx] -= rand_inc;
    }
}

#[test]
fn test_poseidon2_default() {
    let mut rng = create_seeded_rng();
    let poseidon2_config = Poseidon2Config::default();
    let poseidon2_subchip = Arc::new(Poseidon2SubChip::<BabyBear, 0>::new(
        poseidon2_config.constants,
    ));
    run_poseidon2_subchip_test(poseidon2_subchip, &mut rng);
}

#[test]
fn test_poseidon2_random_constants() {
    let mut rng = create_seeded_rng();
    let external_constants =
        ExternalLayerConstants::new_from_rng(2 * BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, &mut rng);
    let beginning_full_round_constants_vec = external_constants.get_initial_constants();
    let beginning_full_round_constants = from_fn(|i| beginning_full_round_constants_vec[i]);
    let ending_full_round_constants_vec = external_constants.get_terminal_constants();
    let ending_full_round_constants = from_fn(|i| ending_full_round_constants_vec[i]);
    let partial_round_constants = from_fn(|_| BabyBear::from_wrapped_u32(rng.next_u32()));
    let constants = Poseidon2Constants {
        beginning_full_round_constants,
        partial_round_constants,
        ending_full_round_constants,
    };
    let poseidon2_subchip = Arc::new(Poseidon2SubChip::<BabyBear, 0>::new(constants));
    run_poseidon2_subchip_test(poseidon2_subchip, &mut rng);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_tracegen_poseidon2() {
    const WIDTH: usize = 16; // constant for BabyBear
    const N: usize = 16;
    const SBOX_REGS: usize = 1;
    const HALF_FULL_ROUNDS: usize = 4; // Constant for BabyBear
    const PARTIAL_ROUNDS: usize = 13; // Constant for BabyBear

    // Generate random states and prepare GPU inputs
    let mut rng = create_seeded_rng();
    let cpu_inputs: Vec<[F; WIDTH]> = (0..N)
        .map(|_| std::array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32))))
        .collect();

    // Flatten inputs in row-major order for GPU (same layout as cpu_inputs)
    let inputs_dev = cpu_inputs
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    // Launch GPU tracegen
    let num_cols = 1
        + WIDTH
        + HALF_FULL_ROUNDS * (WIDTH * SBOX_REGS + WIDTH)
        + PARTIAL_ROUNDS * (SBOX_REGS + 1)
        + HALF_FULL_ROUNDS * (WIDTH * SBOX_REGS + WIDTH);

    let gpu_mat = DeviceMatrix::<F>::with_capacity(N, num_cols);

    unsafe {
        poseidon2::dummy_tracegen(gpu_mat.buffer(), &inputs_dev, SBOX_REGS as u32, N as u32)
            .expect("GPU tracegen failed");
    }

    // Run CPU tracegen and compare results
    let config = Poseidon2Config::<BabyBear>::default();
    let chip: Poseidon2SubChip<_, SBOX_REGS> = Poseidon2SubChip::new(config.constants);
    let cpu_trace = Arc::new(chip.generate_trace(cpu_inputs));
    assert_eq_host_and_device_matrix(cpu_trace, &gpu_mat);
}
