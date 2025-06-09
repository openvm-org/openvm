use std::sync::Arc;

use crate::{
    dummy::var_range::DummyInteractionChipGPU, primitives::var_range::VariableRangeCheckerChipGPU,
};
use cuda_utils::copy::MemCopyD2H;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_stark_sdk::{
    config::{setup_tracing, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::prover::hal::{DeviceDataTransporter, MatrixDimensions},
};
use p3_field::PrimeField32;
use rand::Rng;
use stark_backend_gpu::{
    engine::GpuBabyBearPoseidon2Engine,
    prover_backend::GpuBackend,
    types::{DeviceAirProofRawInput, DeviceProofInput},
};

const LOG_BLOWUP: usize = 2;
const RANGE_MAX_BITS: usize = 10;
const RANGE_BIT_MASK: u32 = (1 << RANGE_MAX_BITS) - 1;
const NUM_INPUTS: usize = 1 << 16;

#[test]
fn var_range_test() {
    setup_tracing();

    let mut rng = rand::thread_rng();
    let random_values: Vec<u32> = (0..NUM_INPUTS)
        .map(|_| rng.gen::<u32>() & RANGE_BIT_MASK)
        .collect();

    let range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        VariableRangeCheckerBus::new(1, RANGE_MAX_BITS),
    ));
    let dummy_chip = DummyInteractionChipGPU::new(range_checker.clone(), random_values);
    let dummy_trace = dummy_chip.generate_trace();
    let range_checker_trace = range_checker.generate_trace();

    assert_eq!(1 << (RANGE_MAX_BITS + 1), range_checker_trace.height());
    let range_checker_sum: u32 = range_checker_trace
        .to_host()
        .unwrap()
        .iter()
        .map(|x| x.as_canonical_u32())
        .sum();
    assert_eq!(range_checker_sum, NUM_INPUTS as u32);

    let engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let dummy_air_id = keygen_builder.add_air(Arc::new(dummy_chip.air));
    let range_checker_air_id = keygen_builder.add_air(Arc::new(range_checker.air));
    let pk = keygen_builder.generate_pk();

    let proof_input = DeviceProofInput::<GpuBackend> {
        per_air: vec![
            (
                dummy_air_id,
                DeviceAirProofRawInput::<GpuBackend> {
                    cached_mains: vec![],
                    common_main: Some(dummy_trace),
                    public_values: vec![],
                },
            ),
            (
                range_checker_air_id,
                DeviceAirProofRawInput::<GpuBackend> {
                    cached_mains: vec![],
                    common_main: Some(range_checker_trace),
                    public_values: vec![],
                },
            ),
        ],
    };
    let mpk_view = engine
        .device()
        .transport_pk_to_device(&pk, vec![dummy_air_id, range_checker_air_id]);
    let gpu_proof = engine.gpu_prove(mpk_view, proof_input);

    engine.verify(&pk.get_vk(), &gpu_proof).unwrap();
}
