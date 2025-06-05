use std::sync::Arc;
use core::array;

// use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus;
use openvm_stark_sdk::{
    config::{setup_tracing, FriParameters}, engine::{StarkEngine, StarkFriEngine}, openvm_stark_backend::prover::hal::{DeviceDataTransporter},
};
use rand::Rng;
use stark_backend_gpu::{
    engine::GpuBabyBearPoseidon2Engine,
    prover_backend::GpuBackend,
    types::{DeviceAirProofRawInput, DeviceProofInput},
};
use tracegen_gpu::{primitives::range_tuple::RangeTupleCheckerChipGPU, dummy::range_tuple::DummyInteractionChipGPU};

const LOG_BLOWUP: usize = 2;
const LIST_LEN: usize = 16;

#[test]
fn range_tuple_test() {
    setup_tracing();

    let mut rng = rand::thread_rng();

    let bus_index  = 0;
    let sizes : [u32; 3] = array::from_fn(|_| 1 << rng.gen_range(1..5));

    let mut gen_tuple = || {
        sizes
            .iter()
            .map(|&size| rng.gen_range(0..size))
            .collect::<Vec<_>>()
    };

    let vals = (0..LIST_LEN).map(|_| gen_tuple()).collect::<Vec<_>>();

    let bus = RangeTupleCheckerBus::new(bus_index, sizes);
    let range_tuple_checker = Arc::new(RangeTupleCheckerChipGPU::new(bus));

    let dummy_chip = DummyInteractionChipGPU::new(range_tuple_checker.clone(), vals.concat().to_vec());
    let dummy_trace = dummy_chip.generate_trace();
    let range_tuple_trace = range_tuple_checker.generate_trace();

    let engine = GpuBabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );

    let mut keygen_builder = engine.keygen_builder();
    let dummy_air_id = keygen_builder.add_air(Arc::new(dummy_chip.air));
    let range_tuple_checker_id = keygen_builder.add_air(Arc::new(range_tuple_checker.air));
    let pk = keygen_builder.generate_pk();

    let proof_input = DeviceProofInput::<GpuBackend> {
        per_air: vec![
            (dummy_air_id, DeviceAirProofRawInput::<GpuBackend> {
                cached_mains: vec![],
                common_main: Some(dummy_trace),
                public_values: vec![],
            }),
            (range_tuple_checker_id, DeviceAirProofRawInput::<GpuBackend> {
                cached_mains: vec![],
                common_main: Some(range_tuple_trace),
                public_values: vec![],
            }),
        ],
    };

    let mpk_view = engine
        .device()
        .transport_pk_to_device(&pk, vec![dummy_air_id, range_tuple_checker_id]);
    let gpu_proof = engine.gpu_prove(mpk_view, proof_input);

    engine.verify(&pk.get_vk(), &gpu_proof).unwrap();

}
