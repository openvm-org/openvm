use std::sync::Arc;

use openvm_circuit_primitives::var_range::{
    VariableRangeCheckerAir, VariableRangeCheckerBus, VariableRangeCheckerChip,
};
use openvm_stark_backend::prover::types::AirProvingContext;
use openvm_stark_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_air::BaseAir;
use p3_field::FieldAlgebra;
use rand::Rng;
use stark_backend_gpu::{base::DeviceMatrix, cuda::copy::MemCopyH2D, types::F};

use crate::{
    dummy::var_range::DummyInteractionChipGPU, primitives::var_range::VariableRangeCheckerChipGPU,
    testing::GpuChipTestBuilder,
};

const RANGE_MAX_BITS: usize = 10;
const RANGE_BIT_MASK: u32 = (1 << RANGE_MAX_BITS) - 1;
const NUM_INPUTS: usize = 1 << 16;

#[test]
fn var_range_test() {
    let mut tester = GpuChipTestBuilder::default();
    let bus = VariableRangeCheckerBus::new(1, RANGE_MAX_BITS);
    let random_values: Vec<u32> = (0..NUM_INPUTS)
        .map(|_| tester.rng().gen::<u32>() & RANGE_BIT_MASK)
        .collect();

    let range_checker = Arc::new(VariableRangeCheckerChipGPU::new(bus));
    let dummy_chip = DummyInteractionChipGPU::new(range_checker.clone(), random_values);

    tester
        .build()
        .load_periphery(DummyInteractionAir::new(2, true, bus.index()), dummy_chip)
        .load_periphery(VariableRangeCheckerAir::new(bus), range_checker)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[test]
fn var_range_hybrid_test() {
    let mut tester = GpuChipTestBuilder::default();
    let bus = VariableRangeCheckerBus::new(1, RANGE_MAX_BITS);
    let range_checker = Arc::new(VariableRangeCheckerChipGPU::hybrid(Arc::new(
        VariableRangeCheckerChip::new(bus),
    )));

    let gpu_random_values: Vec<u32> = (0..NUM_INPUTS)
        .map(|_| tester.rng().gen::<u32>() & RANGE_BIT_MASK)
        .collect();
    let gpu_dummy_chip = DummyInteractionChipGPU::new(range_checker.clone(), gpu_random_values);

    let cpu_chip = range_checker.cpu_chip.clone().unwrap();
    let cpu_pairs = (0..NUM_INPUTS)
        .map(|_| {
            let bits = tester.rng().gen_range(0..=(RANGE_MAX_BITS as u32));
            let mask = (1 << bits) - 1;
            let value = tester.rng().gen::<u32>() & mask;
            cpu_chip.add_count(value, bits as usize);
            [value, bits]
        })
        .collect::<Vec<_>>();
    let cpu_dummy_trace = (0..NUM_INPUTS)
        .map(|_| F::ONE)
        .chain(
            cpu_pairs
                .iter()
                .map(|pair| F::from_canonical_u32(pair[0]))
                .chain(cpu_pairs.iter().map(|pair| F::from_canonical_u32(pair[1]))),
        )
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let dummy_air = DummyInteractionAir::new(2, true, bus.index());
    let cpu_proving_ctx = AirProvingContext {
        cached_mains: vec![],
        common_main: Some(DeviceMatrix::new(
            Arc::new(cpu_dummy_trace),
            NUM_INPUTS,
            BaseAir::<F>::width(&dummy_air),
        )),
        public_values: vec![],
    };

    tester
        .build()
        .load_air_proving_ctx(Arc::new(dummy_air), cpu_proving_ctx)
        .load_periphery(dummy_air, gpu_dummy_chip)
        .load_periphery(VariableRangeCheckerAir::new(bus), range_checker)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
