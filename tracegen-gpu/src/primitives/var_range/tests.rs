use std::sync::Arc;

use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use rand::Rng;

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
    let random_values: Vec<u32> = (0..NUM_INPUTS)
        .map(|_| tester.rng().gen::<u32>() & RANGE_BIT_MASK)
        .collect();

    let range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        VariableRangeCheckerBus::new(1, RANGE_MAX_BITS),
    ));
    let dummy_chip = DummyInteractionChipGPU::new(range_checker.clone(), random_values);

    tester
        .build()
        .load(dummy_chip)
        .load(range_checker)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
