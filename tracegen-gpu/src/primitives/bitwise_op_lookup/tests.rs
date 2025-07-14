use std::sync::Arc;

use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_air::BaseAir;
use p3_field::FieldAlgebra;
use rand::Rng;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::copy::MemCopyH2D,
    types::{DeviceAirProofRawInput, F},
};

use crate::{
    dummy::bitwise_op_lookup::DummyInteractionChipGPU,
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU, testing::GpuChipTestBuilder,
};

const NUM_BITS: usize = RV32_CELL_BITS;
const BIT_MASK: u32 = (1 << NUM_BITS) - 1;
const NUM_INPUTS: usize = 1 << 16;

#[test]
fn bitwise_op_lookup_test() {
    let mut tester = GpuChipTestBuilder::default();
    let bitwise = Arc::new(BitwiseOperationLookupChipGPU::<RV32_CELL_BITS>::new(
        BitwiseOperationLookupBus::new(0),
    ));

    let random_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            let x = tester.rng().gen::<u32>() & BIT_MASK;
            let y = tester.rng().gen::<u32>() & BIT_MASK;
            let op = tester.rng().gen_bool(0.5);
            [x, y, op as u32]
        })
        .collect::<Vec<_>>();
    let dummy_chip = DummyInteractionChipGPU::new(bitwise.clone(), random_values);

    tester
        .build()
        .load(dummy_chip)
        .load(bitwise)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[test]
fn bitwise_op_lookup_hybrid_test() {
    let mut tester = GpuChipTestBuilder::default();
    let bus = BitwiseOperationLookupBus::new(0);
    let bitwise = Arc::new(BitwiseOperationLookupChipGPU::<RV32_CELL_BITS>::hybrid(
        Arc::new(BitwiseOperationLookupChip::new(bus)),
    ));

    let gpu_random_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            let x = tester.rng().gen::<u32>() & BIT_MASK;
            let y = tester.rng().gen::<u32>() & BIT_MASK;
            let op = tester.rng().gen_bool(0.5);
            [x, y, op as u32]
        })
        .collect::<Vec<_>>();
    let gpu_dummy_chip = DummyInteractionChipGPU::new(bitwise.clone(), gpu_random_values);

    let cpu_chip = bitwise.cpu_chip.clone().unwrap();
    let cpu_values = (0..NUM_INPUTS)
        .map(|_| {
            let x = tester.rng().gen::<u32>() & BIT_MASK;
            let y = tester.rng().gen::<u32>() & BIT_MASK;
            let op_xor = tester.rng().gen_bool(0.5);
            let z = if op_xor {
                cpu_chip.request_xor(x, y)
            } else {
                cpu_chip.request_range(x, y);
                0
            };
            [x, y, z, op_xor as u32]
        })
        .collect::<Vec<_>>();
    let cpu_dummy_trace = (0..NUM_INPUTS)
        .map(|_| F::ONE)
        .chain(
            cpu_values
                .iter()
                .map(|v| F::from_canonical_u32(v[0]))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[1])))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[2])))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[3]))),
        )
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();
    let cpu_air = DummyInteractionAir::new(4, true, bus.inner.index);

    let mut tester = tester.build();
    tester.airs.push(Arc::new(cpu_air));
    tester.raw_inputs.push(DeviceAirProofRawInput {
        cached_mains: vec![],
        common_main: Some(DeviceMatrix::new(
            Arc::new(cpu_dummy_trace),
            NUM_INPUTS,
            BaseAir::<F>::width(&cpu_air),
        )),
        public_values: vec![],
    });

    tester
        .load(gpu_dummy_chip)
        .load(bitwise)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
