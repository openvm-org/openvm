use core::array::from_fn;
use std::sync::Arc;

use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
};
use openvm_stark_backend::prover::types::AirProvingContext;
use openvm_stark_sdk::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir;
use p3_air::BaseAir;
use p3_field::FieldAlgebra;
use rand::Rng;
use stark_backend_gpu::{base::DeviceMatrix, cuda::copy::MemCopyH2D, types::F};

use crate::{
    dummy::range_tuple::DummyInteractionChipGPU, primitives::range_tuple::RangeTupleCheckerChipGPU,
    testing::GpuChipTestBuilder,
};

const TUPLE_SIZE: usize = 3;
const NUM_INPUTS: usize = 1 << 16;

#[test]
fn range_tuple_test() {
    let mut tester = GpuChipTestBuilder::default();

    let sizes: [u32; TUPLE_SIZE] = from_fn(|_| 1 << tester.rng().gen_range(1..5));
    let bus = RangeTupleCheckerBus::<TUPLE_SIZE>::new(0, sizes);
    let random_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            sizes
                .iter()
                .map(|&size| tester.rng().gen_range(0..size))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let range_tuple_checker = Arc::new(RangeTupleCheckerChipGPU::new(bus.sizes));
    let dummy_chip = DummyInteractionChipGPU::new(range_tuple_checker.clone(), random_values);

    tester
        .build()
        .load_periphery(
            DummyInteractionAir::new(TUPLE_SIZE, true, bus.inner.index),
            dummy_chip,
        )
        .load_periphery(
            RangeTupleCheckerAir::<TUPLE_SIZE> { bus },
            range_tuple_checker,
        )
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[test]
fn range_tuple_hybrid_test() {
    let mut tester = GpuChipTestBuilder::default();
    let sizes: [u32; TUPLE_SIZE] = from_fn(|_| 1 << tester.rng().gen_range(1..5));
    let bus = RangeTupleCheckerBus::<TUPLE_SIZE>::new(0, sizes);
    let range_tuple_checker = Arc::new(RangeTupleCheckerChipGPU::hybrid(Arc::new(
        RangeTupleCheckerChip::new(bus),
    )));

    let gpu_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            sizes
                .iter()
                .map(|&size| tester.rng().gen_range(0..size))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let gpu_dummy_chip = DummyInteractionChipGPU::new(range_tuple_checker.clone(), gpu_values);

    let cpu_chip = range_tuple_checker.cpu_chip.clone().unwrap();
    let cpu_values = (0..NUM_INPUTS)
        .map(|_| {
            let values = sizes
                .iter()
                .map(|&size| tester.rng().gen_range(0..size))
                .collect::<Vec<_>>();
            cpu_chip.add_count(&values);
            values
        })
        .collect::<Vec<_>>();
    let cpu_dummy_trace = (0..NUM_INPUTS)
        .map(|_| F::ONE)
        .chain(
            cpu_values
                .iter()
                .map(|v| F::from_canonical_u32(v[0]))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[1])))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[2]))),
        )
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let dummy_air = DummyInteractionAir::new(TUPLE_SIZE, true, bus.inner.index);
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
        .load_periphery(
            RangeTupleCheckerAir::<TUPLE_SIZE> { bus },
            range_tuple_checker,
        )
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
