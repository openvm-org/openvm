use core::array::from_fn;
use std::sync::Arc;

use openvm_circuit_primitives::range_tuple::RangeTupleCheckerBus;
use rand::Rng;

use crate::{
    dummy::range_tuple::DummyInteractionChipGPU, primitives::range_tuple::RangeTupleCheckerChipGPU,
    testing::GpuChipTestBuilder,
};

#[test]
fn range_tuple_test() {
    let mut tester = GpuChipTestBuilder::default();

    let sizes: [u32; 3] = from_fn(|_| 1 << tester.rng().gen_range(1..5));
    let random_values = (0..64)
        .flat_map(|_| {
            sizes
                .iter()
                .map(|&size| tester.rng().gen_range(0..size))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let range_tuple_checker = Arc::new(RangeTupleCheckerChipGPU::new(
        RangeTupleCheckerBus::<3>::new(0, sizes),
    ));
    let dummy_chip = DummyInteractionChipGPU::new(range_tuple_checker.clone(), random_values);

    tester
        .build()
        .load(dummy_chip)
        .load(range_tuple_checker)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
