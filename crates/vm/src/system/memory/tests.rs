use std::array;

use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;
use test_case::test_case;

use crate::arch::{
    testing::{TestBuilder, VmChipTestBuilder},
    MemoryConfig, CONST_BLOCK_SIZE,
};

type F = BabyBear;

fn test_memory_write_by_tester(tester: &mut impl TestBuilder<F>, its: usize) {
    let mut rng = create_seeded_rng();

    // Have a lot of equal and intersecting/overlapping blocks,
    // by limiting the space of valid pointers.
    let max_ptr = 20;
    let value_bound: u32 = 256;
    for _ in 0..its {
        let addr_sp = rng.random_range(1..=3);
        let ptr = rng.random_range(0..max_ptr / CONST_BLOCK_SIZE) * CONST_BLOCK_SIZE;
        tester.write::<CONST_BLOCK_SIZE>(
            addr_sp,
            ptr,
            array::from_fn(|_| F::from_u32(rng.random_range(0..value_bound))),
        );
    }
}

#[test_case(1000)]
#[test_case(0)]
fn test_memory_write(its: usize) {
    let mut tester = VmChipTestBuilder::<F>::from_config(MemoryConfig::default());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(1000)]
#[test_case(0)]
fn test_cuda_memory_write(its: usize) {
    use crate::arch::testing::{default_var_range_checker_bus, GpuChipTestBuilder};
    let mut tester =
        GpuChipTestBuilder::new(MemoryConfig::default(), default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}
