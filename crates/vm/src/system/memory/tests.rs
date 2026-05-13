use std::array;

use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;
use test_case::test_case;

use crate::arch::{
    testing::{TestBuilder, VmChipTestBuilder},
    MemoryConfig, BLOCK_FE_WIDTH,
};

type F = BabyBear;

fn test_memory_write_by_tester(tester: &mut impl TestBuilder<F>, its: usize) {
    let mut rng = create_seeded_rng();

    // The point here is to have a lot of equal
    // and intersecting/overlapping blocks,
    // by limiting the space of valid pointers.
    let max_ptr = 20;
    let value_bounds = [256, 256, 256];
    for _ in 0..its {
        let addr_sp = rng.random_range(1..=value_bounds.len());
        let value_bound: u32 = value_bounds[addr_sp - 1];
        let ptr = rng.random_range(0..max_ptr / BLOCK_FE_WIDTH) * BLOCK_FE_WIDTH;
        tester.write::<BLOCK_FE_WIDTH>(
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
    // The default PUBLIC_VALUES_AS sizing (`DEFAULT_MAX_NUM_PUBLIC_VALUES /
    // U16_CELL_SIZE = 16` cells) is too small for the random `max_ptr = 20`
    // cell range the helper uses. Bump it so all three writable address
    // spaces fit the test's pointer range.
    let mut mem_config = MemoryConfig::default();
    use crate::system::memory::merkle::public_values::PUBLIC_VALUES_AS;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 10;
    let mut tester = GpuChipTestBuilder::new(mem_config, default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}
