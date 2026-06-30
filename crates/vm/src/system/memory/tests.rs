use std::array;

use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;
use test_case::test_case;

#[cfg(feature = "cuda")]
use crate::arch::testing::{default_var_range_checker_bus, GpuChipTestBuilder};
use crate::{
    arch::{
        testing::{TestBuilder, VmChipTestBuilder},
        MemoryConfig, BLOCK_FE_WIDTH,
    },
    system::memory::merkle::public_values::PUBLIC_VALUES_AS,
};

type F = BabyBear;

fn test_memory_write_by_tester(tester: &mut impl TestBuilder<F>, its: usize) {
    let mut rng = create_seeded_rng();

    // The point here is to have a lot of equal
    // and intersecting/overlapping blocks,
    // by limiting the space of valid pointers.
    let max_ptr = 10;
    let value_bounds = [u16::MAX as u32 + 1; 3];
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
    // Use a small uniform capacity for every AS touched by the randomized helper.
    let mut mem_config = MemoryConfig::default();
    let small_bits = 10;
    let small = 1 << small_bits;
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = small;
    mem_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = small;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = small;
    let mut tester = VmChipTestBuilder::<F>::from_config(mem_config);
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_memory_write_max_address() {
    let mut rng = create_seeded_rng();
    let mem_config = MemoryConfig::default();
    // The default config gives RV64_MEMORY_AS its full 2^32-byte capacity (2^31 u16 cells).
    // Touch the last cell block so the boundary/merkle chips process the top of the address range.
    let last_block = mem_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells - BLOCK_FE_WIDTH;
    let mut tester = VmChipTestBuilder::<F>::from_config(mem_config);
    let values: [F; BLOCK_FE_WIDTH] =
        array::from_fn(|_| F::from_u32(rng.random_range(0..u16::MAX as u32 + 1)));
    tester.write::<BLOCK_FE_WIDTH>(RV64_MEMORY_AS as usize, last_block, values);
    assert_eq!(
        tester.read::<BLOCK_FE_WIDTH>(RV64_MEMORY_AS as usize, last_block),
        values
    );
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(1000)]
#[test_case(0)]
fn test_cuda_memory_write(its: usize) {
    // Keep the targeted GPU test small while preserving a consistent address shape.
    let mut mem_config = MemoryConfig::default();
    let small_bits = 10;
    let small = 1 << small_bits;
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = small;
    mem_config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = small;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = small;
    mem_config.pointer_max_bits = small_bits;
    let mut tester = GpuChipTestBuilder::new(mem_config, default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}
