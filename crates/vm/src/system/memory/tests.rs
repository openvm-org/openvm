use std::array;

use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;
use test_case::test_case;

use crate::arch::{
    testing::{TestBuilder, VmChipTestBuilder},
    MemoryConfig,
};

type F = BabyBear;

fn test_memory_write_by_tester(tester: &mut impl TestBuilder<F>, its: usize) {
    let mut rng = create_seeded_rng();

    // The point here is to have a lot of equal
    // and intersecting/overlapping blocks,
    // by limiting the space of valid pointers.
    let max_ptr = 20;
    let aligns = [4, 4, 4];
    let value_bounds = [256, 256, 256];
    for _ in 0..its {
        let addr_sp = rng.random_range(1..=aligns.len());
        let align: usize = aligns[addr_sp - 1];
        let value_bound: u32 = value_bounds[addr_sp - 1];
        let ptr = rng.random_range(0..max_ptr / align) * align;
        // Access adapters are removed, so accesses use the address space minimum block size.
        let log_len = align.trailing_zeros();
        match log_len {
            0 => tester.write::<1>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_u32(rng.random_range(0..value_bound))),
            ),
            1 => tester.write::<2>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_u32(rng.random_range(0..value_bound))),
            ),
            2 => tester.write::<4>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_u32(rng.random_range(0..value_bound))),
            ),
            3 => tester.write::<8>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_u32(rng.random_range(0..value_bound))),
            ),
            4 => tester.write::<16>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_u32(rng.random_range(0..value_bound))),
            ),
            _ => unreachable!(),
        }
    }
}

#[test_case(0)]
fn test_memory_write_volatile(its: usize) {
    let mut tester = VmChipTestBuilder::<F>::volatile(MemoryConfig::default());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(1000)]
#[test_case(0)]
fn test_memory_write_persistent(its: usize) {
    let mut tester = VmChipTestBuilder::<F>::persistent(MemoryConfig::default());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(1000)]
#[test_case(0)]
fn test_cuda_memory_write_volatile(its: usize) {
    use crate::arch::testing::{default_var_range_checker_bus, GpuChipTestBuilder};
    let mut tester =
        GpuChipTestBuilder::volatile(MemoryConfig::default(), default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(1000)]
#[test_case(0)]
fn test_cuda_memory_write_persistent(its: usize) {
    use crate::arch::testing::{default_var_range_checker_bus, GpuChipTestBuilder};
    let mut tester =
        GpuChipTestBuilder::persistent(MemoryConfig::default(), default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}
