use std::array;

<<<<<<< HEAD
use openvm_instructions::{
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    DEFERRAL_AS,
};
=======
>>>>>>> 6806403bd (style: remove access adapters)
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

<<<<<<< HEAD
fn memory_config_for_test() -> MemoryConfig {
    let mut memory_config = MemoryConfig::default();
    memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 29;
    memory_config
}

#[test_case(1000)]
=======
>>>>>>> 6806403bd (style: remove access adapters)
#[test_case(0)]
fn test_memory_write_volatile(its: usize) {
    let mut tester = VmChipTestBuilder::<F>::volatile(memory_config_for_test());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(1000)]
#[test_case(0)]
fn test_memory_write_persistent(its: usize) {
    let mut tester = VmChipTestBuilder::<F>::persistent(memory_config_for_test());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

<<<<<<< HEAD
fn test_no_adapter_records_for_singleton_accesses<T: Copy + Debug, const BLOCK_SIZE: usize>(
    address_space: u32,
    mut sample: impl FnMut(&mut StdRng) -> T,
) {
    let memory_config = memory_config_for_test();
    let mut memory = TracingMemory::new(&memory_config, BLOCK_SIZE, 0);
    let max_ptr = (memory_config.addr_spaces[address_space as usize].num_cells / BLOCK_SIZE) as u32;

    let mut rng = create_seeded_rng();
    for _ in 0..1000 {
        let pointer = rng.random_range(0..max_ptr) * BLOCK_SIZE as u32;

        if rng.random_bool(0.5) {
            let data: [T; BLOCK_SIZE] = array::from_fn(|_| sample(&mut rng));
            unsafe {
                memory.write::<T, BLOCK_SIZE, BLOCK_SIZE>(address_space, pointer, data);
            }
        } else {
            unsafe {
                memory.read::<T, BLOCK_SIZE, BLOCK_SIZE>(address_space, pointer);
            }
        }
    }
    assert!(memory.access_adapter_records.allocated().is_empty());
}

#[test]
fn test_no_adapter_records() {
    test_no_adapter_records_for_singleton_accesses::<u8, 4>(RV32_REGISTER_AS, |rng| rng.random());
    test_no_adapter_records_for_singleton_accesses::<u8, 4>(RV32_MEMORY_AS, |rng| rng.random());
    test_no_adapter_records_for_singleton_accesses::<u8, 4>(PUBLIC_VALUES_AS, |rng| rng.random());
    test_no_adapter_records_for_singleton_accesses::<F, 1>(DEFERRAL_AS, |rng| {
        F::from_u32(rng.random_range(0..(1 << 30)))
    });
}

=======
>>>>>>> 6806403bd (style: remove access adapters)
#[cfg(feature = "cuda")]
#[test_case(1000)]
#[test_case(0)]
fn test_cuda_memory_write_volatile(its: usize) {
    use crate::arch::testing::{default_var_range_checker_bus, GpuChipTestBuilder};
    let memory_config = memory_config_for_test();
    let mut tester = GpuChipTestBuilder::volatile(memory_config, default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(1000)]
#[test_case(0)]
fn test_cuda_memory_write_persistent(its: usize) {
    use crate::arch::testing::{default_var_range_checker_bus, GpuChipTestBuilder};
    let memory_config = memory_config_for_test();
    let mut tester = GpuChipTestBuilder::persistent(memory_config, default_var_range_checker_bus());
    test_memory_write_by_tester(&mut tester, its);
    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}
