use std::{array, fmt::Debug};

use openvm_instructions::{
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    NATIVE_AS,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use test_case::test_case;

use crate::{
    arch::{
        testing::{TestBuilder, VmChipTestBuilder},
        MemoryConfig,
    },
    system::memory::{merkle::public_values::PUBLIC_VALUES_AS, online::TracingMemory},
};

type F = BabyBear;

fn test_memory_write_by_tester(tester: &mut impl TestBuilder<F>, its: usize) {
    let mut rng = create_seeded_rng();

    // The point here is to have a lot of equal
    // and intersecting/overlapping blocks,
    // by limiting the space of valid pointers.
    let max_ptr = 20;
    let aligns = [4, 4, 4, 1];
    let value_bounds = [256, 256, 256, (1 << 30)];
    let max_log_block_size = 4;
    for _ in 0..its {
        let addr_sp = rng.gen_range(1..=aligns.len());
        let align: usize = aligns[addr_sp - 1];
        let value_bound: u32 = value_bounds[addr_sp - 1];
        let ptr = rng.gen_range(0..max_ptr / align) * align;
        let log_len = rng.gen_range(align.trailing_zeros()..=max_log_block_size);
        match log_len {
            0 => tester.write::<1>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
            ),
            1 => tester.write::<2>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
            ),
            2 => tester.write::<4>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
            ),
            3 => tester.write::<8>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
            ),
            4 => tester.write::<16>(
                addr_sp,
                ptr,
                array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
            ),
            _ => unreachable!(),
        }
    }
}

#[test_case(1000)]
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

fn test_no_adapter_records_for_singleton_accesses<T, const BLOCK_SIZE: usize>(address_space: u32)
where
    T: Copy + Debug,
    Standard: Distribution<T>,
{
    let memory_config = MemoryConfig::default();
    let mut memory = TracingMemory::new(&memory_config, BLOCK_SIZE, 0);
    let max_ptr = (memory_config.addr_spaces[address_space as usize].num_cells / BLOCK_SIZE) as u32;

    let mut rng = thread_rng();
    for _ in 0..1000 {
        let pointer = rng.gen_range(0..max_ptr) * BLOCK_SIZE as u32;

        if rng.gen_bool(0.5) {
            let data: [T; BLOCK_SIZE] = array::from_fn(|_| rng.gen());
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
    test_no_adapter_records_for_singleton_accesses::<u8, 4>(RV32_REGISTER_AS);
    test_no_adapter_records_for_singleton_accesses::<u8, 4>(RV32_MEMORY_AS);
    test_no_adapter_records_for_singleton_accesses::<u8, 4>(PUBLIC_VALUES_AS);
    test_no_adapter_records_for_singleton_accesses::<F, 1>(NATIVE_AS);
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
