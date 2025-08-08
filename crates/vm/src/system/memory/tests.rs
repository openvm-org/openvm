use std::array;

use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{thread_rng, Rng};

use crate::{
    arch::{testing::VmChipTestBuilder, MemoryConfig},
    system::memory::online::TracingMemory,
};

type F = BabyBear;

fn test_memory_write_by_tester(mut tester: VmChipTestBuilder<F>) {
    let mut rng = create_seeded_rng();

    // The point here is to have a lot of equal
    // and intersecting/overlapping blocks,
    // by limiting the space of valid pointers.
    let max_ptr = 20;
    let aligns = [4, 4, 4, 1];
    let value_bounds = [256, 256, 256, (1 << 30)];
    let max_log_block_size = 4;
    let its = 1000;
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

    let tester = tester.build().finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_memory_write_volatile() {
    test_memory_write_by_tester(VmChipTestBuilder::<F>::volatile(MemoryConfig::default()));
}

#[test]
fn test_memory_write_persistent() {
    test_memory_write_by_tester(VmChipTestBuilder::<F>::persistent(MemoryConfig::default()));
}

#[test]
fn test_no_adapter_records_for_singleton_accesses() {
    let memory_config = MemoryConfig::default();
    let mut memory = TracingMemory::new(&memory_config, 1, 0);

    let mut rng = thread_rng();
    for _ in 0..1000 {
        // TODO[jpw]: test other address spaces?
        let address_space = 4u32;
        let pointer = rng.gen_range(0..1 << memory_config.pointer_max_bits);

        if rng.gen_bool(0.5) {
            let data = F::from_canonical_u32(rng.gen_range(0..1 << 30));
            // address space is 4 so cell type is `F`
            unsafe {
                memory.write::<F, 1, 1>(address_space, pointer, [data]);
            }
        } else {
            unsafe {
                memory.read::<F, 1, 1>(address_space, pointer);
            }
        }
    }
    assert!(memory.access_adapter_records.allocated().is_empty());
}
