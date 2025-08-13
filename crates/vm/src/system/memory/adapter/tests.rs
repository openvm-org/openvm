use std::array;

use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use rand::Rng;

use crate::{arch::testing::VmChipTestBuilder, system::memory::adapter::AccessAdapterInventory};

type F = BabyBear;

#[test]
fn test_access_adapters_cpu_gpu_equivalence() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

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

    let touched = tester.memory.memory.finalize(false);
    let mut access_adapter_inv =
        AccessAdapterInventory::new(tester.range_checker(), tester.memory_bus(), tester);
    let allocated = tester.memory.memory.access_adapter_records.allocated_mut();
    let gpu_traces = access_adapter_inv
        .generate_traces_from_records(allocated)
        .into_iter()
        .map(|trace| trace.unwrap_or_else(DeviceMatrix::dummy))
        .collect::<Vec<_>>();

    let mut controller = MemoryController::with_volatile_memory(
        MemoryBus::new(MEMORY_BUS),
        mem_config,
        tester.cpu_range_checker(),
    );
    let all_memory_traces = controller
        .generate_proving_ctx::<SC>(tester.memory.memory.access_adapter_records, touched)
        .into_iter()
        .map(|ctx| ctx.common_main.unwrap())
        .collect::<Vec<_>>();
    let num_memory_traces = all_memory_traces.len();
    let cpu_traces: Vec<_> = all_memory_traces
        .into_iter()
        .skip(num_memory_traces - NUM_ADAPTERS)
        .collect::<Vec<_>>();

    for (cpu_trace, gpu_trace) in cpu_traces.into_iter().zip(gpu_traces.iter()) {
        assert_eq!(
            cpu_trace.height() == 0,
            gpu_trace.height() == 0,
            "Exactly one of CPU and GPU traces is empty"
        );
        if cpu_trace.height() != 0 {
            assert_eq_cpu_and_gpu_matrix(cpu_trace, gpu_trace);
        }
    }
}
