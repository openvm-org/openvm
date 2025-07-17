use std::sync::Arc;

use openvm_circuit::{
    arch::{CustomBorrow, DenseRecordArena, SizedRecord},
    system::memory::adapter::{
        records::{AccessLayout, AccessRecordMut},
        AccessAdapterCols,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::prover::types::AirProvingContext;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prover_backend::GpuBackend, types::F,
};

use crate::{
    primitives::var_range::VariableRangeCheckerChipGPU, system::cuda::access_adapters::tracegen,
};

pub struct AccessAdapterChipGPU<const N: usize>;

pub(crate) const NUM_ADAPTERS: usize = 5;

pub enum GenericAccessAdapterChipGPU {
    N2(AccessAdapterChipGPU<2>),
    N4(AccessAdapterChipGPU<4>),
    N8(AccessAdapterChipGPU<8>),
    N16(AccessAdapterChipGPU<16>),
    N32(AccessAdapterChipGPU<32>),
}

pub struct AccessAdapterInventoryGPU {
    _chips: [GenericAccessAdapterChipGPU; NUM_ADAPTERS],
    range_checker: Arc<VariableRangeCheckerChipGPU>,
}

#[repr(C)]
pub struct OffsetInfo {
    pub record_offset: u32,
    pub adapter_rows: [u32; NUM_ADAPTERS],
}

pub(crate) fn generate_traces_from_records(
    records: &[u8],
    range_checker: Arc<VariableRangeCheckerChipGPU>,
) -> [DeviceMatrix<F>; NUM_ADAPTERS] {
    // TODO: Temporary hack to get mut access to `records`, should have `self` or `&mut self` as a parameter
    // **SAFETY**: `records` should be non-empty at this point
    let records =
        unsafe { std::slice::from_raw_parts_mut(records.as_ptr() as *mut u8, records.len()) };

    let mut offsets = Vec::new();
    let mut offset = 0;
    let mut row_ids = [0; NUM_ADAPTERS];

    while offset < records.len() {
        offsets.push(OffsetInfo {
            record_offset: offset as u32,
            adapter_rows: row_ids,
        });
        let layout: AccessLayout = unsafe { records[offset..].extract_layout() };
        let record: AccessRecordMut<'_> = records[offset..].custom_borrow(layout.clone());
        offset += <AccessRecordMut<'_> as SizedRecord<AccessLayout>>::size(&layout);
        let bs = record.header.block_size;
        let lbs = record.header.lowest_block_size;
        for logn in lbs.ilog2()..bs.ilog2() {
            row_ids[logn as usize] += bs >> (1 + logn);
        }
    }

    let d_records = records.to_device().unwrap();
    let d_record_offsets = offsets.to_device().unwrap();
    let widths: [_; NUM_ADAPTERS] = std::array::from_fn(|i| match i {
        0 => size_of::<AccessAdapterCols<u8, 2>>(),
        1 => size_of::<AccessAdapterCols<u8, 4>>(),
        2 => size_of::<AccessAdapterCols<u8, 8>>(),
        3 => size_of::<AccessAdapterCols<u8, 16>>(),
        4 => size_of::<AccessAdapterCols<u8, 32>>(),
        _ => panic!(),
    });
    let unpadded_heights: [_; NUM_ADAPTERS] = std::array::from_fn(|i| row_ids[i] as usize);
    let traces = std::array::from_fn(|i| match unpadded_heights[i] {
        0 => DeviceMatrix::<F>::dummy(),
        h => DeviceMatrix::<F>::with_capacity(next_power_of_two_or_zero(h), widths[i]),
    });
    let trace_ptrs: [_; NUM_ADAPTERS] =
        std::array::from_fn(|i| traces[i].buffer().as_mut_raw_ptr());
    let d_trace_ptrs = trace_ptrs.to_device().unwrap();
    let d_unpadded_heights = unpadded_heights.to_device().unwrap();
    let d_widths = widths.to_device().unwrap();

    unsafe {
        tracegen(
            &d_trace_ptrs,
            &d_unpadded_heights,
            &d_widths,
            offsets.len(),
            &d_records,
            &d_record_offsets,
            &range_checker.count,
        )
        .unwrap();
    }

    traces
}

impl AccessAdapterInventoryGPU {
    pub fn new(range_checker: Arc<VariableRangeCheckerChipGPU>) -> Self {
        Self {
            _chips: [
                GenericAccessAdapterChipGPU::N2(AccessAdapterChipGPU),
                GenericAccessAdapterChipGPU::N4(AccessAdapterChipGPU),
                GenericAccessAdapterChipGPU::N8(AccessAdapterChipGPU),
                GenericAccessAdapterChipGPU::N16(AccessAdapterChipGPU),
                GenericAccessAdapterChipGPU::N32(AccessAdapterChipGPU),
            ],
            range_checker,
        }
    }

    pub fn generate_air_proving_ctxs(
        &self,
        arena: &DenseRecordArena,
    ) -> [AirProvingContext<GpuBackend>; NUM_ADAPTERS] {
        let records = arena.allocated();
        generate_traces_from_records(records, self.range_checker.clone())
            .map(AirProvingContext::simple_no_pis)
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use openvm_circuit::arch::{testing::RANGE_CHECKER_BUS, MemoryConfig};
    use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
    use openvm_stark_backend::{p3_field::FieldAlgebra, prover::hal::MatrixDimensions};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use stark_backend_gpu::{prelude::SC, types::F};

    use super::*;
    use crate::testing::{assert_eq_cpu_and_gpu_matrix, GpuChipTestBuilder};

    #[test]
    fn test_access_adapters_cpu_gpu_equivalence() {
        let mem_config = MemoryConfig::default();

        let mut rng = StdRng::seed_from_u64(42);
        let decomp = mem_config.decomp;
        let mut tester = GpuChipTestBuilder::volatile(mem_config)
            .with_variable_range_checker(VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, decomp));

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
        let allocated = tester.memory.memory.access_adapter_records.allocated();
        let gpu_traces = generate_traces_from_records(allocated, tester.range_checker());

        let all_memory_traces = tester
            .memory
            .controller
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
}
