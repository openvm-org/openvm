use std::{ptr::null_mut, sync::Arc};

use openvm_circuit::{
    arch::{CustomBorrow, DenseRecordArena, SizedRecord},
    system::memory::adapter::{
        records::{AccessLayout, AccessRecordMut},
        AccessAdapterCols,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, prover_backend::GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::types::AirProvingContext;

use crate::cuda_abi::access_adapters::tracegen;

pub(crate) const NUM_ADAPTERS: usize = 5;

pub struct AccessAdapterInventoryGPU {
    max_access_adapter_n: usize,
    timestamp_max_bits: usize,
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    #[cfg(feature = "metrics")]
    pub(super) unpadded_heights: Vec<usize>,
}

#[repr(C)]
pub struct OffsetInfo {
    pub record_offset: u32,
    pub adapter_rows: [u32; NUM_ADAPTERS],
}

impl AccessAdapterInventoryGPU {
    pub(crate) fn generate_traces_from_records(
        &mut self,
        records: &mut [u8],
    ) -> Vec<Option<DeviceMatrix<F>>> {
        let max_access_adapter_n = &self.max_access_adapter_n;
        let timestamp_max_bits = self.timestamp_max_bits;
        let range_checker = &self.range_checker;

        assert!(max_access_adapter_n.is_power_of_two());
        let cnt_adapters = max_access_adapter_n.ilog2() as usize;
        if records.is_empty() {
            return vec![None; cnt_adapters];
        }

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
        let unpadded_heights = row_ids
            .iter()
            .take(cnt_adapters)
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let traces = unpadded_heights
            .iter()
            .enumerate()
            .map(|(i, &h)| match h {
                0 => None,
                h => Some(DeviceMatrix::<F>::with_capacity(
                    next_power_of_two_or_zero(h),
                    widths[i],
                )),
            })
            .collect::<Vec<_>>();
        let trace_ptrs = traces
            .iter()
            .map(|trace| {
                trace
                    .as_ref()
                    .map_or_else(null_mut, |t| t.buffer().as_mut_raw_ptr())
            })
            .collect::<Vec<_>>();
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
                timestamp_max_bits,
            )
            .unwrap();
        }
        #[cfg(feature = "metrics")]
        {
            self.unpadded_heights = unpadded_heights;
        }

        traces
    }

    pub fn new(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        max_access_adapter_n: usize,
        timestamp_max_bits: usize,
    ) -> Self {
        Self {
            range_checker,
            max_access_adapter_n,
            timestamp_max_bits,
            #[cfg(feature = "metrics")]
            unpadded_heights: Vec::new(),
        }
    }

    // @dev: mutable borrow is only to update `self.unpadded_heights` for metrics
    pub fn generate_air_proving_ctxs(
        &mut self,
        mut arena: DenseRecordArena,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let records = arena.allocated_mut();
        self.generate_traces_from_records(records)
            .into_iter()
            .map(|trace| AirProvingContext {
                cached_mains: vec![],
                common_main: trace,
                public_values: vec![],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use openvm_circuit::{
        arch::{
            testing::{MEMORY_BUS, RANGE_CHECKER_BUS},
            MemoryConfig,
        },
        system::memory::{offline_checker::MemoryBus, MemoryController},
    };
    use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
    use openvm_cuda_backend::{data_transporter::assert_eq_host_and_device_matrix, prelude::SC};
    use openvm_stark_backend::{p3_field::FieldAlgebra, prover::hal::MatrixDimensions};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::arch::testing::{GpuChipTestBuilder, TestBuilder};

    #[test]
    fn test_cuda_access_adapters_cpu_gpu_equivalence() {
        let mem_config = MemoryConfig::default();

        let mut rng = StdRng::seed_from_u64(42);
        let decomp = mem_config.decomp;
        let mut tester = GpuChipTestBuilder::volatile(
            mem_config.clone(),
            VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, decomp),
        );

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
        let mut access_adapter_inv = AccessAdapterInventoryGPU::new(
            tester.range_checker(),
            mem_config.max_access_adapter_n,
            mem_config.timestamp_max_bits,
        );
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
                assert_eq_host_and_device_matrix(cpu_trace, gpu_trace);
            }
        }
    }
}
