use std::{borrow::Borrow, mem::size_of, slice::from_raw_parts, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use p3_field::{Field, PrimeField32};

use super::columns::NativePoseidon2Cols;
use crate::cuda_abi::poseidon2_cuda;

#[derive(new)]
pub struct NativePoseidon2ChipGpu<const SBOX_REGISTERS: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl<const SBOX_REGISTERS: usize> Chip<DenseRecordArena, GpuBackend>
    for NativePoseidon2ChipGpu<SBOX_REGISTERS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        // For Poseidon2, the records are already the trace rows
        // Use the columns width directly
        let width = NativePoseidon2Cols::<F, SBOX_REGISTERS>::width();

        let record_size = width * size_of::<F>();
        assert_eq!(records.len() % record_size, 0);

        let height = records.len() / record_size;
        let padded_height = next_power_of_two_or_zero(height);

        let d_chunk_start = {
            let mut row_idx = 0;
            let row_slice = unsafe {
                let raw_ptr = records.as_ptr();
                from_raw_parts(raw_ptr as *const F, records.len() / size_of::<F>())
            };
            let mut chunk_start = Vec::new();
            // Allocated rows are not empty. Determine the chunk start indices.
            while row_idx < height {
                let start = row_idx * width;
                let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                    row_slice[start..(start + width)].borrow();
                chunk_start.push(row_idx as u32);
                if cols.simple.is_one() {
                    row_idx += 1;
                } else {
                    let num_non_inside_row = cols.inner.export.as_canonical_u32() as usize;
                    let non_inside_start = start + (num_non_inside_row - 1) * width;
                    let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                        row_slice[non_inside_start..(non_inside_start + width)].borrow();
                    let total_num_row = cols.inner.export.as_canonical_u32() as usize;
                    row_idx += total_num_row;
                };
            }
            chunk_start.to_device().unwrap()
        };

        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        let d_records = records.to_device().unwrap();

        unsafe {
            poseidon2_cuda::tracegen(
                trace.buffer(),
                padded_height,
                width,
                &d_records,
                height,
                &d_chunk_start,
                d_chunk_start.len() as u32,
                &self.range_checker.count,
                SBOX_REGISTERS as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
