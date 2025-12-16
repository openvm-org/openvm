use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix,
    chip::get_empty_air_proving_ctx,
    prelude::F,
    prover_backend::GpuBackend,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::xorin::{columns::NUM_XORIN_VM_COLS, trace::XorinVmRecordHeader};

mod cuda_abi;

#[cfg(test)]
mod tests;

#[derive(new)]
pub struct XorinVmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for XorinVmChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<XorinVmRecordHeader>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = NUM_XORIN_VM_COLS;
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            cuda_abi::xorin::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.pointer_max_bits as u32,
                self.timestamp_max_bits,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}