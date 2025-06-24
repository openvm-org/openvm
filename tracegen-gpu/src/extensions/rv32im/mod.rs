use std::{mem::size_of, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_rv32im_circuit::{
    adapters::{Rv32RdWriteAdapterRecord, RV32_CELL_BITS},
    Rv32AuipcAir, Rv32AuipcCoreRecord,
};
use openvm_stark_backend::{rap::get_air_name, AirRef, ChipUsageGetter};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend, types::SC,
};

use crate::{
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
    DeviceChip,
};

mod cuda;
use cuda::auipc::tracegen;

#[cfg(test)]
mod tests;

pub struct Rv32AuipcChipGpu<'a> {
    pub air: Rv32AuipcAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub arena: Option<&'a DenseRecordArena>,
}

impl<'a> Rv32AuipcChipGpu<'a> {
    pub fn new(
        air: Rv32AuipcAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
        arena: Option<&'a DenseRecordArena>,
    ) -> Self {
        Self {
            air,
            range_checker,
            bitwise_lookup,
            arena,
        }
    }
}

impl ChipUsageGetter for Rv32AuipcChipGpu<'_> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        const RECORD_SIZE: usize = size_of::<(Rv32RdWriteAdapterRecord, Rv32AuipcCoreRecord)>();
        let records_len = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .len();
        assert_eq!(records_len % RECORD_SIZE, 0);
        records_len / RECORD_SIZE
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl DeviceChip<SC, GpuBackend> for Rv32AuipcChipGpu<'_> {
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let d_records = self.arena.as_ref().unwrap().records_buffer.get_ref()
            [..self.arena.as_ref().unwrap().records_buffer.position() as usize]
            .to_device()
            .unwrap();
        let trace_height = next_power_of_two_or_zero(self.current_trace_height());
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        unsafe {
            tracegen(
                trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
            )
            .unwrap();
        }
        trace
    }
}
