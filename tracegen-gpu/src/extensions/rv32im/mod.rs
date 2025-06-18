use std::{mem::size_of, sync::Arc};

use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_rv32im_circuit::{
    adapters::{Rv32RdWriteAdapterRecord, RV32_CELL_BITS},
    Rv32AuipcAir, Rv32AuipcCoreRecord,
};
use p3_air::BaseAir;
use stark_backend_gpu::{base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F};

use crate::primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};

mod cuda;
use cuda::auipc::tracegen;

#[cfg(test)]
mod test;

pub struct Rv32AuipcChipGpu {
    pub air: Rv32AuipcAir,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
}

impl Rv32AuipcChipGpu {
    pub fn new(
        air: Rv32AuipcAir,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    ) -> Self {
        Self {
            air,
            range_checker,
            bitwise_lookup,
        }
    }

    pub fn generate_trace(&self, records: &[u8]) -> DeviceMatrix<F> {
        let record_size = size_of::<(Rv32RdWriteAdapterRecord, Rv32AuipcCoreRecord)>();
        assert_eq!(records.len() % record_size, 0);
        let num_records = records.len() / record_size;
        let width = BaseAir::<F>::width(&self.air);

        let d_records = records.to_device().unwrap();
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, width);
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
