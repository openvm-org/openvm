use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use crate::{
    adapters::{
        Rv32BranchAdapterCols, Rv32BranchAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    cuda_abi::branch_lt_cuda::tracegen,
    BranchLessThanCoreCols, BranchLessThanCoreRecord,
};

#[derive(new)]
pub struct Rv32BranchLessThanChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32BranchLessThanChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchLessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width =
            BranchLessThanCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
                + Rv32BranchAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
