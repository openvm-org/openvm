use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    adapters::{Rv64BaseAluRegU16AdapterCols, Rv64BaseAluRegU16AdapterRecord, U16_BITS},
    cuda_abi::less_than_cuda::tracegen,
    LessThanCoreCols, LessThanCoreRecord,
};

#[derive(new)]
pub struct Rv64LessThanChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
    /// M-GPUDEC shared decode state (device operand table + per-segment
    /// emission mode).
    pub rvr_decode: std::sync::Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64LessThanChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64BaseAluRegU16AdapterRecord,
            LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
        )>();
        #[cfg(feature = "rvr")]
        let rvr_wire = arena.rvr_wire;
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = Rv64BaseAluRegU16AdapterCols::<F>::width()
            + LessThanCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;
        // M-GPUDEC (G2): this segment's arena carries compact wire records —
        // decode them on device against the per-exe operand table.
        #[cfg(feature = "rvr")]
        if rvr_wire {
            use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
            assert_eq!(
                records.len() % PREFLIGHT_ADDSUB_RECORD_SIZE,
                0,
                "compact arena stride mismatch"
            );
            let trace_height =
                next_power_of_two_or_zero(records.len() / PREFLIGHT_ADDSUB_RECORD_SIZE);
            let (d_table, pc_base) = self
                .rvr_decode
                .device_operand_table(device_ctx)
                .expect("compact segment without a bound operand table");
            let d_records = records.to_device_on(device_ctx).unwrap();
            let d_trace =
                DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
            unsafe {
                crate::cuda_abi::less_than_cuda::tracegen_compact(
                    d_trace.buffer(),
                    trace_height,
                    &d_records,
                    &d_table,
                    pc_base,
                    &self.range_checker.count,
                    self.timestamp_max_bits as u32,
                    device_ctx.stream.as_raw(),
                )
                .unwrap();
            }
            return AirProvingContext::simple_no_pis(d_trace);
        }

        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let d_records = tracing::info_span!("trace_gen.h2d_records")
            .in_scope(|| records.to_device_on(device_ctx))
            .unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
