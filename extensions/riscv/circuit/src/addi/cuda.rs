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

use super::{AddICoreCols, AddICoreRecord};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterCols, Rv64BaseAluImmU16AdapterRecord,
        Rv64BaseAluWImmU16AdapterCols, Rv64BaseAluWImmU16AdapterRecord, RV64_WORD_U16_LIMBS,
        U16_BITS,
    },
    cuda_abi::{addi_cuda, addi_w_cuda},
};

const _: () = assert!(
    size_of::<(
        Rv64BaseAluWImmU16AdapterRecord,
        AddICoreRecord<RV64_WORD_U16_LIMBS>,
    )>() == 48
);

#[derive(new)]
pub struct Rv64AddIChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: std::sync::Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

#[derive(new)]
pub struct Rv64AddIWChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64AddIChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64BaseAluImmU16AdapterRecord,
            AddICoreRecord<BLOCK_FE_WIDTH>,
        )>();
        #[cfg(feature = "rvr")]
        let rvr_wire = arena.rvr_wire;
        let records = arena.allocated();
        #[cfg(feature = "rvr")]
        let delta_records = self.rvr_decode.device_delta_records(
            crate::rvr_gpu_decode::DeltaAirKind::AddI,
            &self.range_checker.device_ctx,
        );
        #[cfg(feature = "rvr")]
        let g2_records = self.rvr_decode.device_g2_trace_input(
            crate::rvr_gpu_decode::DeltaAirKind::AddI,
            &self.range_checker.device_ctx,
        );
        #[cfg(feature = "rvr")]
        let no_delta_records = delta_records.is_none() && g2_records.is_none();
        #[cfg(not(feature = "rvr"))]
        let no_delta_records = true;
        if records.is_empty() && no_delta_records {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = Rv64BaseAluImmU16AdapterCols::<F>::width()
            + AddICoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
        let device_ctx = &self.range_checker.device_ctx;

        #[cfg(feature = "rvr")]
        if let Some(g2_records) = g2_records {
            return AirProvingContext::simple_no_pis(g2_records.tracegen(
                trace_width,
                0,
                &self.range_checker.count,
                None,
                None,
                crate::cuda_abi::UInt2::new(0, 0),
                self.timestamp_max_bits as u32,
                device_ctx,
            ));
        }

        #[cfg(feature = "rvr")]
        if rvr_wire || delta_records.is_some() {
            use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
            let compact_len = delta_records
                .as_ref()
                .map_or(records.len(), |buffer| buffer.len());
            assert_eq!(
                compact_len % PREFLIGHT_ADDSUB_RECORD_SIZE,
                0,
                "compact AddI arena stride mismatch"
            );
            let trace_height =
                next_power_of_two_or_zero(compact_len / PREFLIGHT_ADDSUB_RECORD_SIZE);
            let (d_table, pc_base) = self
                .rvr_decode
                .device_operand_table(device_ctx)
                .expect("compact AddI segment without a bound operand table");
            let d_records = delta_records
                .unwrap_or_else(|| Arc::new(records.to_device_on(device_ctx).unwrap()));
            let d_trace =
                DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
            unsafe {
                crate::cuda_abi::addi_cuda::tracegen_compact(
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
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            addi_cuda::tracegen(
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

impl Chip<DenseRecordArena, GpuBackend> for Rv64AddIWChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv64BaseAluWImmU16AdapterRecord,
            AddICoreRecord<RV64_WORD_U16_LIMBS>,
        )>();
        #[cfg(feature = "rvr")]
        let rvr_wire = arena.rvr_wire;
        let records = arena.allocated();
        #[cfg(feature = "rvr")]
        let delta_records = self.rvr_decode.device_delta_records(
            crate::rvr_gpu_decode::DeltaAirKind::AddIW,
            &self.range_checker.device_ctx,
        );
        #[cfg(feature = "rvr")]
        let g2_records = self.rvr_decode.device_g2_trace_input(
            crate::rvr_gpu_decode::DeltaAirKind::AddIW,
            &self.range_checker.device_ctx,
        );
        #[cfg(feature = "rvr")]
        let no_compact_records = delta_records.is_none() && g2_records.is_none();
        #[cfg(not(feature = "rvr"))]
        let no_compact_records = true;
        if records.is_empty() && no_compact_records {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = Rv64BaseAluWImmU16AdapterCols::<F>::width()
            + AddICoreCols::<F, RV64_WORD_U16_LIMBS, U16_BITS>::width();
        let device_ctx = &self.range_checker.device_ctx;

        #[cfg(feature = "rvr")]
        if let Some(g2_records) = g2_records {
            return AirProvingContext::simple_no_pis(g2_records.tracegen(
                trace_width,
                0,
                &self.range_checker.count,
                None,
                None,
                crate::cuda_abi::UInt2::new(0, 0),
                self.timestamp_max_bits as u32,
                device_ctx,
            ));
        }
        #[cfg(feature = "rvr")]
        if rvr_wire || delta_records.is_some() {
            use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
            let compact_len = delta_records
                .as_ref()
                .map_or(records.len(), |buf| buf.len());
            assert_eq!(compact_len % PREFLIGHT_ADDSUB_RECORD_SIZE, 0);
            let trace_height =
                next_power_of_two_or_zero(compact_len / PREFLIGHT_ADDSUB_RECORD_SIZE);
            let (table, pc_base) = self
                .rvr_decode
                .device_operand_table(device_ctx)
                .expect("compact AddIW segment without operand table");
            let compact = delta_records
                .unwrap_or_else(|| Arc::new(records.to_device_on(device_ctx).unwrap()));
            let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
            unsafe {
                addi_w_cuda::tracegen_compact(
                    trace.buffer(),
                    trace_height,
                    &compact,
                    &table,
                    pc_base,
                    &self.range_checker.count,
                    self.timestamp_max_bits as u32,
                    device_ctx.stream.as_raw(),
                )
                .unwrap();
            }
            return AirProvingContext::simple_no_pis(trace);
        }

        debug_assert_eq!(records.len() % RECORD_SIZE, 0);
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            addi_w_cuda::tracegen(
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
