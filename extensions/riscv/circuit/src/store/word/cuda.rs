use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::AirProvingContext;

use super::STORE_WORD_VALUE_CELLS;
use crate::{
    adapters::{Rv64StoreMultiByteAdapterCols, Rv64StoreMultiByteAdapterRecord, RV64_BYTE_BITS},
    cuda_abi::store_word_cuda,
    store::{core::StoreCoreCols, StoreRecord},
};

#[derive(new)]
pub struct Rv64StoreWordChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64StoreWordChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(Rv64StoreMultiByteAdapterRecord, StoreRecord)>();
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        let rvr_wire = arena.rvr_wire;
        let records = arena.allocated();
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        let delta_records = self.rvr_decode.device_delta_records(
            crate::rvr_gpu_decode::DeltaAirKind::StoreWord,
            &self.range_checker.device_ctx,
        );
        #[cfg(all(feature = "cuda", feature = "rvr"))]
        let no_delta_records = delta_records.is_none();
        #[cfg(not(all(feature = "cuda", feature = "rvr")))]
        let no_delta_records = true;
        if records.is_empty() && no_delta_records {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let trace_width = Rv64StoreMultiByteAdapterCols::<F>::width()
            + StoreCoreCols::<F, STORE_WORD_VALUE_CELLS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let device_ctx = &self.range_checker.device_ctx;

        #[cfg(all(feature = "cuda", feature = "rvr"))]
        if rvr_wire || delta_records.is_some() {
            use openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE;
            let compact_len = delta_records
                .as_ref()
                .map_or(records.len(), |buf| buf.len());
            assert_eq!(compact_len % PREFLIGHT_ADDSUB_RECORD_SIZE, 0);
            let compact_height =
                next_power_of_two_or_zero(compact_len / PREFLIGHT_ADDSUB_RECORD_SIZE);
            let (table, pc_base) = self
                .rvr_decode
                .device_operand_table(device_ctx)
                .expect("compact segment without a bound operand table");
            let compact_records = delta_records
                .unwrap_or_else(|| Arc::new(records.to_device_on(device_ctx).unwrap()));
            let trace =
                DeviceMatrix::<F>::with_capacity_on(compact_height, trace_width, device_ctx);
            unsafe {
                store_word_cuda::tracegen_compact(
                    trace.buffer(),
                    compact_height,
                    &compact_records,
                    &table,
                    pc_base,
                    self.pointer_max_bits,
                    &self.range_checker.count,
                    self.timestamp_max_bits as u32,
                    device_ctx.stream.as_raw(),
                )
                .unwrap();
            }
            return AirProvingContext::simple_no_pis(trace);
        }

        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let d_records = tracing::info_span!("trace_gen.h2d_records")
            .in_scope(|| records.to_device_on(device_ctx))
            .unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            store_word_cuda::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                self.pointer_max_bits,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
