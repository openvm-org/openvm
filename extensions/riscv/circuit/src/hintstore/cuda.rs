use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, RecordSeeker},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    cuda_abi::hintstore_cuda::{decode_offsets, tracegen, tracegen_replay},
    Rv64HintStoreCols, Rv64HintStoreLayout, Rv64HintStoreRecordMut,
};

#[derive(new)]
pub struct Rv64HintStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: Arc<crate::rvr_gpu_decode::RvrGpuDecodeState>,
}

// This is the info needed by each row to do parallel tracegen
#[repr(C)]
#[derive(new)]
pub struct OffsetInfo {
    pub record_offset: u32,
    pub local_idx: u32,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64HintStoreChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let width = Rv64HintStoreCols::<u8>::width();
        let device_ctx = &self.range_checker.device_ctx;
        #[cfg(feature = "rvr")]
        if let Some(replay) = self
            .rvr_decode
            .device_delta_records(crate::rvr_gpu_decode::DeltaAirKind::HintStore, device_ctx)
        {
            let segment_id = self
                .rvr_decode
                .g2_segment_id()
                .expect("G2 HintStore replay without a bound segment id");
            const REPLAY_ROW_SIZE: usize = 64;
            assert_eq!(
                replay.len() % REPLAY_ROW_SIZE,
                0,
                "G2 HintStore replay-row stride mismatch"
            );
            let rows_used = replay.len() / REPLAY_ROW_SIZE;
            if rows_used == 0 {
                return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
            }
            let trace_height = next_power_of_two_or_zero(rows_used);
            let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, width, device_ctx);
            let d_error = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
            d_error
                .fill_zero_on(device_ctx)
                .expect("G2 HintStore replay error clear");
            let timer = openvm_circuit::arch::rvr::gpu_profile::CudaStageTimer::start(device_ctx);
            unsafe {
                tracegen_replay(
                    d_trace.buffer(),
                    trace_height,
                    &replay,
                    rows_used,
                    self.pointer_max_bits as u32,
                    &self.range_checker.count,
                    self.timestamp_max_bits as u32,
                    &d_error,
                    device_ctx.stream.as_raw(),
                )
                .expect("G2 HintStore replay tracegen launch");
            }
            if let Some(timer) = timer {
                timer.finish("hintstore_replay", segment_id, rows_used * REPLAY_ROW_SIZE);
            }
            let error = d_error
                .to_host_on(device_ctx)
                .expect("G2 HintStore replay error D2H")[0];
            assert_eq!(
                error, 0,
                "G2 HintStore replay validation failed closed with code {error}"
            );
            return AirProvingContext::simple_no_pis(d_trace);
        }
        let direct_rows = arena.rvr_variable_rows;
        let records = arena.allocated_mut();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let d_records = tracing::info_span!("trace_gen.h2d_records")
            .in_scope(|| records.to_device_on(device_ctx))
            .unwrap();
        let (rows_used, d_record_offsets) = if let Some(rows_used) = direct_rows {
            let d_record_offsets =
                DeviceBuffer::<OffsetInfo>::with_capacity_on(rows_used, device_ctx);
            let d_error = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
            d_error
                .fill_zero_on(device_ctx)
                .expect("HintStore decoder error clear");
            unsafe {
                decode_offsets(
                    &d_records,
                    records.len(),
                    rows_used,
                    &d_record_offsets,
                    &d_error,
                    device_ctx.stream.as_raw(),
                )
                .expect("HintStore on-device variable-row decode launch");
            }
            let error = d_error
                .to_host_on(device_ctx)
                .expect("HintStore decoder error D2H")[0];
            assert_eq!(
                error, 0,
                "HintStore on-device variable-row decoder failed closed with code {error}"
            );
            (rows_used, d_record_offsets)
        } else {
            // Interpreter-produced Dense arenas do not carry rvr's direct
            // row-count metadata. Retain the established reference route for
            // those arenas; the rvr direct-final path above never scans on CPU.
            let mut offsets = Vec::<OffsetInfo>::new();
            let mut offset = 0;
            while offset < records.len() {
                let prev_offset = offset;
                let record = RecordSeeker::<
                    DenseRecordArena,
                    Rv64HintStoreRecordMut,
                    Rv64HintStoreLayout,
                >::get_record_at(&mut offset, records);
                for idx in 0..record.inner.num_words {
                    offsets.push(OffsetInfo::new(prev_offset as u32, idx));
                }
            }
            (offsets.len(), offsets.to_device_on(device_ctx).unwrap())
        };

        let trace_height = next_power_of_two_or_zero(rows_used);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, width, device_ctx);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                rows_used,
                &d_record_offsets,
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
