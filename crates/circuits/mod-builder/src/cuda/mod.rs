//! GPU tracegen for FieldExpr-based chips: the kernel in `cuda/src/field_expr.cu`
//! interprets the device program serialized by [`crate::device_program`] (one thread
//! per row, grid-stride), filling both the Rv64VecHeapAdapter columns and the core
//! FieldExpr columns.
//!
//! The core-column interpreter is validated bit-exact against
//! `FieldExpressionFiller::fill_trace_row` (rows and range-checker counts); see the
//! `device_program_tests` module.

use std::sync::Arc;

use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};

use crate::{device_program::serialize_field_expr, FieldExpressionFiller};

pub mod cuda_abi;

/// Hard bound on the per-launch aux scratch; the launcher caps the grid at
/// 512 x 256 threads (grid-stride), so this is a fixed bound, not per-row.
const MAX_AUX_BYTES: usize = 1 << 30; // 1 GiB

pub struct FieldExprChipGpu {
    d_blob: DeviceBuffer<u32>,
    pub num_reads: usize,
    pub blocks: usize,
    pub adapter_width: usize,
    pub core_width: usize,
    /// Bytes per record in the dense arena (aligned (adapter, core) pair).
    pub record_stride: usize,
    /// Offset of the core record (opcode byte + input limbs) within a record.
    pub record_core_offset: usize,
    aux_words: usize,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub should_finalize: bool,
}

impl FieldExprChipGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new<A>(
        filler: &FieldExpressionFiller<A>,
        num_reads: usize,
        blocks: usize,
        adapter_width: usize,
        record_stride: usize,
        record_core_offset: usize,
        pointer_max_bits: u32,
        timestamp_max_bits: u32,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        let core_width = BaseAir::<F>::width(&filler.expr);
        let prog = serialize_field_expr(
            &filler.expr,
            filler.local_opcode_idx.clone(),
            filler.opcode_flag_idx.clone(),
            core_width,
        );
        let aux_words =
            (prog.num_value_slots + prog.num_vars) * prog.k + prog.scratch_len + 4 * prog.k;
        let blob = prog.to_blob();
        let d_blob = blob.to_device_on(&range_checker.device_ctx).unwrap();
        Self {
            d_blob,
            num_reads,
            blocks,
            adapter_width,
            core_width,
            record_stride,
            record_core_offset,
            aux_words,
            pointer_max_bits,
            timestamp_max_bits,
            range_checker,
            should_finalize: filler.should_finalize,
        }
    }

    /// Generates the full (adapter + core) trace on device from dense records.
    pub fn generate_proving_ctx(
        &self,
        records: &[u8],
        g2_segment_id: Option<u32>,
    ) -> AirProvingContext<GpuBackend> {
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % self.record_stride, 0);
        let rows_used = records.len() / self.record_stride;
        let height = rows_used.next_power_of_two();
        let width = self.adapter_width + self.core_width;

        let device_ctx = &self.range_checker.device_ctx;
        let h2d_timer = g2_segment_id.and_then(|_| {
            openvm_circuit::arch::rvr::gpu_profile::CudaStageTimer::start(device_ctx)
        });
        let d_records = records.to_device_on(device_ctx).unwrap();
        if let (Some(timer), Some(segment_id)) = (h2d_timer, g2_segment_id) {
            timer.finish("opaque_h2d", segment_id, records.len());
        }
        let d_trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);

        let n_threads = height.div_ceil(256).min(512) * 256;
        assert!(n_threads * self.aux_words * 4 <= MAX_AUX_BYTES);
        let d_aux = DeviceBuffer::<u32>::with_capacity_on(n_threads * self.aux_words, device_ctx);
        let d_err = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);

        unsafe {
            cuda_abi::field_expr_tracegen(
                d_trace.buffer(),
                height,
                rows_used,
                &self.d_blob,
                &d_records,
                self.record_stride,
                self.record_core_offset,
                &self.range_checker.count,
                &d_aux,
                self.aux_words,
                self.num_reads,
                self.blocks,
                self.pointer_max_bits,
                self.timestamp_max_bits,
                self.should_finalize,
                &d_err,
                device_ctx,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
