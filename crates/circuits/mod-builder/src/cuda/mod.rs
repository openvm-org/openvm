//! GPU tracegen for FieldExpr-based chips (SKELETON — replaces the Hybrid* CPU
//! tracegen path in extensions/{algebra,ecc,pairing}/circuit/src/extension/hybrid.rs).
//!
//! The kernel (cuda/src/field_expr.cu) interprets the device program serialized by
//! [`crate::device_program`]; it is validated bit-exact against
//! `FieldExpressionFiller::fill_trace_row` (rows and range-checker histograms) on
//! EcAddNe / MulDiv-with-flags / IntMul-IntAdd expressions (L40S).
//!
//! Remaining wiring (documented, not blocking the kernel):
//!  1. Adapter columns: fill [0, adapter_width) with the Rv64VecHeapAdapter device
//!     fill (same as bigint's cuda kernels); this module fills core columns only.
//!  2. Per-chip constructors in the extension crates: build the chip's FieldExpr,
//!     call `serialize_field_expr`, and construct [`FieldExprChipGpu`] with the
//!     adapter width/record layout for that chip.
//!  3. `should_finalize`: mirror the CPU filler's flag per chip.

use std::sync::Arc;

use openvm_circuit::{arch::DenseRecordArena, system::cuda::extensions::SystemGpuBuilder};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::{p3_util::next_power_of_two_or_zero, prover::AirProvingContext, Chip};

use crate::device_program::DeviceFieldExprProgram;

pub mod cuda_abi;

/// VRAM budget guard for the per-thread aux scratch (grid is capped at 512 blocks
/// x 256 threads in the launcher, so this is a hard bound, not per-row).
const MAX_AUX_BYTES: usize = 1 << 30; // 1 GiB, far below the 15 GiB budget

pub struct FieldExprChipGpu {
    /// Serialized device program, uploaded once.
    pub d_blob: DeviceBuffer<u32>,
    pub prog_meta: FieldExprGpuMeta,
    /// Shared GPU range checker histogram (same layout as
    /// `VariableRangeCheckerChip::add_count` indexing: bin = (1 << bits) + value - 1).
    pub range_checker: Arc<openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU>,
    pub should_finalize: bool,
}

#[derive(Clone, Copy)]
pub struct FieldExprGpuMeta {
    /// Core sub-trace width (== BaseAir::width of the FieldExpr).
    pub core_width: usize,
    /// Adapter columns preceding the core columns in the full trace.
    pub adapter_width: usize,
    /// Bytes per record in the dense arena: aligned (adapter_record, core_record).
    pub record_stride: usize,
    /// Offset of the core record (opcode byte + input limbs) within a record.
    pub record_core_offset: usize,
    /// Per-thread scratch words: (num_slots + num_vars) * k + scratch_len + 4k.
    pub aux_words: usize,
}

impl FieldExprGpuMeta {
    pub fn from_program(
        prog: &DeviceFieldExprProgram,
        adapter_width: usize,
        record_stride: usize,
        record_core_offset: usize,
    ) -> Self {
        Self {
            core_width: prog.width,
            adapter_width,
            record_stride,
            record_core_offset,
            aux_words: (prog.num_value_slots + prog.num_vars) * prog.k
                + prog.scratch_len
                + 4 * prog.k,
        }
    }
}

impl Chip<DenseRecordArena, GpuBackend> for FieldExprChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        let m = &self.prog_meta;
        debug_assert_eq!(records.len() % m.record_stride, 0);
        let rows_used = records.len() / m.record_stride;
        let height = next_power_of_two_or_zero(rows_used);
        let width = m.adapter_width + m.core_width;

        let device_ctx = &self.range_checker.device_ctx;
        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);

        // Aux scratch: bounded by the launcher's 512x256 grid cap.
        let n_threads = (height.div_ceil(256)).min(512) * 256;
        let aux_bytes = n_threads * m.aux_words * 4;
        assert!(aux_bytes <= MAX_AUX_BYTES, "aux scratch exceeds budget");
        let d_aux = DeviceBuffer::<u32>::with_capacity_on(n_threads * m.aux_words, device_ctx);
        let d_err = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);

        // TODO(integration): launch the vec-heap adapter fill for columns
        // [0, adapter_width) here (bigint.cu pattern), or fuse it into the kernel.

        unsafe {
            cuda_abi::field_expr_tracegen(
                // core columns start after the adapter columns
                d_trace.buffer(),
                m.adapter_width,
                height,
                rows_used,
                &self.d_blob,
                &d_records,
                m.record_stride,
                m.record_core_offset,
                &self.range_checker.count,
                &d_aux,
                m.aux_words,
                self.should_finalize,
                &d_err,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
