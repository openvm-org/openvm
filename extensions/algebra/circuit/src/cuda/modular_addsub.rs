use std::sync::Arc;

use num_bigint::BigUint;
use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
use openvm_circuit::arch::rvr::cuda::{
    GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_riscv_adapters::Rv64VecHeapAdapterCols;
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};

use super::{
    cuda_abi,
    vec_heap::{checked_trace_shape, gather_vec_heap_trace_inputs_device},
    DeferredGpuRangeCheckerCounts,
};
use crate::modular_chip::ModularChip;

pub struct ModularAddSubReplayChipGpu<const BLOCKS: usize> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    modulus: DeviceBuffer<u8>,
    opcode_base: usize,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
    width: usize,
}

impl<const BLOCKS: usize> ModularAddSubReplayChipGpu<BLOCKS> {
    pub fn new(
        chip: &ModularChip<F, BLOCKS>,
        modulus: &BigUint,
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Result<Option<Self>, GpuRvrInputError> {
        if !matches!(BLOCKS, 4 | 6) {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "unsupported modular add/sub block count {BLOCKS}"
            )));
        }
        let num_bytes = BLOCKS * openvm_circuit::arch::MEMORY_BLOCK_BYTES;
        let mut modulus_bytes = modulus.to_bytes_le();
        if modulus_bytes.is_empty() || modulus_bytes.len() > num_bytes {
            return Err(GpuRvrInputError::InvalidTranscript(
                "modular add/sub replay requires a nonzero modulus fitting its heap layout"
                    .to_string(),
            ));
        }
        modulus_bytes.resize(num_bytes, 0);

        let adapter_width = Rv64VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS>::width();
        let core_width = BaseAir::<F>::width(&chip.inner.expr);
        let expected_core_width = 4usize
            .checked_mul(num_bytes)
            .and_then(|width| width.checked_add(4))
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "modular add/sub replay trace width overflow".to_string(),
                )
            })?;
        if core_width != expected_core_width {
            return Ok(None);
        }
        let width = adapter_width.checked_add(core_width).ok_or_else(|| {
            GpuRvrInputError::InvalidTranscript(
                "modular add/sub replay trace width overflow".to_string(),
            )
        })?;
        let modulus = modulus_bytes
            .as_slice()
            .to_device_on(&range_checker.device_ctx)?;
        Ok(Some(Self {
            range_checker,
            modulus,
            opcode_base,
            pointer_max_bits,
            timestamp_max_bits,
            width,
        }))
    }

    pub fn generate_proving_ctx(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let local_opcodes = [
            Rv64ModularArithmeticOpcode::ADD as usize,
            Rv64ModularArithmeticOpcode::SUB as usize,
            Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize,
        ];
        let projection = gather_vec_heap_trace_inputs_device::<2, BLOCKS>(
            program,
            transcript,
            replay_plan,
            self.opcode_base,
            &local_opcodes,
            self.pointer_max_bits,
            &self.range_checker.device_ctx,
        )?;
        if projection.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let (height, _) =
            checked_trace_shape(projection.len(), self.width, self.timestamp_max_bits)?;
        let device_ctx = &self.range_checker.device_ctx;
        let trace = DeviceMatrix::<F>::with_capacity_on(height, self.width, device_ctx);
        let delta = DeviceBuffer::with_capacity_on(self.range_checker.count.len(), device_ctx);
        delta.fill_zero_on(device_ctx)?;
        unsafe {
            cuda_abi::modular_addsub_replay_tracegen(
                trace.buffer(),
                height,
                &projection.inputs,
                &self.modulus,
                Rv64ModularArithmeticOpcode::ADD as u32,
                Rv64ModularArithmeticOpcode::SUB as u32,
                Rv64ModularArithmeticOpcode::SETUP_ADDSUB as u32,
                &delta,
                u32::try_from(self.pointer_max_bits).map_err(|_| {
                    GpuRvrInputError::InvalidTranscript(
                        "modular add/sub pointer width does not fit u32".to_string(),
                    )
                })?,
                u32::try_from(self.timestamp_max_bits).map_err(|_| {
                    GpuRvrInputError::InvalidTranscript(
                        "modular add/sub timestamp width does not fit u32".to_string(),
                    )
                })?,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        transcript.synchronize()?;
        let error = transcript.error_code()?;
        if error != 0 {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "modular add/sub replay rejected transcript with code {error}"
            )));
        }
        DeferredGpuRangeCheckerCounts {
            target: self.range_checker.count.clone(),
            delta,
            device_ctx: device_ctx.clone(),
        }
        .commit()?;
        drop(projection);
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}
