//! Arena-free checkpoint replay for algebra GPU trace generation.

use std::{ops::Range, sync::Arc};

use openvm_algebra_transpiler::Rv64ModularArithmeticOpcode;
use openvm_circuit::arch::rvr::cuda::{
    GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    VmOpcode,
};
use openvm_riscv_adapters::Rv64IsEqualModU16AdapterCols;
use openvm_stark_backend::prover::AirProvingContext;

use crate::modular_chip::ModularIsEqualCoreCols;

mod cuda_abi;
pub mod field_expr;
pub(crate) mod modular_addsub;
pub mod vec_heap;

const MAX_ALGEBRA_TRACE_HEIGHT: usize = 1 << 26;

struct DeferredGpuRangeCheckerCounts {
    target: Arc<DeviceBuffer<F>>,
    delta: DeviceBuffer<F>,
    device_ctx: GpuDeviceCtx,
}

impl DeferredGpuRangeCheckerCounts {
    pub fn commit(self) -> Result<(), GpuRvrInputError> {
        unsafe {
            cuda_abi::merge_range_counts(
                self.target.as_ref(),
                &self.delta,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        Ok(())
    }
}

fn checked_opcode(base: usize, local: usize) -> Result<VmOpcode, GpuRvrInputError> {
    let opcode = base.checked_add(local).ok_or_else(|| {
        GpuRvrInputError::InvalidTranscript("algebra opcode overflow".to_string())
    })?;
    u32::try_from(opcode).map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))?;
    Ok(VmOpcode::from_usize(opcode))
}

fn opcode_pair_range(
    replay_plan: &GpuRvrReplayPlan,
    opcodes: [VmOpcode; 2],
) -> Result<Range<usize>, GpuRvrInputError> {
    let ranges = opcodes.map(|opcode| replay_plan.opcode_range(opcode));
    let Some(start) = ranges
        .iter()
        .filter(|range| !range.is_empty())
        .map(|range| range.start)
        .min()
    else {
        return Ok(0..0);
    };
    let end = ranges
        .iter()
        .filter(|range| !range.is_empty())
        .map(|range| range.end)
        .max()
        .unwrap();
    if end - start != ranges.iter().map(Range::len).sum::<usize>() {
        return Err(GpuRvrInputError::InvalidTranscript(
            "ModularIsEqual opcode ranges are not contiguous".to_string(),
        ));
    }
    Ok(start..end)
}

pub struct ModularIsEqualReplayChipGpu<const NUM_LANES: usize, const TOTAL_LIMBS: usize> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    d_modulus: DeviceBuffer<u16>,
    opcode_base: usize,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize>
    ModularIsEqualReplayChipGpu<NUM_LANES, TOTAL_LIMBS>
{
    pub fn new(
        modulus_limbs: [u16; TOTAL_LIMBS],
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        let d_modulus = modulus_limbs
            .as_slice()
            .to_device_on(&range_checker.device_ctx)
            .unwrap();
        Self {
            range_checker,
            d_modulus,
            opcode_base,
            pointer_max_bits,
            timestamp_max_bits,
        }
    }

    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let is_eq_opcode = checked_opcode(
            self.opcode_base,
            Rv64ModularArithmeticOpcode::IS_EQ as usize,
        )?;
        let setup_opcode = checked_opcode(
            self.opcode_base,
            Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize,
        )?;
        let range = opcode_pair_range(replay_plan, [is_eq_opcode, setup_opcode])?;
        if range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let width = Rv64IsEqualModU16AdapterCols::<F, 2, NUM_LANES>::width()
            .checked_add(ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width())
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "ModularIsEqual trace width overflow".to_string(),
                )
            })?;
        let height = range.len().checked_next_power_of_two().ok_or_else(|| {
            GpuRvrInputError::InvalidTranscript("ModularIsEqual trace height overflow".to_string())
        })?;
        let timestamp_limit = 1usize
            .checked_shl(u32::try_from(self.timestamp_max_bits).map_err(|_| {
                GpuRvrInputError::InvalidTranscript(
                    "timestamp width cannot be represented as a trace height".to_string(),
                )
            })?)
            .ok_or_else(|| {
                GpuRvrInputError::InvalidTranscript(
                    "timestamp width cannot be represented as a trace height".to_string(),
                )
            })?;
        let max_height = timestamp_limit.min(MAX_ALGEBRA_TRACE_HEIGHT);
        if height > max_height || height.checked_mul(width).is_none() {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "ModularIsEqual trace shape {height}x{width} exceeds the replay allocation limit"
            )));
        }
        let trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        let delta = DeviceBuffer::with_capacity_on(self.range_checker.count.len(), device_ctx);
        delta.fill_zero_on(device_ctx)?;
        let opcode_base = u32::try_from(self.opcode_base)
            .map_err(|_| GpuRvrInputError::OpcodeTooLarge(self.opcode_base))?;
        let pointer_max_bits = u32::try_from(self.pointer_max_bits).map_err(|_| {
            GpuRvrInputError::InvalidTranscript(
                "ModularIsEqual pointer width does not fit u32".to_string(),
            )
        })?;
        let timestamp_max_bits = u32::try_from(self.timestamp_max_bits).map_err(|_| {
            GpuRvrInputError::InvalidTranscript(
                "ModularIsEqual timestamp width does not fit u32".to_string(),
            )
        })?;
        unsafe {
            cuda_abi::replay_tracegen(
                trace.buffer(),
                height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                range.start,
                range.len(),
                transcript.error_ptr(),
                opcode_base,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                &self.d_modulus,
                &delta,
                NUM_LANES,
                pointer_max_bits,
                timestamp_max_bits,
                device_ctx.stream.as_raw(),
            )?;
        }
        transcript.synchronize()?;
        let error = transcript.error_code()?;
        if error != 0 {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "ModularIsEqual replay rejected transcript with code {error}"
            )));
        }
        DeferredGpuRangeCheckerCounts {
            target: self.range_checker.count.clone(),
            delta,
            device_ctx: device_ctx.clone(),
        }
        .commit()?;
        Ok(AirProvingContext::simple_no_pis(trace))
    }

    pub fn checkpoint_opcodes(&self) -> Result<[VmOpcode; 2], GpuRvrInputError> {
        Ok([
            checked_opcode(
                self.opcode_base,
                Rv64ModularArithmeticOpcode::IS_EQ as usize,
            )?,
            checked_opcode(
                self.opcode_base,
                Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize,
            )?,
        ])
    }
}
