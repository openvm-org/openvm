use std::sync::{Arc, Mutex};

use derive_new::new;
use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_instructions::riscv::RV64_BYTE_BITS;
use openvm_stark_backend::prover::AirProvingContext;
use p3_keccak_air::NUM_ROUNDS;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode,
    },
    openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode},
};

use crate::{
    keccakf_op::{columns::NUM_KECCAKF_OP_COLS, NUM_OP_ROWS_PER_INS},
    keccakf_perm::NUM_KECCAKF_PERM_COLS,
    xorin::columns::NUM_XORIN_VM_COLS,
};

mod cuda_abi;

// ========================== XorinVmChipGpu ==========================

#[derive(new)]
pub struct XorinVmChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: u32,
}

#[cfg(feature = "rvr")]
impl XorinVmChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(XorinOpcode::XORIN.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_height = next_power_of_two_or_zero(step_range.len());
        let d_trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, NUM_XORIN_VM_COLS, device_ctx);
        unsafe {
            cuda_abi::xorin::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                transcript.error_ptr(),
                XorinOpcode::XORIN.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.timestamp_max_bits,
                device_ctx.stream.as_raw(),
            )?;
        }
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

// ========================== Shared state for KeccakfOp <-> KeccakfPerm ==========================

/// Replay handoff from the Keccak operation trace to its permutation trace.
#[derive(Default)]
pub struct SharedKeccakfState {
    /// Twenty-five preimage words per active KECCAKF.
    /// The permutation chip takes this buffer, so it cannot survive trace generation.
    pub d_replay_preimages: Option<DeviceBuffer<u64>>,
    pub num_replay_steps: usize,
}

pub type SharedKeccakfStateGpu = Arc<Mutex<SharedKeccakfState>>;

// ========================== KeccakfOpChipGpu ==========================

#[derive(new)]
pub struct KeccakfOpChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: u32,
    pub shared_state: SharedKeccakfStateGpu,
}

#[cfg(feature = "rvr")]
impl KeccakfOpChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(KeccakfOpcode::KECCAKF.global_opcode());
        if step_range.is_empty() {
            let mut shared = self.shared_state.lock().unwrap();
            shared.d_replay_preimages = None;
            shared.num_replay_steps = 0;
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let trace_height = next_power_of_two_or_zero(step_range.len() * NUM_OP_ROWS_PER_INS);
        let d_trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, NUM_KECCAKF_OP_COLS, device_ctx);
        let d_preimages = DeviceBuffer::<u64>::with_capacity_on(step_range.len() * 25, device_ctx);
        unsafe {
            cuda_abi::keccakf_op::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                &d_preimages,
                transcript.error_ptr(),
                KeccakfOpcode::KECCAKF.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                self.timestamp_max_bits,
                device_ctx.stream.as_raw(),
            )?;
        }
        let mut shared = self.shared_state.lock().unwrap();
        shared.d_replay_preimages = Some(d_preimages);
        shared.num_replay_steps = step_range.len();
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

// ========================== KeccakfPermChipGpu ==========================

#[derive(new)]
pub struct KeccakfPermChipGpu {
    pub shared_state: SharedKeccakfStateGpu,
    pub device_ctx: GpuDeviceCtx,
}

#[cfg(feature = "rvr")]
impl KeccakfPermChipGpu {
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        program.ensure_replay_inputs(transcript, replay_plan, &self.device_ctx)?;
        let step_range = replay_plan.opcode_range(KeccakfOpcode::KECCAKF.global_opcode());
        let (d_preimages, num_replay_steps) = {
            let mut shared = self.shared_state.lock().unwrap();
            (shared.d_replay_preimages.take(), shared.num_replay_steps)
        };
        if step_range.is_empty() {
            if d_preimages.is_some() || num_replay_steps != 0 {
                return Err(GpuPostflightError::InvalidTranscript(
                    "Keccak replay state exists without executed KECCAKF".to_string(),
                ));
            }
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }
        let d_preimages = d_preimages.ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "KeccakfPerm replay ran before KeccakfOp replay".to_string(),
            )
        })?;
        if num_replay_steps != step_range.len() || d_preimages.len() != step_range.len() * 25 {
            return Err(GpuPostflightError::InvalidTranscript(
                "Keccak replay preimage handoff has the wrong length".to_string(),
            ));
        }

        let trace_height = next_power_of_two_or_zero(step_range.len() * NUM_ROUNDS);
        let d_trace = DeviceMatrix::<F>::with_capacity_on(
            trace_height,
            NUM_KECCAKF_PERM_COLS,
            &self.device_ctx,
        );
        let padded_permutations = trace_height.div_ceil(NUM_ROUNDS);
        let d_round_states = DeviceBuffer::<u64>::with_capacity_on(
            padded_permutations * NUM_ROUNDS * 25,
            &self.device_ctx,
        );
        unsafe {
            cuda_abi::keccakf_perm::replay_tracegen(
                d_trace.buffer(),
                trace_height,
                transcript.program_log(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                &d_preimages,
                &d_round_states,
                transcript.error_ptr(),
                self.device_ctx.stream.as_raw(),
            )?;
        }
        // Both replay-only buffers are dropped here, after their kernels have been
        // enqueued on the owning stream and before proving starts.
        drop(d_round_states);
        drop(d_preimages);
        Ok(AirProvingContext::simple_no_pis(d_trace))
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests {
    use openvm_circuit::utils::test_gpu_engine;
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
    };
    use openvm_stark_backend::StarkEngine;
    use rvr_state::PreflightProgramEvent;

    use super::*;
    use crate::KECCAK_WIDTH_U64S;

    #[test]
    fn keccakf_permutation_replay_rejects_wrapped_program_index() {
        let engine = test_gpu_engine();
        let device_ctx = &engine.device().device_ctx;
        let height = 32usize;
        let program = [
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 27,
            },
        ]
        .to_device_on(device_ctx)
        .unwrap();
        let steps = [[u32::MAX, 0u32]].to_device_on(device_ctx).unwrap();
        let preimages = [0u64; KECCAK_WIDTH_U64S].to_device_on(device_ctx).unwrap();
        let blocks_to_fill = height.div_ceil(NUM_ROUNDS);
        let round_states = DeviceBuffer::<u64>::with_capacity_on(
            blocks_to_fill * NUM_ROUNDS * KECCAK_WIDTH_U64S,
            device_ctx,
        );
        let trace = DeviceBuffer::<F>::with_capacity_on(height * NUM_KECCAKF_PERM_COLS, device_ctx);
        let error = [0u32].to_device_on(device_ctx).unwrap();

        unsafe {
            cuda_abi::keccakf_perm::replay_tracegen(
                &trace,
                height,
                program.view(),
                steps.view(),
                0,
                1,
                &preimages,
                &round_states,
                error.as_mut_ptr(),
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        assert_eq!(error.to_host_on(device_ctx).unwrap()[0], 821);
    }
}
