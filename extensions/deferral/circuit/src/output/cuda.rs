use std::{array::from_fn, mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, SizedRecord},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV64_BYTE_BITS;
use openvm_stark_backend::{p3_field::PrimeCharacteristicRing, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
#[cfg(feature = "rvr")]
use {
    crate::cuda_abi::output::DeferralOutputReplayCall,
    openvm_circuit::arch::rvr::cuda::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    openvm_cuda_common::copy::MemCopyD2H,
    openvm_deferral_transpiler::DeferralOpcode,
    openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode,
    },
};

use super::{
    DeferralOutputCols, DeferralOutputLayout, DeferralOutputMetadata, DeferralOutputRecordHeader,
    DeferralOutputRecordMut,
};
use crate::{
    cuda_abi::output::{self, DeferralOutputPerCall, DeferralOutputPerRow},
    poseidon2::{
        deferral_poseidon2_chip, DeferralPoseidon2ProducerBuffer, DeferralPoseidon2SharedBuffer,
    },
    utils::f_commit_to_bytes,
};

#[derive(new)]
pub struct DeferralOutputChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub address_bits: usize,
    pub timestamp_max_bits: usize,
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
    pub poseidon2: DeferralPoseidon2SharedBuffer,
}

#[cfg(feature = "rvr")]
pub(crate) fn checked_replay_trace_shape(
    rows_used: u64,
    trace_width: usize,
    max_trace_height: usize,
) -> Result<(usize, usize), GpuPostflightError> {
    let rows_used = usize::try_from(rows_used).map_err(|_| {
        GpuPostflightError::InvalidTranscript("Deferral OUTPUT row count exceeds usize".to_string())
    })?;
    let trace_height = rows_used.checked_next_power_of_two().ok_or_else(|| {
        GpuPostflightError::InvalidTranscript("Deferral OUTPUT trace height overflow".to_string())
    })?;
    if trace_height > max_trace_height {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "Deferral OUTPUT padded trace height {trace_height} exceeds segment limit {max_trace_height}"
        )));
    }
    let trace_elements = trace_height.checked_mul(trace_width).ok_or_else(|| {
        GpuPostflightError::InvalidTranscript(
            "Deferral OUTPUT trace allocation overflow".to_string(),
        )
    })?;
    trace_elements.checked_mul(size_of::<F>()).ok_or_else(|| {
        GpuPostflightError::InvalidTranscript(
            "Deferral OUTPUT trace byte allocation overflow".to_string(),
        )
    })?;
    rows_used
        .checked_mul(2 * DIGEST_SIZE)
        .and_then(|elements| elements.checked_mul(size_of::<F>()))
        .ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "Deferral OUTPUT Poseidon producer allocation overflow".to_string(),
            )
        })?;
    Ok((rows_used, trace_height))
}

#[cfg(feature = "rvr")]
impl DeferralOutputChipGpu {
    /// Generates OUTPUT directly from canonical program/memory logs. The only
    /// CPU-derived data is the compact call/row prefix index; no execution
    /// record or record-shaped byte buffer is materialized. The caller must
    /// pass the VM's existing padded segment trace-height limit; it is checked
    /// before allocating the main trace or Poseidon producer.
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
        max_trace_height: usize,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let device_ctx = &self.range_checker.device_ctx;
        program.ensure_replay_inputs(transcript, replay_plan, device_ctx)?;
        let step_range = replay_plan.opcode_range(DeferralOpcode::OUTPUT.global_opcode());
        if step_range.is_empty() {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        let d_row_counts = DeviceBuffer::<u32>::with_capacity_on(step_range.len(), device_ctx);
        unsafe {
            output::replay_count_rows(
                &d_row_counts,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                DeferralOpcode::OUTPUT.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                u32::try_from(self.num_deferral_circuits).map_err(|_| {
                    GpuPostflightError::InvalidTranscript(
                        "deferral circuit count exceeds u32".to_string(),
                    )
                })?,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        let counts = d_row_counts.to_host_on(device_ctx)?;
        let replay_error = transcript.error_code()?;
        if replay_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "Deferral OUTPUT row indexing rejected replay with code {replay_error}"
            )));
        }

        let mut calls = Vec::with_capacity(counts.len());
        let mut rows_used = 0u64;
        for num_rows in counts {
            if num_rows == 0 {
                return Err(GpuPostflightError::InvalidTranscript(
                    "Deferral OUTPUT replay produced an empty call".to_string(),
                ));
            }
            let row_start = u32::try_from(rows_used).map_err(|_| {
                GpuPostflightError::InvalidTranscript(
                    "Deferral OUTPUT row count exceeds u32".to_string(),
                )
            })?;
            calls.push(DeferralOutputReplayCall {
                row_start,
                num_rows,
            });
            rows_used = rows_used.checked_add(u64::from(num_rows)).ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "Deferral OUTPUT row count overflow".to_string(),
                )
            })?;
        }
        drop(d_row_counts);

        let trace_width = DeferralOutputCols::<F>::width();
        let (rows_used, trace_height) =
            checked_replay_trace_shape(rows_used, trace_width, max_trace_height)?;
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);
        trace.buffer().fill_zero_on(device_ctx)?;
        let d_calls = calls.to_device_on(device_ctx)?;
        let poseidon2 = DeferralPoseidon2ProducerBuffer::new(rows_used, device_ctx);
        unsafe {
            output::replay_tracegen(
                trace.buffer(),
                trace_height,
                trace_width,
                program.instructions(),
                program.pc_base(),
                transcript.program_log(),
                transcript.memory_log(),
                transcript.initial_write_log(),
                transcript.memory_predecessors(),
                replay_plan.steps(),
                step_range.start,
                step_range.len(),
                &d_calls,
                rows_used,
                DeferralOpcode::OUTPUT.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                self.address_bits as u32,
                &self.count,
                self.num_deferral_circuits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &self.bitwise_lookup.count,
                self.address_bits,
                &poseidon2.records,
                &poseidon2.counts,
                &poseidon2.idx,
                transcript.error_ptr(),
                device_ctx.stream.as_raw(),
            )?;
        }
        // CUDA owns these compact replay views only until the queued kernels
        // complete. They never survive trace generation into proving.
        transcript.synchronize()?;
        drop(d_calls);
        let replay_error = transcript.error_code()?;
        if replay_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "Deferral OUTPUT tracegen rejected replay with code {replay_error}"
            )));
        }
        self.poseidon2.push(poseidon2);
        Ok(AirProvingContext::simple_no_pis(trace))
    }
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralOutputChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let poseidon2_chip = deferral_poseidon2_chip::<F>();
        let mut per_call = Vec::<DeferralOutputPerCall>::new();
        let mut per_row = Vec::<DeferralOutputPerRow>::new();

        let mut offset = 0usize;
        while offset < records.len() {
            let header_offset = offset;
            let header: &DeferralOutputRecordHeader = unsafe {
                &*(records.as_ptr().add(header_offset) as *const DeferralOutputRecordHeader)
            };

            let num_rows = header.num_rows as usize;
            let output_len = (num_rows - 1) * DIGEST_SIZE;
            let write_bytes = unsafe {
                std::slice::from_raw_parts(
                    records
                        .as_ptr()
                        .add(header_offset + size_of::<DeferralOutputRecordHeader>()),
                    output_len,
                )
            };

            let mut current_poseidon2_res = [F::ZERO; DIGEST_SIZE];
            let call_idx =
                u32::try_from(per_call.len()).expect("deferral output call index should fit u32");
            let header_offset_u32 =
                u32::try_from(header_offset).expect("record byte offset should fit u32");

            for section_idx in 0..num_rows {
                let sponge_inputs = if section_idx == 0 {
                    let mut input = [F::ZERO; DIGEST_SIZE];
                    input[0] = F::from_u32(header.deferral_idx);
                    input[1] = F::from_usize(output_len);
                    input
                } else {
                    let base = (section_idx - 1) * DIGEST_SIZE;
                    from_fn(|i| F::from_u8(write_bytes[base + i]))
                };

                current_poseidon2_res = poseidon2_chip.perm(
                    &sponge_inputs,
                    &current_poseidon2_res,
                    section_idx + 1 == num_rows,
                );

                per_row.push(DeferralOutputPerRow {
                    header_offset: header_offset_u32,
                    section_idx: section_idx as u32,
                    call_idx,
                    poseidon2_res: current_poseidon2_res,
                });
            }

            per_call.push(DeferralOutputPerCall {
                output_commit: f_commit_to_bytes(&current_poseidon2_res),
            });

            let layout = DeferralOutputLayout::new(DeferralOutputMetadata { num_rows });
            let record_size =
                <DeferralOutputRecordMut<'_> as SizedRecord<DeferralOutputLayout>>::size(&layout);
            let record_alignment = <DeferralOutputRecordMut<'_> as SizedRecord<
                DeferralOutputLayout,
            >>::alignment(&layout);
            offset += record_size.next_multiple_of(record_alignment);
        }
        debug_assert_eq!(offset, records.len());

        let rows_used = per_row.len();
        let trace_height = next_power_of_two_or_zero(rows_used);
        let trace_width = DeferralOutputCols::<F>::width();
        let device_ctx = &self.range_checker.device_ctx;
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        let d_raw_records = records.to_device_on(device_ctx).unwrap();
        let d_per_call = per_call.to_device_on(device_ctx).unwrap();
        let d_per_row = per_row.to_device_on(device_ctx).unwrap();
        let poseidon2 = DeferralPoseidon2ProducerBuffer::new(rows_used, device_ctx);

        unsafe {
            output::tracegen(
                trace.buffer(),
                trace_height,
                trace_width,
                &d_raw_records,
                &d_per_call,
                &d_per_row,
                rows_used,
                &self.count,
                self.num_deferral_circuits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &self.bitwise_lookup.count,
                self.address_bits,
                &poseidon2.records,
                &poseidon2.counts,
                &poseidon2.idx,
                // Length in F elements; the CUDA side converts to record count.
                poseidon2.records.len(),
                device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral output trace");
        }
        self.poseidon2.push(poseidon2);

        AirProvingContext::simple_no_pis(trace)
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests {
    use openvm_circuit::{arch::rvr::cuda::PostflightInstruction, utils::test_gpu_engine};
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::GpuDeviceCtx,
    };
    use openvm_deferral_transpiler::DeferralOpcode;
    use openvm_instructions::{
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode,
    };
    use openvm_stark_backend::StarkEngine;
    use rvr_state::{PreflightMemoryEvent, PreflightProgramEvent};

    use super::*;

    fn output_replay_count_error(
        device_ctx: &GpuDeviceCtx,
        pc_base: u32,
        program: [PreflightProgramEvent; 2],
        program_index: u32,
    ) -> u32 {
        let opcode = DeferralOpcode::OUTPUT.global_opcode().as_usize() as u32;
        let instruction = PostflightInstruction {
            words: [opcode, 8, 16, 0, RV64_REGISTER_AS, RV64_MEMORY_AS, 0, 0],
        };
        let mut memory = [PreflightMemoryEvent::default(); 11];
        memory[6] = PreflightMemoryEvent {
            timestamp: 7,
            address_space_and_kind: RV64_MEMORY_AS,
            pointer: 0,
            value: [DIGEST_SIZE as u16, 0, 0, 0],
        };
        let steps = [[program_index, 0u32]];
        let instructions = [instruction].to_device_on(device_ctx).unwrap();
        let program = program.to_device_on(device_ctx).unwrap();
        let memory = memory.to_device_on(device_ctx).unwrap();
        let steps = steps.to_device_on(device_ctx).unwrap();
        let counts = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        let error = [0u32].to_device_on(device_ctx).unwrap();

        unsafe {
            output::replay_count_rows(
                &counts,
                instructions.view(),
                pc_base,
                program.view(),
                memory.view(),
                steps.view(),
                0,
                1,
                opcode,
                RV64_REGISTER_AS,
                RV64_MEMORY_AS,
                1,
                error.as_mut_ptr(),
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        error.to_host_on(device_ctx).unwrap()[0]
    }

    #[test]
    fn output_replay_rejects_wrapped_program_index_and_pc() {
        let engine = test_gpu_engine();
        let device_ctx = &engine.device().device_ctx;
        let ordinary_program = [
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 12,
            },
        ];
        assert_eq!(
            output_replay_count_error(device_ctx, 0, ordinary_program, u32::MAX),
            1101
        );

        let overflowing_pc = u32::MAX - 3;
        let wrapped_program = [
            PreflightProgramEvent {
                pc: overflowing_pc,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 0,
                timestamp: 12,
            },
        ];
        assert_eq!(
            output_replay_count_error(device_ctx, overflowing_pc, wrapped_program, 0),
            1101
        );
    }
}
