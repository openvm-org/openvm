use core::convert::TryInto;
use std::{
    borrow::BorrowMut,
    mem::{align_of, size_of},
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::MemoryReadAuxRecord, online::TracingMemory, MemoryAuxColsFactory,
        SharedMemoryHelper,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    var_range::SharedVariableRangeCheckerChip, AlignedBytesBorrow, Chip, U16_BITS,
};
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr_value, ptr_bound_from_ptr, ptr_to_field_u16_limbs, rv64_bytes_to_u16_block,
    rv64_bytes_to_u32, rv64_u16_block_to_bytes, timed_write, tracing_read, try_rv64_bytes_to_u32,
};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::AirProvingContext,
    StarkProtocolConfig, Val,
};

use super::{KeccakfExecutor, NUM_OP_ROWS_PER_INS};
use crate::{
    keccakf_op::{columns::KeccakfOpCols, keccakf_postimage_bytes},
    KECCAK_WIDTH_BYTES, KECCAK_WIDTH_MEM_OPS,
};

#[derive(derive_new::new)]
pub struct KeccakfOpChip<F> {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    pub(crate) shared_preimages: Arc<Mutex<Vec<KeccakfPreimage>>>,
}

impl<SC, RA> Chip<RA, CpuBackend<SC>> for KeccakfOpChip<Val<SC>>
where
    SC: StarkProtocolConfig,
    Val<SC>: PrimeField32,
    RA: RowMajorMatrixArena<Val<SC>>,
{
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<CpuBackend<SC>> {
        let rows_used = arena.trace_offset() / arena.width();
        let mut trace = arena.into_matrix();
        let mem_helper = self.mem_helper.as_borrowed();
        self.fill_trace(&mem_helper, &mut trace, rows_used);
        AirProvingContext::simple_no_pis(trace)
    }
}

#[derive(Clone, Copy, Default)]
pub struct KeccakfMetadata;

impl MultiRowMetadata for KeccakfMetadata {
    fn get_num_rows(&self) -> usize {
        NUM_OP_ROWS_PER_INS
    }
}

pub(crate) type KeccakfRecordLayout = MultiRowLayout<KeccakfMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct KeccakfRecord {
    pub pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub buffer_ptr: u32,
    pub rd_aux: MemoryReadAuxRecord,
    pub buffer_word_aux: [MemoryReadAuxRecord; KECCAK_WIDTH_MEM_OPS],
    pub preimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES],
}

struct KeccakfTraceInput {
    pc: u32,
    timestamp: u32,
    rd_ptr: u32,
    buffer_ptr: u32,
    rd_aux: MemoryReadAuxRecord,
    buffer_word_aux: [MemoryReadAuxRecord; KECCAK_WIDTH_MEM_OPS],
    preimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES],
}

impl From<KeccakfRecord> for KeccakfTraceInput {
    fn from(record: KeccakfRecord) -> Self {
        Self {
            pc: record.pc,
            timestamp: record.timestamp,
            rd_ptr: record.rd_ptr,
            buffer_ptr: record.buffer_ptr,
            rd_aux: record.rd_aux,
            buffer_word_aux: record.buffer_word_aux,
            preimage_buffer_bytes: record.preimage_buffer_bytes,
        }
    }
}

#[derive(Clone)]
pub(crate) struct KeccakfPreimage {
    pub timestamp: u32,
    pub bytes: [u8; KECCAK_WIDTH_BYTES],
}

/// Mutable reference wrapper for KeccakfRecord, used for record seeking in CUDA tests
pub struct KeccakfRecordMut<'a> {
    pub inner: &'a mut KeccakfRecord,
}

impl<'a> CustomBorrow<'a, KeccakfRecordMut<'a>, KeccakfRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: KeccakfRecordLayout) -> KeccakfRecordMut<'a> {
        let (record_buf, _rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<KeccakfRecord>()) };
        KeccakfRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> KeccakfRecordLayout {
        KeccakfRecordLayout::new(KeccakfMetadata)
    }
}

impl SizedRecord<KeccakfRecordLayout> for KeccakfRecordMut<'_> {
    fn size(_layout: &KeccakfRecordLayout) -> usize {
        size_of::<KeccakfRecord>()
    }

    fn alignment(_layout: &KeccakfRecordLayout) -> usize {
        align_of::<KeccakfRecord>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for KeccakfExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, KeccakfRecordLayout, &'buf mut KeccakfRecord>,
{
    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { a, .. } = instruction;
        let rd_ptr = a.as_canonical_u32();

        let record = state.ctx.alloc(KeccakfRecordLayout::new(KeccakfMetadata));

        record.pc = *state.pc;
        record.timestamp = state.memory.timestamp();
        record.rd_ptr = rd_ptr;
        let rd_val: [u8; 8] = tracing_read(
            state.memory,
            RV64_REGISTER_AS,
            rd_ptr,
            &mut record.rd_aux.prev_timestamp,
        );
        let buffer_ptr = rv64_bytes_to_u32(rd_val);
        record.buffer_ptr = buffer_ptr;

        let guest_mem = state.memory.data();
        // SAFETY:
        // - RV64_MEMORY_AS is u16-celled; `get_u8_slice` reads raw bytes at the byte pointer.
        // - `get_u8_slice` will panic (if protected mode) if out of bounds.
        let prestate = unsafe {
            guest_mem.get_u8_slice(RV64_MEMORY_AS, record.buffer_ptr, KECCAK_WIDTH_BYTES)
        };
        record.preimage_buffer_bytes.copy_from_slice(prestate);
        let poststate = keccakf_postimage_bytes(&record.preimage_buffer_bytes);
        for (word_idx, (word, aux)) in poststate
            .chunks_exact(MEMORY_BLOCK_BYTES)
            .zip(&mut record.buffer_word_aux)
            .enumerate()
        {
            // We don't need prev_data since we read it earlier
            let (t_prev, _) = timed_write::<MEMORY_BLOCK_BYTES>(
                state.memory,
                RV64_MEMORY_AS,
                buffer_ptr + (word_idx * MEMORY_BLOCK_BYTES) as u32,
                word.try_into().unwrap(),
            );
            aux.prev_timestamp = t_prev;
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for KeccakfOpChip<F> {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }
        assert!(rows_used.is_multiple_of(NUM_OP_ROWS_PER_INS));

        let width = trace_matrix.width();
        let (trace, dummy_trace) = trace_matrix.values.split_at_mut(rows_used * width);
        // For clarity we just clone the records into a separate vector to avoid dealing with unsafe
        // overwriting
        let inputs = trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .map(|mut row| {
                let record: &mut KeccakfRecord = unsafe {
                    get_record_from_slice(&mut row, KeccakfRecordLayout::new(KeccakfMetadata))
                };
                KeccakfTraceInput::from(record.clone())
            })
            .collect::<Vec<_>>();
        dummy_trace.fill(F::ZERO);
        self.fill_trace_inputs(mem_helper, trace, &inputs);
        *self.shared_preimages.lock().unwrap() = inputs
            .iter()
            .map(|input| KeccakfPreimage {
                timestamp: input.timestamp,
                bytes: input.preimage_buffer_bytes,
            })
            .collect();
    }
}

impl<F: PrimeField32> KeccakfOpChip<F> {
    fn fill_trace_inputs(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut [F],
        inputs: &[KeccakfTraceInput],
    ) {
        let width = KeccakfOpCols::<F>::width();
        trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .zip(inputs.par_iter())
            .for_each(|(row, input)| {
                row.fill(F::ZERO);

                let postimage_buffer_bytes = keccakf_postimage_bytes(&input.preimage_buffer_bytes);

                let local: &mut KeccakfOpCols<F> = row.borrow_mut();

                local.pc = F::from_u32(input.pc);
                local.is_valid = F::ONE;
                local.timestamp = F::from_u32(input.timestamp);
                local.rd_ptr = F::from_u32(input.rd_ptr);
                local.buffer_ptr_limbs = ptr_to_field_u16_limbs(input.buffer_ptr);

                // Pack consecutive pairs of state bytes into u16 cells.
                for (dst, bytes) in local
                    .preimage
                    .iter_mut()
                    .zip(input.preimage_buffer_bytes.chunks_exact(2))
                {
                    *dst = F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]]));
                }
                for (dst, bytes) in local
                    .postimage
                    .iter_mut()
                    .zip(postimage_buffer_bytes.chunks_exact(2))
                {
                    *dst = F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]]));
                }

                let mut timestamp = input.timestamp;
                mem_helper.fill(
                    input.rd_aux.prev_timestamp,
                    input.timestamp,
                    local.rd_aux.as_mut(),
                );
                timestamp += 1;
                for (aux, record_aux) in
                    local.buffer_word_aux.iter_mut().zip(&input.buffer_word_aux)
                {
                    mem_helper.fill(record_aux.prev_timestamp, timestamp, aux);
                    timestamp += 1;
                }

                self.range_checker_chip.add_count(
                    ptr_bound_from_ptr(input.buffer_ptr, self.pointer_max_bits),
                    U16_BITS,
                );
            });
    }
}

/// Generates the KeccakF operation trace directly from immutable preflight history.
pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &KeccakfOpChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(KeccakfOpcode::KECCAKF.global_opcode());
    let inputs = steps
        .par_iter()
        .map(|&step| replay_input(postflight, step, chip.pointer_max_bits))
        .collect::<Result<Vec<_>, _>>()?;
    let width = KeccakfOpCols::<F>::width();
    let height = next_power_of_two_or_zero(inputs.len() * NUM_OP_ROWS_PER_INS);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    chip.fill_trace_inputs(&chip.mem_helper.as_borrowed(), &mut trace.values, &inputs);
    *chip.shared_preimages.lock().unwrap() = inputs
        .iter()
        .map(|input| KeccakfPreimage {
            timestamp: input.timestamp,
            bytes: input.preimage_buffer_bytes,
        })
        .collect();
    Ok(trace)
}

fn replay_input<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
) -> Result<KeccakfTraceInput, PostflightError> {
    let instruction = postflight.instruction(step);
    if instruction.b != F::ZERO
        || instruction.c != F::ZERO
        || instruction.d.as_canonical_u32() != RV64_REGISTER_AS
        || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
    {
        return Err(PostflightError::new(
            "KECCAKF instruction has invalid operands",
        ));
    }

    let pc = postflight.pc(step);
    let timestamp = postflight.timestamp(step);
    let rd_ptr = instruction.a.as_canonical_u32();
    if rd_ptr & 1 != 0 {
        return Err(PostflightError::new(
            "KECCAKF register pointer must be two-byte aligned",
        ));
    }
    let mut replay = postflight.replay(step);
    let rd = replay.read_u16(RV64_REGISTER_AS, byte_ptr_to_u16_ptr_value(rd_ptr))?;
    let buffer_ptr = try_rv64_bytes_to_u32(rv64_u16_block_to_bytes(rd.value))
        .ok_or_else(|| PostflightError::new("KECCAKF buffer pointer exceeds 32 bits"))?;
    if buffer_ptr & 1 != 0 {
        return Err(PostflightError::new(
            "KECCAKF buffer pointer must be two-byte aligned",
        ));
    }
    let domain_end = if pointer_max_bits < 32 {
        1u64 << pointer_max_bits
    } else {
        1u64 << 32
    };
    if u64::from(buffer_ptr) + KECCAK_WIDTH_BYTES as u64 > domain_end {
        return Err(PostflightError::new(
            "KECCAKF state exceeds the pointer domain",
        ));
    }

    let mut preimage_buffer_bytes = [0u8; KECCAK_WIDTH_BYTES];
    for word_index in 0..KECCAK_WIDTH_MEM_OPS {
        let pointer = byte_ptr_to_u16_ptr_value(buffer_ptr) + (word_index * BLOCK_FE_WIDTH) as u32;
        let previous = replay.peek_u16(RV64_MEMORY_AS, pointer)?;
        preimage_buffer_bytes
            [word_index * MEMORY_BLOCK_BYTES..(word_index + 1) * MEMORY_BLOCK_BYTES]
            .copy_from_slice(&rv64_u16_block_to_bytes(previous));
    }
    let postimage = keccakf_postimage_bytes(&preimage_buffer_bytes);
    let mut buffer_word_aux = std::array::from_fn(|_| MemoryReadAuxRecord { prev_timestamp: 0 });
    for (word_index, bytes) in postimage.chunks_exact(MEMORY_BLOCK_BYTES).enumerate() {
        let pointer = byte_ptr_to_u16_ptr_value(buffer_ptr) + (word_index * BLOCK_FE_WIDTH) as u32;
        let access = replay.write_u16(
            RV64_MEMORY_AS,
            pointer,
            rv64_bytes_to_u16_block(bytes.try_into().expect("chunk length is fixed")),
        )?;
        buffer_word_aux[word_index].prev_timestamp = access.previous_timestamp;
        let previous_bytes = rv64_u16_block_to_bytes(access.previous_value);
        if previous_bytes.as_slice()
            != &preimage_buffer_bytes
                [word_index * MEMORY_BLOCK_BYTES..(word_index + 1) * MEMORY_BLOCK_BYTES]
        {
            return Err(PostflightError::new(
                "KECCAKF peek did not resolve the write predecessor",
            ));
        }
    }
    let next_pc = pc
        .checked_add(DEFAULT_PC_STEP)
        .ok_or_else(|| PostflightError::new("KECCAKF program counter overflow"))?;
    replay.finish(next_pc)?;

    Ok(KeccakfTraceInput {
        pc,
        timestamp,
        rd_ptr,
        buffer_ptr,
        rd_aux: MemoryReadAuxRecord {
            prev_timestamp: rd.previous_timestamp,
        },
        buffer_word_aux,
        preimage_buffer_bytes,
    })
}
