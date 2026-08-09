use core::convert::TryInto;
use std::{
    borrow::BorrowMut,
    mem::{align_of, size_of},
    sync::Arc,
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::MemoryReadAuxRecord, online::TracingMemory, MemoryAuxColsFactory,
        SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, AlignedBytesBorrow, Chip,
};
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_rv32im_circuit::adapters::{timed_write, tracing_read};
use openvm_stark_backend::{
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::AirProvingContext,
    StarkProtocolConfig, Val,
};

use super::{
    columns::Poseidon2PermuteOpCols,
    execution::{decompose_bytes, poseidon2_permute_bytes, recompose_words},
    Poseidon2PermuteExecutor, NUM_OP_ROWS_PER_INS,
};
use crate::{
    canonicity::CanonicityTraceGen, periphery::Poseidon2PeripheryChip, POSEIDON2_STATE_BYTES,
    POSEIDON2_WORD_SIZE,
};

#[derive(derive_new::new)]
pub struct Poseidon2PermuteChip<F: VmField> {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    // The periphery chip records the permuted states during trace generation; the periphery chip
    // must be trace-generated _after_ this adapter chip (see the extension wiring for ordering).
    pub periphery: Arc<Poseidon2PeripheryChip<F>>,
}

impl<SC, RA> Chip<RA, CpuBackend<SC>> for Poseidon2PermuteChip<Val<SC>>
where
    SC: StarkProtocolConfig,
    Val<SC>: VmField,
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
pub struct Poseidon2PermuteMetadata;

impl MultiRowMetadata for Poseidon2PermuteMetadata {
    fn get_num_rows(&self) -> usize {
        NUM_OP_ROWS_PER_INS
    }
}

pub(crate) type Poseidon2PermuteRecordLayout = MultiRowLayout<Poseidon2PermuteMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Poseidon2PermuteRecord {
    pub pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub buffer_ptr: u32,
    pub rd_aux: MemoryReadAuxRecord,
    pub buffer_word_aux: [MemoryReadAuxRecord; POSEIDON2_WIDTH],
    pub preimage_buffer_bytes: [u8; POSEIDON2_STATE_BYTES],
}

/// Mutable reference wrapper for Poseidon2PermuteRecord, used for record seeking in CUDA tests
pub struct Poseidon2PermuteRecordMut<'a> {
    pub inner: &'a mut Poseidon2PermuteRecord,
}

impl<'a> CustomBorrow<'a, Poseidon2PermuteRecordMut<'a>, Poseidon2PermuteRecordLayout> for [u8] {
    fn custom_borrow(
        &'a mut self,
        _layout: Poseidon2PermuteRecordLayout,
    ) -> Poseidon2PermuteRecordMut<'a> {
        let (record_buf, _rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Poseidon2PermuteRecord>()) };
        Poseidon2PermuteRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> Poseidon2PermuteRecordLayout {
        Poseidon2PermuteRecordLayout::new(Poseidon2PermuteMetadata)
    }
}

impl SizedRecord<Poseidon2PermuteRecordLayout> for Poseidon2PermuteRecordMut<'_> {
    fn size(_layout: &Poseidon2PermuteRecordLayout) -> usize {
        size_of::<Poseidon2PermuteRecord>()
    }

    fn alignment(_layout: &Poseidon2PermuteRecordLayout) -> usize {
        align_of::<Poseidon2PermuteRecord>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for Poseidon2PermuteExecutor
where
    F: VmField,
    for<'buf> RA: RecordArena<'buf, Poseidon2PermuteRecordLayout, &'buf mut Poseidon2PermuteRecord>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        "PERMUTE".to_string()
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { a, .. } = instruction;
        let rd_ptr = a.as_canonical_u32();

        let record = state
            .ctx
            .alloc(Poseidon2PermuteRecordLayout::new(Poseidon2PermuteMetadata));

        record.pc = *state.pc;
        record.timestamp = state.memory.timestamp();
        record.rd_ptr = rd_ptr;
        let buffer_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            rd_ptr,
            &mut record.rd_aux.prev_timestamp,
        ));
        record.buffer_ptr = buffer_ptr;

        let guest_mem = state.memory.data();
        // SAFETY:
        // - RV32_MEMORY_AS (2) consists of `u8`
        // - get_slice will panic (if protected mode) if out of bounds
        let prestate = unsafe {
            guest_mem.get_slice(RV32_MEMORY_AS, record.buffer_ptr, POSEIDON2_STATE_BYTES)
        };
        record.preimage_buffer_bytes.copy_from_slice(prestate);
        let poststate = poseidon2_permute_bytes::<F>(&record.preimage_buffer_bytes);
        for (word_idx, (word, aux)) in poststate
            .chunks_exact(POSEIDON2_WORD_SIZE)
            .zip(&mut record.buffer_word_aux)
            .enumerate()
        {
            // We don't need prev_data since we read it earlier
            let (t_prev, _) = timed_write::<POSEIDON2_WORD_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                buffer_ptr + (word_idx * POSEIDON2_WORD_SIZE) as u32,
                word.try_into().unwrap(),
            );
            aux.prev_timestamp = t_prev;
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F: VmField> TraceFiller<F> for Poseidon2PermuteChip<F> {
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
        let records = trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .map(|mut row| {
                let record: &mut Poseidon2PermuteRecord = unsafe {
                    get_record_from_slice(
                        &mut row,
                        Poseidon2PermuteRecordLayout::new(Poseidon2PermuteMetadata),
                    )
                };
                record.clone()
            })
            .collect::<Vec<_>>();
        dummy_trace.fill(F::ZERO);

        trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .zip(records.par_iter())
            .for_each(|(row, record)| {
                row.fill(F::ZERO);

                let input_words = recompose_words::<F>(&record.preimage_buffer_bytes);
                // Record the permutation so the periphery chip's trace includes this state.
                let output_words = self.periphery.perm_and_record(input_words);
                let postimage_buffer_bytes = decompose_bytes(output_words);
                let buffer_ptr_limbs = record.buffer_ptr.to_le_bytes();

                let local: &mut Poseidon2PermuteOpCols<F> = row.borrow_mut();

                local.pc = F::from_u32(record.pc);
                local.is_valid = F::ONE;
                local.timestamp = F::from_u32(record.timestamp);
                local.rd_ptr = F::from_u32(record.rd_ptr);
                local.buffer_ptr_limbs = buffer_ptr_limbs.map(F::from_u8);

                for (dst, &byte) in local.preimage.iter_mut().zip(&record.preimage_buffer_bytes) {
                    *dst = F::from_u8(byte);
                }
                for (dst, &byte) in local.postimage.iter_mut().zip(&postimage_buffer_bytes) {
                    *dst = F::from_u8(byte);
                }

                let mut timestamp = record.timestamp;
                mem_helper.fill(
                    record.rd_aux.prev_timestamp,
                    record.timestamp,
                    local.rd_aux.as_mut(),
                );
                timestamp += 1;
                for (aux, record_aux) in local
                    .buffer_word_aux
                    .iter_mut()
                    .zip(&record.buffer_word_aux)
                {
                    mem_helper.fill(record_aux.prev_timestamp, timestamp, aux);
                    timestamp += 1;
                }

                let limb_shift = 1u32
                    << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits) as u32;
                let scaled_limb =
                    (buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1] as u32) * limb_shift;
                self.bitwise_lookup_chip
                    .request_range(scaled_limb, scaled_limb);

                for pair in postimage_buffer_bytes.chunks_exact(2) {
                    self.bitwise_lookup_chip
                        .request_range(pair[0] as u32, pair[1] as u32);
                }

                // `decompose_bytes` uses `as_canonical_u32`, so every postimage word here is
                // already the canonical encoding; these aux columns prove that to the verifier.
                let canonicity_rcs: [u32; POSEIDON2_WIDTH] = std::array::from_fn(|word_idx| {
                    let start = word_idx * POSEIDON2_WORD_SIZE;
                    let word_le: [F; POSEIDON2_WORD_SIZE] = local.postimage
                        [start..start + POSEIDON2_WORD_SIZE]
                        .try_into()
                        .unwrap();
                    CanonicityTraceGen::generate_subrow(
                        &word_le,
                        &mut local.postimage_canonicity_aux[word_idx],
                    )
                });
                for pair in canonicity_rcs.chunks_exact(2) {
                    self.bitwise_lookup_chip.request_range(pair[0], pair[1]);
                }
            });
    }
}
