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
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, AlignedBytesBorrow,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_rv32im_circuit::adapters::{timed_write, tracing_read};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    Chip,
};

use super::{KeccakfExecutor, NUM_OP_ROWS_PER_INS};
use crate::{
    keccakf_op::{columns::KeccakfOpCols, keccakf_postimage_bytes},
    KECCAK_WIDTH_BYTES, KECCAK_WIDTH_WORDS, KECCAK_WORD_SIZE,
};

#[derive(derive_new::new)]
pub struct KeccakfOpChip<F> {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    // NOTE[jpw]: this is an awkward way to pass data from this execution chip to the
    // KeccakfPeriphery chip. This can be improved with a redesign of how record arenas are shared
    // with chips.
    pub shared_records: Arc<Mutex<Vec<KeccakfRecord>>>,
}

impl<SC, RA> Chip<RA, CpuBackend<SC>> for KeccakfOpChip<Val<SC>>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    RA: RowMajorMatrixArena<Val<SC>>,
{
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<CpuBackend<SC>> {
        let rows_used = arena.trace_offset() / arena.width();
        let mut trace = arena.into_matrix();
        let mem_helper = self.mem_helper.as_borrowed();
        self.fill_trace(&mem_helper, &mut trace, rows_used);
        AirProvingContext::simple_no_pis(Arc::new(trace))
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
    pub buffer_word_aux: [MemoryReadAuxRecord; KECCAK_WIDTH_WORDS],
    pub preimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES],
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
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", KeccakfOpcode::KECCAKF)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { a, .. } = instruction;
        let rd_ptr = a.as_canonical_u32();

        let record = state.ctx.alloc(KeccakfRecordLayout::new(KeccakfMetadata));

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
        let prestate =
            unsafe { guest_mem.get_slice(RV32_MEMORY_AS, record.buffer_ptr, KECCAK_WIDTH_BYTES) };
        record.preimage_buffer_bytes.copy_from_slice(prestate);
        let poststate = keccakf_postimage_bytes(&record.preimage_buffer_bytes);
        for (word_idx, (word, aux)) in poststate
            .chunks_exact(KECCAK_WORD_SIZE)
            .zip(&mut record.buffer_word_aux)
            .enumerate()
        {
            // We don't need prev_data since we read it earlier
            let (t_prev, _) = timed_write::<KECCAK_WORD_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                buffer_ptr + (word_idx * KECCAK_WORD_SIZE) as u32,
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
        let records = trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .map(|mut double_rows| {
                let record: &mut KeccakfRecord = unsafe {
                    get_record_from_slice(
                        &mut double_rows,
                        KeccakfRecordLayout::new(KeccakfMetadata),
                    )
                };
                record.clone()
            })
            .collect::<Vec<_>>();
        dummy_trace.fill(F::ZERO);

        trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .zip(records.par_iter())
            .for_each(|(double_rows, record)| {
                double_rows.fill(F::ZERO);

                let postimage_buffer_bytes = keccakf_postimage_bytes(&record.preimage_buffer_bytes);
                let buffer_ptr_limbs = record.buffer_ptr.to_le_bytes();

                let (local_row, next_row) = double_rows.split_at_mut(width);
                let local: &mut KeccakfOpCols<F> = local_row.borrow_mut();
                let next: &mut KeccakfOpCols<F> = next_row.borrow_mut();

                local.pc = F::from_canonical_u32(record.pc);
                local.is_valid = F::ONE;
                local.timestamp = F::from_canonical_u32(record.timestamp);
                local.rd_ptr = F::from_canonical_u32(record.rd_ptr);
                local.buffer_ptr_limbs = buffer_ptr_limbs.map(F::from_canonical_u8);
                next.is_after_valid = F::ONE;
                next.timestamp = local.timestamp;

                for (dst, &byte) in local.buffer.iter_mut().zip(&record.preimage_buffer_bytes) {
                    *dst = F::from_canonical_u8(byte);
                }
                for (dst, &byte) in next.buffer.iter_mut().zip(&postimage_buffer_bytes) {
                    *dst = F::from_canonical_u8(byte);
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
            });
        *self.shared_records.lock().unwrap() = records;
    }
}
