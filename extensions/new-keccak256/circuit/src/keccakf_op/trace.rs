use core::convert::TryInto;
use std::{
    borrow::BorrowMut,
    mem::{align_of, size_of},
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_rv32im_circuit::adapters::{memory_read, tracing_read, tracing_write};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{KeccakfVmExecutor, KeccakfVmFiller, NUM_KECCAKF_OP_ROWS};
use crate::{
    keccakf_op::columns::KeccakfOpCols,
    KECCAK_WIDTH_BYTES, KECCAK_WIDTH_U64S, KECCAK_WORD_SIZE,
};

const KECCAK_WIDTH_U32_LIMBS: usize = KECCAK_WIDTH_BYTES / KECCAK_WORD_SIZE;

#[derive(Clone, Copy)]
pub struct KeccakfVmMetadata {}

impl MultiRowMetadata for KeccakfVmMetadata {
    fn get_num_rows(&self) -> usize {
        NUM_KECCAKF_OP_ROWS
    }
}

pub(crate) type KeccakfVmRecordLayout = MultiRowLayout<KeccakfVmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct KeccakfVmRecordHeader {
    pub pc: u32,
    pub timestamp: u32,
    pub buffer: u32,
    pub preimage_buffer_bytes: [u8; KECCAK_WIDTH_BYTES],
    pub rd_ptr: u32,
    pub register_aux_cols: [MemoryReadAuxRecord; 1],
    pub buffer_read_aux_cols: [MemoryReadAuxRecord; KECCAK_WIDTH_U32_LIMBS],
    pub buffer_write_aux_cols: [MemoryWriteBytesAuxRecord<KECCAK_WORD_SIZE>;
        KECCAK_WIDTH_U32_LIMBS],
}

pub struct KeccakfVmRecordMut<'a> {
    pub inner: &'a mut KeccakfVmRecordHeader,
}

impl<'a> CustomBorrow<'a, KeccakfVmRecordMut<'a>, KeccakfVmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: KeccakfVmRecordLayout) -> KeccakfVmRecordMut<'a> {
        let (record_buf, _rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<KeccakfVmRecordHeader>()) };
        KeccakfVmRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> KeccakfVmRecordLayout {
        KeccakfVmRecordLayout {
            metadata: KeccakfVmMetadata {},
        }
    }
}

impl SizedRecord<KeccakfVmRecordLayout> for KeccakfVmRecordMut<'_> {
    fn size(_layout: &KeccakfVmRecordLayout) -> usize {
        size_of::<KeccakfVmRecordHeader>()
    }

    fn alignment(_layout: &KeccakfVmRecordLayout) -> usize {
        align_of::<KeccakfVmRecordHeader>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for KeccakfVmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, KeccakfVmRecordLayout, KeccakfVmRecordMut<'buf>>,
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

        let record = state
            .ctx
            .alloc(KeccakfVmRecordLayout::new(KeccakfVmMetadata {}));

        record.inner.pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.buffer = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            a.as_canonical_u32(),
            &mut record.inner.register_aux_cols[0].prev_timestamp,
        ));

        let guest_mem = state.memory.data();
        for idx in 0..KECCAK_WIDTH_U32_LIMBS {
            let read = memory_read::<KECCAK_WORD_SIZE>(
                guest_mem,
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * KECCAK_WORD_SIZE) as u32,
            );
            record.inner.preimage_buffer_bytes
                [KECCAK_WORD_SIZE * idx..KECCAK_WORD_SIZE * (idx + 1)]
                .copy_from_slice(&read);
        }

        let postimage_buffer_bytes = keccakf_postimage_bytes(&record.inner.preimage_buffer_bytes);
        for idx in 0..KECCAK_WIDTH_U32_LIMBS {
            let chunk: [u8; KECCAK_WORD_SIZE] = postimage_buffer_bytes
                [KECCAK_WORD_SIZE * idx..KECCAK_WORD_SIZE * (idx + 1)]
                .try_into()
                .unwrap();
            tracing_write(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * KECCAK_WORD_SIZE) as u32,
                chunk,
                &mut record.inner.buffer_write_aux_cols[idx].prev_timestamp,
                &mut record.inner.buffer_write_aux_cols[idx].prev_data,
            );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for KeccakfVmFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace_matrix.width();
        let (trace, dummy_trace) = trace_matrix
            .values
            .split_at_mut(rows_used * width);
        dummy_trace.fill(F::ZERO);

        trace
            .chunks_exact_mut(width * NUM_KECCAKF_OP_ROWS)
            .for_each(|record_slice| {
                let record = {
                    let mut record_slice = record_slice;
                    let record: KeccakfVmRecordMut = unsafe {
                        get_record_from_slice(
                            &mut record_slice,
                            KeccakfVmRecordLayout::new(KeccakfVmMetadata {}),
                        )
                    };
                    record.inner.clone()
                };

                let postimage_buffer_bytes =
                    keccakf_postimage_bytes(&record.preimage_buffer_bytes);
                let buffer_ptr_limbs = record.buffer.to_le_bytes();
                let buffer_ptr_limbs_f = buffer_ptr_limbs.map(F::from_canonical_u8);

                let (read_row, write_row) = record_slice.split_at_mut(width);
                read_row.fill(F::ZERO);
                write_row.fill(F::ZERO);

                let read_cols: &mut KeccakfOpCols<F> = read_row.borrow_mut();
                read_cols.pc = F::from_canonical_u32(record.pc);
                read_cols.is_valid = F::ONE;
                read_cols.is_after_valid = F::ZERO;
                read_cols.timestamp = F::from_canonical_u32(record.timestamp);
                read_cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
                read_cols.buffer_ptr_limbs = buffer_ptr_limbs_f;
                for (dst, &byte) in read_cols
                    .buffer
                    .iter_mut()
                    .zip(record.preimage_buffer_bytes.iter())
                {
                    *dst = F::from_canonical_u8(byte);
                }

                mem_helper.fill(
                    record.register_aux_cols[0].prev_timestamp,
                    record.timestamp,
                    read_cols.rd_aux.as_mut(),
                );
                let mut timestamp = record.timestamp + 1;
                for (aux, write_aux) in read_cols
                    .buffer_word_aux
                    .iter_mut()
                    .zip(record.buffer_write_aux_cols.iter())
                {
                    mem_helper.fill(write_aux.prev_timestamp, timestamp, aux);
                    timestamp += 1;
                }

                let write_cols: &mut KeccakfOpCols<F> = write_row.borrow_mut();
                write_cols.pc = F::from_canonical_u32(record.pc);
                write_cols.is_valid = F::ZERO;
                write_cols.is_after_valid = F::ONE;
                write_cols.timestamp = F::from_canonical_u32(record.timestamp);
                write_cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
                write_cols.buffer_ptr_limbs = buffer_ptr_limbs.map(F::from_canonical_u8);
                for (dst, &byte) in write_cols
                    .buffer
                    .iter_mut()
                    .zip(postimage_buffer_bytes.iter())
                {
                    *dst = F::from_canonical_u8(byte);
                }

                let limb_shift = 1u32
                    << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits) as u32;
                let scaled_limb = (buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1] as u32)
                    * limb_shift;
                self.bitwise_lookup_chip
                    .request_range(scaled_limb, scaled_limb);

                for pair in postimage_buffer_bytes.chunks_exact(2) {
                    self.bitwise_lookup_chip
                        .request_range(pair[0] as u32, pair[1] as u32);
                }
            });
    }
}

fn keccakf_postimage_bytes(
    preimage_buffer_bytes: &[u8; KECCAK_WIDTH_BYTES],
) -> [u8; KECCAK_WIDTH_BYTES] {
    let mut state = [0u64; KECCAK_WIDTH_U64S];
    for (idx, chunk) in preimage_buffer_bytes.chunks_exact(8).enumerate() {
        state[idx] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    tiny_keccak::keccakf(&mut state);

    let mut result = [0u8; KECCAK_WIDTH_BYTES];
    for (idx, word) in state.into_iter().enumerate() {
        let bytes = word.to_le_bytes();
        result[8 * idx..8 * idx + 8].copy_from_slice(&bytes);
    }
    result
}
