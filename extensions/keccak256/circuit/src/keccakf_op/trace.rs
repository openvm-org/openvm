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
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
    AlignedBytesBorrow, Chip,
};
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_riscv_circuit::adapters::{rv64_bytes_to_u32, timed_write, tracing_read};
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
    /// Kept for parity with the rest of the keccak256 extension's bus wiring; this chip
    /// no longer emits any 8-bit bitwise-lookup messages now that `buffer_ptr` is stored
    /// as u16 cells. The high-cell range check goes through `range_checker_chip` instead.
    #[allow(dead_code)]
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    pub pointer_max_bits: usize,
    pub mem_helper: SharedMemoryHelper<F>,
    // NOTE[jpw]: this is an awkward way to pass data from this execution chip to the
    // KeccakfPeriphery chip. This can be improved with a redesign of how record arenas are shared
    // with chips.
    pub shared_records: Arc<Mutex<Vec<KeccakfRecord>>>,
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
        // - RV64_MEMORY_AS (2) consists of `u8`
        // - get_slice will panic (if protected mode) if out of bounds
        let prestate =
            unsafe { guest_mem.get_slice(RV64_MEMORY_AS, record.buffer_ptr, KECCAK_WIDTH_BYTES) };
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
        let records = trace
            .par_chunks_exact_mut(width * NUM_OP_ROWS_PER_INS)
            .map(|mut row| {
                let record: &mut KeccakfRecord = unsafe {
                    get_record_from_slice(&mut row, KeccakfRecordLayout::new(KeccakfMetadata))
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

                let postimage_buffer_bytes = keccakf_postimage_bytes(&record.preimage_buffer_bytes);

                let local: &mut KeccakfOpCols<F> = row.borrow_mut();

                local.pc = F::from_u32(record.pc);
                local.is_valid = F::ONE;
                local.timestamp = F::from_u32(record.timestamp);
                local.rd_ptr = F::from_u32(record.rd_ptr);
                // Pack the low 4 bytes of `buffer_ptr` (a u32 memory address) into
                // `BUFFER_PTR_NUM_LIMBS = 2` u16 cells. The upper 4 bytes of the RV64
                // register are zero and hardcoded in the memory bus interaction via
                // `expand_to_rv64_block`.
                let ptr_bytes = record.buffer_ptr.to_le_bytes();
                local.buffer_ptr_limbs = std::array::from_fn(|i| {
                    F::from_u16(u16::from_le_bytes([ptr_bytes[2 * i], ptr_bytes[2 * i + 1]]))
                });

                // Pack consecutive pairs of state bytes into u16 cells.
                for (dst, bytes) in local
                    .preimage
                    .iter_mut()
                    .zip(record.preimage_buffer_bytes.chunks_exact(2))
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

                // Mirror the AIR's high-cell range check for `buffer_ptr`. The high
                // u16 cell covers bits [16, 32) of `buffer_ptr`; scale it by
                // `1 << (32 - pointer_max_bits)` and range-check the result to 16 bits.
                let mem_ptr_msl_lshift: u32 = (32 - self.pointer_max_bits) as u32;
                let buffer_ptr_high_u16 = record.buffer_ptr >> 16;
                self.range_checker_chip
                    .add_count(buffer_ptr_high_u16 << mem_ptr_msl_lshift, 16);
            });
        *self.shared_records.lock().unwrap() = records;
    }
}
