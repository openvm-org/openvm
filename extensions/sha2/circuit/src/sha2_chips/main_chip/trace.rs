use std::{
    array::{self, from_fn},
    borrow::{Borrow, BorrowMut},
    cmp::min,
    sync::Arc,
};

use ndarray::ArrayViewMut;
use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{read_rv32_register, tracing_read, tracing_write};
use openvm_sha2_air::set_arrayview_from_u8_slice;
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    Chip,
};

use crate::{
    Sha2ColsRef, Sha2ColsRefMut, Sha2Config, Sha2MainChip, Sha2Metadata, Sha2RecordHeader,
    Sha2RecordLayout, Sha2RecordMut, Sha2SharedRecords, SHA2_WRITE_SIZE,
};

// We will allocate a new trace matrix instead of using the record arena directly,
// because we want to pass the record arena to Sha2BlockHasherChip when we are done.
impl<RA, SC: StarkGenericConfig, C: Sha2Config> Chip<RA, CpuBackend<SC>>
    for Sha2MainChip<Val<SC>, C>
where
    Val<SC>: PrimeField32,
    SC: StarkGenericConfig,
    RA: RowMajorMatrixArena<Val<SC>> + Send + Sync,
{
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<CpuBackend<SC>> {
        // Since Sha2Metadata::get_num_rows() = 1, the number of rows used is equal to the number of
        // SHA-2 instructions executed.
        let rows_used = arena.trace_offset() / arena.width();

        // We will fill the trace into a separate buffer, because we want to pass the arena to the
        // Sha2BlockHasherChip when we are done.
        // Sha2MainChip uses 1 row per instruction, we allocate rows_used * arena.width() space for
        // the trace.
        let height = next_power_of_two_or_zero(rows_used);
        let trace = Val::<SC>::zero_vec(height * arena.width());
        let mut trace_matrix = RowMajorMatrix::new(trace, arena.width());
        let mem_helper = self.mem_helper.as_borrowed();

        let mut records = arena.into_matrix();

        self.fill_trace(&mem_helper, &mut trace_matrix, rows_used, &mut records);

        // Pass the records to Sha2BlockHasherChip
        *self.records.lock().unwrap() = Some(Sha2SharedRecords {
            num_records: rows_used,
            matrix: records,
        });

        AirProvingContext::simple(Arc::new(trace_matrix), vec![])
    }
}

// Note: we would like to just impl TraceFiller here, but we can't because we need to pass the
// records and row_idx to the tracegen functions.
impl<F: PrimeField32, C: Sha2Config> Sha2MainChip<F, C> {
    // Preconditions:
    // - trace should be a matrix with width = Sha2MainAir::width() and height = rows_used
    // - trace should be filled with all zeros
    // - records should be a matrix with height = rows_used, where each row stores a record
    pub fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
        records: &mut RowMajorMatrix<F>,
    ) {
        let width = trace.width();
        trace.values[..rows_used * width]
            .par_chunks_exact_mut(width)
            .zip(records.par_rows_mut())
            .enumerate()
            .for_each(|(row_idx, (row_slice, record))| {
                self.fill_trace_row_with_row_idx(mem_helper, row_slice, row_idx, record);
            });
    }

    // Same as TraceFiller::fill_trace_row, except we also take the row index as a parameter.
    //
    // Note: the only reason the record parameter is mutable is that get_record_from_slice
    // requires a &mut &mut [F] slice. This parameter type is useful in other places where
    // get_record_from_slice is used, to circumvent the borrow checker. Here, we don't actually need
    // this workaround (we could duplicate get_record_from_slice and modify it to take a &mut
    // [F] slice), but we just use the existing function for simplicity.
    fn fill_trace_row_with_row_idx(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        row_slice: &mut [F],
        row_idx: usize,
        mut record: &mut [F],
    ) where
        F: Clone,
    {
        // SAFETY:
        // - caller ensures `record` contains a valid record representation that was previously
        //   written by the executor
        // - record contains a valid Sha2RecordMut with the exact layout specified
        // - get_record_from_slice will correctly split the buffer into header and other components
        //   based on this layout.
        let record: Sha2RecordMut = unsafe {
            get_record_from_slice(
                &mut record,
                Sha2RecordLayout::new(Sha2Metadata {
                    variant: C::VARIANT,
                }),
            )
        };

        // save all the components of the record on the stack so that we don't overwrite them when
        // filling in the trace matrix.
        let vm_record = record.inner.clone();

        let mut message_bytes = Vec::with_capacity(C::BLOCK_BYTES);
        message_bytes.extend_from_slice(record.message_bytes);

        let mut prev_state = Vec::with_capacity(C::STATE_BYTES);
        prev_state.extend_from_slice(record.prev_state);

        let mut new_state = Vec::with_capacity(C::STATE_BYTES);
        new_state.extend_from_slice(record.new_state);

        let mut input_reads_aux =
            Vec::with_capacity(C::BLOCK_READS * size_of::<MemoryReadAuxRecord>());
        input_reads_aux.extend_from_slice(record.input_reads_aux);

        let mut state_reads_aux =
            Vec::with_capacity(C::STATE_READS * size_of::<MemoryReadAuxRecord>());
        state_reads_aux.extend_from_slice(record.state_reads_aux);

        let mut write_aux = Vec::with_capacity(
            C::STATE_WRITES * size_of::<MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>>(),
        );
        write_aux.extend_from_slice(record.write_aux);

        let mut cols = Sha2ColsRefMut::from::<C>(row_slice);

        *cols.block.request_id = F::from_canonical_usize(row_idx);
        set_arrayview_from_u8_slice(&mut cols.block.message_bytes, message_bytes);
        set_arrayview_from_u8_slice(&mut cols.block.prev_state, prev_state);
        set_arrayview_from_u8_slice(&mut cols.block.new_state, new_state);

        *cols.instruction.is_enabled = F::ONE;
        cols.instruction.from_state.timestamp = F::from_canonical_u32(vm_record.timestamp);
        cols.instruction.from_state.pc = F::from_canonical_u32(vm_record.from_pc);
        *cols.instruction.dst_reg_ptr = F::from_canonical_u32(vm_record.dst_reg_ptr);
        *cols.instruction.state_reg_ptr = F::from_canonical_u32(vm_record.state_reg_ptr);
        *cols.instruction.input_reg_ptr = F::from_canonical_u32(vm_record.input_reg_ptr);

        let dst_ptr_limbs = vm_record.dst_ptr.to_le_bytes();
        let state_ptr_limbs = vm_record.state_ptr.to_le_bytes();
        let input_ptr_limbs = vm_record.input_ptr.to_le_bytes();
        set_arrayview_from_u8_slice(&mut cols.instruction.dst_ptr_limbs, dst_ptr_limbs);
        set_arrayview_from_u8_slice(&mut cols.instruction.state_ptr_limbs, state_ptr_limbs);
        set_arrayview_from_u8_slice(&mut cols.instruction.input_ptr_limbs, input_ptr_limbs);
        let needs_range_check = [
            dst_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            state_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            input_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            input_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
        ];
        let shift: u32 = 1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits);
        for pair in needs_range_check.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] as u32 * shift, pair[1] as u32 * shift);
        }

        // fill in the register reads aux
        let mut timestamp = vm_record.timestamp;
        for (cols, vm_record) in cols
            .mem
            .register_aux
            .iter_mut()
            .zip(vm_record.register_reads_aux.iter())
        {
            mem_helper.fill(vm_record.prev_timestamp, timestamp, cols.as_mut());
            timestamp += 1;
        }

        input_reads_aux.iter().zip(cols.mem.input_reads).for_each(
            |(read_aux_record, read_aux_cols)| {
                mem_helper.fill(
                    read_aux_record.prev_timestamp,
                    timestamp,
                    read_aux_cols.as_mut(),
                );
                timestamp += 1;
            },
        );

        state_reads_aux.iter().zip(cols.mem.state_reads).for_each(
            |(state_aux_record, state_aux_cols)| {
                mem_helper.fill(
                    state_aux_record.prev_timestamp,
                    timestamp,
                    state_aux_cols.as_mut(),
                );
                timestamp += 1;
            },
        );

        write_aux
            .iter()
            .zip(cols.mem.write_aux)
            .for_each(|(write_aux_record, write_aux_cols)| {
                write_aux_cols.set_prev_data(write_aux_record.prev_data.map(F::from_canonical_u8));
                mem_helper.fill(
                    write_aux_record.prev_timestamp,
                    timestamp,
                    write_aux_cols.as_mut(),
                );
                timestamp += 1;
            });
    }
}
