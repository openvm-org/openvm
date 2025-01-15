use std::{borrow::BorrowMut, sync::Arc};

use itertools::Itertools;
use openvm_circuit::system::memory::{MemoryAuxColsFactory, OfflineMemory};
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    prover::types::AirProofInput,
    rap::AnyRap,
    Chip, ChipUsageGetter,
};
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use crate::chip::NUM_INITIAL_READS;
use crate::verify_batch::{
    chip::{
        CellRecord, IncorporateRowRecord, IncorporateSiblingRecord, InsideRowRecord,
        VerifyBatchChip, VerifyBatchRecord,
    },
    columns::VerifyBatchCols,
    CHUNK,
};

impl<F: Field, const SBOX_REGISTERS: usize> ChipUsageGetter for VerifyBatchChip<F, SBOX_REGISTERS> {
    fn air_name(&self) -> String {
        "VerifyBatchAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.height
    }

    fn trace_width(&self) -> usize {
        VerifyBatchCols::<F, SBOX_REGISTERS>::width()
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> VerifyBatchChip<F, SBOX_REGISTERS> {
    fn generate_subair_cols(&self, input: [F; 2 * CHUNK], cols: &mut [F]) {
        let inner_trace = self.subchip.generate_trace(vec![input]);
        let inner_width = self.air.subair.width();
        cols[..inner_width].copy_from_slice(&inner_trace.values.as_slice());
    }
    fn incorporate_sibling_record_to_row(
        &self,
        record: &IncorporateSiblingRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
        parent: &VerifyBatchRecord<F>,
        proof_index: usize,
        opened_index: usize,
        height: usize,
    ) {
        let &IncorporateSiblingRecord {
            read_sibling_array_start,
            read_root_is_on_right,
            root_is_on_right,
            reads,
            p2_input,
        } = record;

        let read_root_is_on_right = memory.record_by_id(read_root_is_on_right);
        let read_sibling_array_start = memory.record_by_id(read_sibling_array_start);

        self.generate_subair_cols(p2_input, slice);
        let cols: &mut VerifyBatchCols<F, SBOX_REGISTERS> = slice.borrow_mut();
        cols.incorporate_row = F::ZERO;
        cols.incorporate_sibling = F::ONE;
        cols.inside_row = F::ZERO;
        cols.end_inside_row = F::ZERO;
        cols.end_top_level = F::ZERO;
        cols.start_top_level = F::ZERO;
        cols.very_first_timestamp = F::from_canonical_u32(parent.from_state.timestamp);
        cols.start_timestamp = F::from_canonical_u32(read_root_is_on_right.timestamp - NUM_INITIAL_READS as u32);
        cols.end_timestamp =
            F::from_canonical_usize(read_root_is_on_right.timestamp as usize + (2 + CHUNK));
        cols.address_space = F::from_canonical_usize(parent.address_space());
        for (read, cell) in reads.into_iter().zip_eq(cols.cells.iter_mut()) {
            cell.read = aux_cols_factory.make_read_aux_cols(memory.record_by_id(read));
        }
        cols.initial_opened_index = F::from_canonical_usize(opened_index);
        cols.final_opened_index = F::from_canonical_usize(opened_index - 1);
        cols.height = F::from_canonical_usize(height);
        cols.opened_length = F::from_canonical_usize(parent.opened_length);
        cols.dim_base_pointer = parent.dim_base_pointer;
        cols.opened_base_pointer = parent.opened_base_pointer;
        cols.sibling_base_pointer = parent.sibling_base_pointer;
        cols.index_base_pointer = parent.index_base_pointer;

        cols.proof_index = F::from_canonical_usize(proof_index);
        cols.read_initial_height_or_root_is_on_right =
            aux_cols_factory.make_read_aux_cols(read_root_is_on_right);
        cols.read_final_height_or_sibling_array_start =
            aux_cols_factory.make_read_aux_cols(read_sibling_array_start);
        cols.root_is_on_right = F::from_bool(root_is_on_right);
        cols.sibling_array_start = read_sibling_array_start.data[0];
    }
    fn correct_last_top_level_row(
        &self,
        record: &VerifyBatchRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
    ) {
        let &VerifyBatchRecord {
            from_state,
            commit_pointer,
            dim_base_pointer_read,
            opened_base_pointer_read,
            opened_length_read,
            sibling_base_pointer_read,
            index_base_pointer_read,
            commit_pointer_read,
            commit_read,
            ..
        } = record;
        let instruction = &record.instruction;
        let cols: &mut VerifyBatchCols<F, SBOX_REGISTERS> = slice.borrow_mut();
        cols.end_top_level = F::ONE;
        cols.pc = F::from_canonical_u32(from_state.pc);
        cols.dim_register = instruction.a;
        cols.opened_register = instruction.b;
        cols.opened_length_register = instruction.c;
        cols.sibling_register = instruction.d;
        cols.index_register = instruction.e;
        cols.commit_register = instruction.f;
        cols.commit_pointer = commit_pointer;
        cols.dim_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(dim_base_pointer_read));
        cols.opened_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(opened_base_pointer_read));
        cols.opened_length_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(opened_length_read));
        cols.sibling_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(sibling_base_pointer_read));
        cols.index_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(index_base_pointer_read));
        cols.commit_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(commit_pointer_read));
        cols.commit_read = aux_cols_factory.make_read_aux_cols(memory.record_by_id(commit_read));
    }
    fn incorporate_row_record_to_row(
        &self,
        record: &IncorporateRowRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
        parent: &VerifyBatchRecord<F>,
        proof_index: usize,
        height: usize,
    ) {
        let &IncorporateRowRecord {
            initial_opened_index,
            final_opened_index,
            initial_height_read,
            final_height_read,
            p2_input,
            ..
        } = record;

        let initial_height_read = memory.record_by_id(initial_height_read);
        let final_height_read = memory.record_by_id(final_height_read);

        self.generate_subair_cols(p2_input, slice);
        let cols: &mut VerifyBatchCols<F, SBOX_REGISTERS> = slice.borrow_mut();
        cols.incorporate_row = F::ONE;
        cols.incorporate_sibling = F::ZERO;
        cols.inside_row = F::ZERO;
        cols.end_inside_row = F::ZERO;
        cols.end_top_level = F::ZERO;
        cols.start_top_level = F::from_bool(proof_index == 0);
        cols.very_first_timestamp = F::from_canonical_u32(parent.from_state.timestamp);
        cols.start_timestamp = F::from_canonical_u32(
            memory
                .record_by_id(
                    record.chunks[0].cells[0]
                        .read_row_pointer_and_length
                        .unwrap(),
                )
                .timestamp
                - NUM_INITIAL_READS as u32,
        );
        cols.end_timestamp = F::from_canonical_u32(final_height_read.timestamp + 1);
        cols.address_space = F::from_canonical_usize(parent.address_space());

        cols.initial_opened_index = F::from_canonical_usize(initial_opened_index);
        cols.final_opened_index = F::from_canonical_usize(final_opened_index);
        cols.height = F::from_canonical_usize(height);
        cols.opened_length = F::from_canonical_usize(parent.opened_length);
        cols.dim_base_pointer = parent.dim_base_pointer;
        cols.opened_base_pointer = parent.opened_base_pointer;
        cols.sibling_base_pointer = parent.sibling_base_pointer;
        cols.index_base_pointer = parent.index_base_pointer;

        cols.proof_index = F::from_canonical_usize(proof_index);
        cols.read_initial_height_or_root_is_on_right =
            aux_cols_factory.make_read_aux_cols(initial_height_read);
        cols.read_final_height_or_sibling_array_start =
            aux_cols_factory.make_read_aux_cols(final_height_read);
    }
    fn inside_row_record_to_row(
        &self,
        record: &InsideRowRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
        parent: &IncorporateRowRecord<F>,
        grandparent: &VerifyBatchRecord<F>,
        is_last: bool,
    ) {
        let InsideRowRecord { cells, p2_input } = record;

        self.generate_subair_cols(*p2_input, slice);
        let cols: &mut VerifyBatchCols<F, SBOX_REGISTERS> = slice.borrow_mut();
        cols.incorporate_row = F::ZERO;
        cols.incorporate_sibling = F::ZERO;
        cols.inside_row = F::ONE;
        cols.end_inside_row = F::from_bool(is_last);
        cols.end_top_level = F::ZERO;
        cols.very_first_timestamp = F::from_canonical_u32(
            memory
                .record_by_id(
                    parent.chunks[0].cells[0]
                        .read_row_pointer_and_length
                        .unwrap(),
                )
                .timestamp,
        );
        cols.start_timestamp =
            F::from_canonical_u32(memory.record_by_id(cells[0].read).timestamp - 1);
        cols.address_space = F::from_canonical_usize(grandparent.address_space());

        for (record, cell) in cells.iter().zip(cols.cells.iter_mut()) {
            let &CellRecord {
                read,
                opened_index,
                read_row_pointer_and_length,
                row_pointer,
                row_end,
            } = record;
            cell.read = aux_cols_factory.make_read_aux_cols(memory.record_by_id(read));
            cell.opened_index = F::from_canonical_usize(opened_index);
            if let Some(read_row_pointer_and_length) = read_row_pointer_and_length {
                cell.read_row_pointer_and_length = aux_cols_factory
                    .make_read_aux_cols(memory.record_by_id(read_row_pointer_and_length));
            }
            cell.row_pointer = F::from_canonical_usize(row_pointer);
            cell.row_end = F::from_canonical_usize(row_end);
            cell.is_first_in_row = F::from_bool(read_row_pointer_and_length.is_some());
            cell.is_exhausted = F::ZERO;
        }
        for cell in cols.cells.iter_mut().skip(cells.len()) {
            cell.is_exhausted = F::ONE;
            cell.opened_index = F::from_canonical_usize(parent.final_opened_index);
        }

        cols.initial_opened_index = F::from_canonical_usize(parent.initial_opened_index);
        cols.opened_base_pointer = grandparent.opened_base_pointer;
    }
    // returns number of used cells
    fn record_to_rows(
        &self,
        record: &VerifyBatchRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
    ) -> usize {
        let width = VerifyBatchCols::<F, SBOX_REGISTERS>::width();
        let mut used_cells = 0;

        let mut proof_index = 0;
        let mut height = record.initial_height;
        let mut opened_index = 0;
        for top_level in record.top_level.iter() {
            if let Some(incorporate_row) = &top_level.incorporate_row {
                self.incorporate_row_record_to_row(
                    &incorporate_row,
                    aux_cols_factory,
                    &mut slice[used_cells..used_cells + width],
                    memory,
                    &record,
                    proof_index,
                    height,
                );
                opened_index = incorporate_row.final_opened_index + 1;
                used_cells += width;
            }
            if let Some(incorporate_sibling) = &top_level.incorporate_sibling {
                self.incorporate_sibling_record_to_row(
                    &incorporate_sibling,
                    aux_cols_factory,
                    &mut slice[used_cells..used_cells + width],
                    memory,
                    &record,
                    proof_index,
                    opened_index,
                    height,
                );
                used_cells += width;
            }
            height /= 2;
            proof_index += 1;
        }
        self.correct_last_top_level_row(
            record,
            aux_cols_factory,
            &mut slice[used_cells - width..used_cells],
            memory,
        );

        for top_level in record.top_level.iter() {
            if let Some(incorporate_row) = &top_level.incorporate_row {
                for (i, chunk) in incorporate_row.chunks.iter().enumerate() {
                    self.inside_row_record_to_row(
                        &chunk,
                        aux_cols_factory,
                        &mut slice[used_cells..used_cells + width],
                        memory,
                        &incorporate_row,
                        &record,
                        i == incorporate_row.chunks.len() - 1,
                    );
                    used_cells += width;
                }
            }
        }

        used_cells
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let width = self.trace_width();
        let height = next_power_of_two_or_zero(self.height);
        let mut flat_trace = F::zero_vec(width * height);

        let memory = self.offline_memory.lock().unwrap();

        let aux_cols_factory = memory.aux_cols_factory();

        let mut used_cells = 0;
        for record in self.records.iter() {
            used_cells += self.record_to_rows(
                &record,
                &aux_cols_factory,
                &mut flat_trace[used_cells..],
                &memory,
            );
        }
        // poseidon2 constraints are always checked
        // following can be optimized to only hash [0; _] once
        flat_trace[used_cells..]
            .par_chunks_mut(width)
            .for_each(|row| {
                self.generate_subair_cols([F::ZERO; 2 * CHUNK], row);
            });

        RowMajorMatrix::new(flat_trace, width)
    }
}

impl<SC: StarkGenericConfig, const SBOX_REGISTERS: usize> Chip<SC>
    for VerifyBatchChip<Val<SC>, SBOX_REGISTERS>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }
    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        AirProofInput::simple_no_pis(self.air(), self.generate_trace())
    }
}
