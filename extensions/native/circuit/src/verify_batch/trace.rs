use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::system::memory::{MemoryAuxColsFactory, OfflineMemory};
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::{instruction::Instruction, Poseidon2Opcode::COMP_POS2, VmOpcode};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::AnyRap,
    Chip, ChipUsageGetter,
};

use crate::{
    chip::{SimplePoseidonRecord, NUM_INITIAL_READS},
    verify_batch::{
        chip::{
            CellRecord, IncorporateRowRecord, IncorporateSiblingRecord, InsideRowRecord,
            VerifyBatchChip, VerifyBatchRecord,
        },
        columns::{
            InsideRowSpecificCols, SimplePoseidonSpecificCols, TopLevelSpecificCols,
            VerifyBatchCols,
        },
        CHUNK,
    },
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
        cols[..inner_width].copy_from_slice(inner_trace.values.as_slice());
    }
    #[allow(clippy::too_many_arguments)]
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
        cols.simple = F::ZERO;
        cols.end_inside_row = F::ZERO;
        cols.end_top_level = F::ZERO;
        cols.start_top_level = F::ZERO;
        cols.opened_element_size_inv = parent.opened_element_size_inv();
        cols.very_first_timestamp = F::from_canonical_u32(parent.from_state.timestamp);
        cols.start_timestamp =
            F::from_canonical_u32(read_root_is_on_right.timestamp - NUM_INITIAL_READS as u32);

        let specific: &mut TopLevelSpecificCols<F> =
            cols.specific[..TopLevelSpecificCols::<F>::width()].borrow_mut();

        specific.end_timestamp =
            F::from_canonical_usize(read_root_is_on_right.timestamp as usize + (2 + CHUNK));
        specific.reads =
            reads.map(|read| aux_cols_factory.make_read_aux_cols(memory.record_by_id(read)));
        cols.initial_opened_index = F::from_canonical_usize(opened_index);
        specific.final_opened_index = F::from_canonical_usize(opened_index - 1);
        specific.height = F::from_canonical_usize(height);
        specific.opened_length = F::from_canonical_usize(parent.opened_length);
        specific.dim_base_pointer = parent.dim_base_pointer;
        cols.opened_base_pointer = parent.opened_base_pointer;
        specific.sibling_base_pointer = parent.sibling_base_pointer;
        specific.index_base_pointer = parent.index_base_pointer;

        specific.proof_index = F::from_canonical_usize(proof_index);
        specific.read_initial_height_or_root_is_on_right =
            aux_cols_factory.make_read_aux_cols(read_root_is_on_right);
        specific.read_final_height_or_sibling_array_start =
            aux_cols_factory.make_read_aux_cols(read_sibling_array_start);
        specific.root_is_on_right = F::from_bool(root_is_on_right);
        specific.sibling_array_start = read_sibling_array_start.data[0];
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

        let specific: &mut TopLevelSpecificCols<F> =
            cols.specific[..TopLevelSpecificCols::<F>::width()].borrow_mut();

        specific.pc = F::from_canonical_u32(from_state.pc);
        specific.dim_register = instruction.a;
        specific.opened_register = instruction.b;
        specific.opened_length_register = instruction.c;
        specific.sibling_register = instruction.d;
        specific.index_register = instruction.e;
        specific.commit_register = instruction.f;
        specific.commit_pointer = commit_pointer;
        specific.dim_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(dim_base_pointer_read));
        specific.opened_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(opened_base_pointer_read));
        specific.opened_length_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(opened_length_read));
        specific.sibling_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(sibling_base_pointer_read));
        specific.index_base_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(index_base_pointer_read));
        specific.commit_pointer_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(commit_pointer_read));
        specific.commit_read =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(commit_read));
    }
    #[allow(clippy::too_many_arguments)]
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
        cols.simple = F::ZERO;
        cols.end_inside_row = F::ZERO;
        cols.end_top_level = F::ZERO;
        cols.start_top_level = F::from_bool(proof_index == 0);
        cols.opened_element_size_inv = parent.opened_element_size_inv();
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
        let specific: &mut TopLevelSpecificCols<F> =
            cols.specific[..TopLevelSpecificCols::<F>::width()].borrow_mut();

        specific.end_timestamp = F::from_canonical_u32(final_height_read.timestamp + 1);

        cols.initial_opened_index = F::from_canonical_usize(initial_opened_index);
        specific.final_opened_index = F::from_canonical_usize(final_opened_index);
        specific.height = F::from_canonical_usize(height);
        specific.opened_length = F::from_canonical_usize(parent.opened_length);
        specific.dim_base_pointer = parent.dim_base_pointer;
        cols.opened_base_pointer = parent.opened_base_pointer;
        specific.sibling_base_pointer = parent.sibling_base_pointer;
        specific.index_base_pointer = parent.index_base_pointer;

        specific.proof_index = F::from_canonical_usize(proof_index);
        specific.read_initial_height_or_root_is_on_right =
            aux_cols_factory.make_read_aux_cols(initial_height_read);
        specific.read_final_height_or_sibling_array_start =
            aux_cols_factory.make_read_aux_cols(final_height_read);
    }
    #[allow(clippy::too_many_arguments)]
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
        cols.simple = F::ZERO;
        cols.end_inside_row = F::from_bool(is_last);
        cols.end_top_level = F::ZERO;
        cols.opened_element_size_inv = grandparent.opened_element_size_inv();
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
        let specific: &mut InsideRowSpecificCols<F> =
            cols.specific[..InsideRowSpecificCols::<F>::width()].borrow_mut();

        for (record, cell) in cells.iter().zip(specific.cells.iter_mut()) {
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
        }

        for cell in specific.cells.iter_mut().skip(cells.len()) {
            cell.opened_index = F::from_canonical_usize(parent.final_opened_index);
        }

        cols.is_exhausted = std::array::from_fn(|i| F::from_bool(i >= cells.len()));

        cols.initial_opened_index = F::from_canonical_usize(parent.initial_opened_index);
        cols.opened_base_pointer = grandparent.opened_base_pointer;
    }
    // returns number of used cells
    fn verify_batch_record_to_rows(
        &self,
        record: &VerifyBatchRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
    ) -> usize {
        let width = VerifyBatchCols::<F, SBOX_REGISTERS>::width();
        let mut used_cells = 0;

        let mut height = record.initial_height;
        let mut opened_index = 0;
        for (proof_index, top_level) in record.top_level.iter().enumerate() {
            if let Some(incorporate_row) = &top_level.incorporate_row {
                self.incorporate_row_record_to_row(
                    incorporate_row,
                    aux_cols_factory,
                    &mut slice[used_cells..used_cells + width],
                    memory,
                    record,
                    proof_index,
                    height,
                );
                opened_index = incorporate_row.final_opened_index + 1;
                used_cells += width;
            }
            if let Some(incorporate_sibling) = &top_level.incorporate_sibling {
                self.incorporate_sibling_record_to_row(
                    incorporate_sibling,
                    aux_cols_factory,
                    &mut slice[used_cells..used_cells + width],
                    memory,
                    record,
                    proof_index,
                    opened_index,
                    height,
                );
                used_cells += width;
            }
            height /= 2;
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
                        chunk,
                        aux_cols_factory,
                        &mut slice[used_cells..used_cells + width],
                        memory,
                        incorporate_row,
                        record,
                        i == incorporate_row.chunks.len() - 1,
                    );
                    used_cells += width;
                }
            }
        }

        used_cells
    }
    fn simple_record_to_row(
        &self,
        record: &SimplePoseidonRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
    ) {
        let &SimplePoseidonRecord {
            from_state,
            instruction:
                Instruction {
                    opcode,
                    a: output_register,
                    b: input_register_1,
                    c: input_register_2,
                    d: register_address_space,
                    e: data_address_space,
                    ..
                },
            read_input_pointer_1,
            read_input_pointer_2,
            read_output_pointer,
            read_data_1,
            read_data_2,
            write_data_1,
            write_data_2,
            input_pointer_1,
            input_pointer_2,
            output_pointer,
            p2_input,
        } = record;

        let read_input_pointer_1 = memory.record_by_id(read_input_pointer_1);
        let read_output_pointer = memory.record_by_id(read_output_pointer);
        let read_data_1 = memory.record_by_id(read_data_1);
        let read_data_2 = memory.record_by_id(read_data_2);
        let write_data_1 = memory.record_by_id(write_data_1);

        self.generate_subair_cols(p2_input, slice);
        let cols: &mut VerifyBatchCols<F, SBOX_REGISTERS> = slice.borrow_mut();
        cols.incorporate_row = F::ZERO;
        cols.incorporate_sibling = F::ZERO;
        cols.inside_row = F::ZERO;
        cols.simple = F::ONE;
        cols.end_inside_row = F::ZERO;
        cols.end_top_level = F::ZERO;
        cols.is_exhausted = [F::ZERO; CHUNK];

        cols.start_timestamp = F::from_canonical_u32(from_state.timestamp);
        let specific: &mut SimplePoseidonSpecificCols<F> =
            cols.specific[..SimplePoseidonSpecificCols::<F>::width()].borrow_mut();

        specific.pc = F::from_canonical_u32(from_state.pc);
        specific.is_compress = F::from_bool(opcode == VmOpcode::with_default_offset(COMP_POS2));
        specific.output_register = output_register;
        specific.input_register_1 = input_register_1;
        specific.input_register_2 = input_register_2;
        specific.register_address_space = register_address_space;
        specific.data_address_space = data_address_space;
        specific.output_pointer = output_pointer;
        specific.input_pointer_1 = input_pointer_1;
        specific.input_pointer_2 = input_pointer_2;
        specific.read_output_pointer = aux_cols_factory.make_read_aux_cols(read_output_pointer);
        specific.read_input_pointer_1 = aux_cols_factory.make_read_aux_cols(read_input_pointer_1);
        specific.read_data_1 = aux_cols_factory.make_read_aux_cols(read_data_1);
        specific.read_data_2 = aux_cols_factory.make_read_aux_cols(read_data_2);
        specific.write_data_1 = aux_cols_factory.make_write_aux_cols(write_data_1);

        if opcode == VmOpcode::with_default_offset(COMP_POS2) {
            let read_input_pointer_2 = memory.record_by_id(read_input_pointer_2.unwrap());
            specific.read_input_pointer_2 =
                aux_cols_factory.make_read_aux_cols(read_input_pointer_2);
        } else {
            let write_data_2 = memory.record_by_id(write_data_2.unwrap());
            specific.write_data_2 = aux_cols_factory.make_write_aux_cols(write_data_2);
        }
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let width = self.trace_width();
        let height = next_power_of_two_or_zero(self.height);
        let mut flat_trace = F::zero_vec(width * height);

        let memory = self.offline_memory.lock().unwrap();

        let aux_cols_factory = memory.aux_cols_factory();

        let mut used_cells = 0;
        for record in self.record_set.verify_batch_records.iter() {
            used_cells += self.verify_batch_record_to_rows(
                record,
                &aux_cols_factory,
                &mut flat_trace[used_cells..],
                &memory,
            );
        }
        for record in self.record_set.simple_permute_records.iter() {
            self.simple_record_to_row(
                record,
                &aux_cols_factory,
                &mut flat_trace[used_cells..used_cells + width],
                &memory,
            );
            used_cells += width;
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