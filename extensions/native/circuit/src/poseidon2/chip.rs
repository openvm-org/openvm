use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::{
        memory::{offline_checker::MemoryBaseAuxCols, online::TracingMemory, MemoryAuxColsFactory},
        native_adapter::util::{
            memory_read_native, tracing_read_native, tracing_write_native_inplace,
        },
    },
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS,
    Poseidon2Opcode::{COMP_POS2, PERM_POS2},
    VerifyBatchOpcode::VERIFY_BATCH,
};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip, Poseidon2SubCols};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelSliceMut, *},
};

use crate::poseidon2::{
    columns::{
        InsideRowSpecificCols, NativePoseidon2Cols, SimplePoseidonSpecificCols,
        TopLevelSpecificCols,
    },
    CHUNK,
};

#[derive(Clone)]
pub struct NativePoseidon2Executor<F: Field, const SBOX_REGISTERS: usize> {
    pub(super) subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    /// If true, `verify_batch` assumes the verification is always passed and skips poseidon2
    /// computation during execution for performance.
    optimistic: bool,
}

pub struct NativePoseidon2Filler<F: Field, const SBOX_REGISTERS: usize> {
    // pre-computed Poseidon2 sub cols for dummy rows.
    empty_poseidon2_sub_cols: Vec<F>,
    pub(super) subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2Executor<F, SBOX_REGISTERS> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        Self {
            subchip,
            optimistic: true,
        }
    }
    pub fn set_optimistic(&mut self, optimistic: bool) {
        self.optimistic = optimistic;
    }
}

pub(crate) fn compress<F: PrimeField32, const SBOX_REGISTERS: usize>(
    subchip: &Poseidon2SubChip<F, SBOX_REGISTERS>,
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> ([F; 2 * CHUNK], [F; CHUNK]) {
    let concatenated = std::array::from_fn(|i| if i < CHUNK { left[i] } else { right[i - CHUNK] });
    let permuted = subchip.permute(concatenated);
    (concatenated, std::array::from_fn(|i| permuted[i]))
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2Filler<F, SBOX_REGISTERS> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        let empty_poseidon2_sub_cols = subchip.generate_trace(vec![[F::ZERO; CHUNK * 2]]).values;
        Self {
            empty_poseidon2_sub_cols,
            subchip,
        }
    }
}

pub(super) const NUM_INITIAL_READS: usize = 6;
pub(super) const NUM_SIMPLE_ACCESSES: u32 = 7;

#[derive(Debug, Clone, Default)]
pub struct NativePoseidon2Metadata {
    num_rows: usize,
}

impl MultiRowMetadata for NativePoseidon2Metadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_rows
    }
}

type NativePoseidon2RecordLayout = MultiRowLayout<NativePoseidon2Metadata>;

pub struct NativePoseidon2RecordMut<'a, F, const SBOX_REGISTERS: usize>(
    &'a mut [NativePoseidon2Cols<F, SBOX_REGISTERS>],
);

impl<'a, F: PrimeField32, const SBOX_REGISTERS: usize>
    CustomBorrow<'a, NativePoseidon2RecordMut<'a, F, SBOX_REGISTERS>, NativePoseidon2RecordLayout>
    for [u8]
{
    fn custom_borrow(
        &'a mut self,
        layout: NativePoseidon2RecordLayout,
    ) -> NativePoseidon2RecordMut<'a, F, SBOX_REGISTERS> {
        // SAFETY:
        // - align_to_mut() ensures proper alignment for NativePoseidon2Cols<F, SBOX_REGISTERS>
        // - Layout guarantees sufficient length for num_rows records
        // - Slice bounds validated by taking only num_rows elements
        let arr = unsafe {
            self.align_to_mut::<NativePoseidon2Cols<F, SBOX_REGISTERS>>()
                .1
        };
        NativePoseidon2RecordMut(&mut arr[..layout.metadata.num_rows])
    }

    unsafe fn extract_layout(&self) -> NativePoseidon2RecordLayout {
        // Each instruction record consists solely of some number of contiguously
        // stored NativePoseidon2Cols<...> structs, each of which corresponds to a
        // single trace row. Trace fillers don't actually need to know how many rows
        // each instruction uses, and can thus treat each NativePoseidon2Cols<...>
        // as a single record.
        NativePoseidon2RecordLayout {
            metadata: NativePoseidon2Metadata { num_rows: 1 },
        }
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> SizedRecord<NativePoseidon2RecordLayout>
    for NativePoseidon2RecordMut<'_, F, SBOX_REGISTERS>
{
    fn size(layout: &NativePoseidon2RecordLayout) -> usize {
        layout.metadata.num_rows * size_of::<NativePoseidon2Cols<F, SBOX_REGISTERS>>()
    }

    fn alignment(_layout: &NativePoseidon2RecordLayout) -> usize {
        align_of::<NativePoseidon2Cols<F, SBOX_REGISTERS>>()
    }
}

impl<F: PrimeField32, RA, const SBOX_REGISTERS: usize> PreflightExecutor<F, RA>
    for NativePoseidon2Executor<F, SBOX_REGISTERS>
where
    for<'buf> RA: RecordArena<
        'buf,
        MultiRowLayout<NativePoseidon2Metadata>,
        NativePoseidon2RecordMut<'buf, F, SBOX_REGISTERS>,
    >,
{
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let arena = state.ctx;
        let init_timestamp_u32 = state.memory.timestamp;
        if instruction.opcode == PERM_POS2.global_opcode()
            || instruction.opcode == COMP_POS2.global_opcode()
        {
            let cols = &mut arena
                .alloc(MultiRowLayout::new(NativePoseidon2Metadata { num_rows: 1 }))
                .0[0];
            let simple_cols: &mut SimplePoseidonSpecificCols<F> =
                cols.specific[..SimplePoseidonSpecificCols::<u8>::width()].borrow_mut();
            let &Instruction {
                a: output_register,
                b: input_register_1,
                c: input_register_2,
                d: register_address_space,
                e: data_address_space,
                ..
            } = instruction;
            debug_assert_eq!(
                register_address_space,
                F::from_canonical_u32(AS::Native as u32)
            );
            debug_assert_eq!(data_address_space, F::from_canonical_u32(AS::Native as u32));
            let [output_pointer]: [F; 1] = tracing_read_native_helper(
                state.memory,
                output_register.as_canonical_u32(),
                simple_cols.read_output_pointer.as_mut(),
            );
            let output_pointer_u32 = output_pointer.as_canonical_u32();
            let [input_pointer_1]: [F; 1] = tracing_read_native_helper(
                state.memory,
                input_register_1.as_canonical_u32(),
                simple_cols.read_input_pointer_1.as_mut(),
            );
            let input_pointer_1_u32 = input_pointer_1.as_canonical_u32();
            let [input_pointer_2]: [F; 1] = if instruction.opcode == PERM_POS2.global_opcode() {
                state.memory.increment_timestamp();
                [input_pointer_1 + F::from_canonical_usize(CHUNK)]
            } else {
                tracing_read_native_helper(
                    state.memory,
                    input_register_2.as_canonical_u32(),
                    simple_cols.read_input_pointer_2.as_mut(),
                )
            };
            let input_pointer_2_u32 = input_pointer_2.as_canonical_u32();
            let data_1: [F; CHUNK] = tracing_read_native_helper(
                state.memory,
                input_pointer_1_u32,
                simple_cols.read_data_1.as_mut(),
            );
            let data_2: [F; CHUNK] = tracing_read_native_helper(
                state.memory,
                input_pointer_2_u32,
                simple_cols.read_data_2.as_mut(),
            );

            let p2_input = std::array::from_fn(|i| {
                if i < CHUNK {
                    data_1[i]
                } else {
                    data_2[i - CHUNK]
                }
            });
            let output = self.subchip.permute(p2_input);
            tracing_write_native_inplace(
                state.memory,
                output_pointer_u32,
                std::array::from_fn(|i| output[i]),
                &mut simple_cols.write_data_1,
            );
            if instruction.opcode == PERM_POS2.global_opcode() {
                tracing_write_native_inplace(
                    state.memory,
                    output_pointer_u32 + CHUNK as u32,
                    std::array::from_fn(|i| output[i + CHUNK]),
                    &mut simple_cols.write_data_2,
                );
            } else {
                state.memory.increment_timestamp();
            }
            debug_assert_eq!(
                state.memory.timestamp,
                init_timestamp_u32 + NUM_SIMPLE_ACCESSES
            );
            cols.incorporate_row = F::ZERO;
            cols.incorporate_sibling = F::ZERO;
            cols.inside_row = F::ZERO;
            cols.simple = F::ONE;
            cols.end_inside_row = F::ZERO;
            cols.end_top_level = F::ZERO;
            cols.is_exhausted = [F::ZERO; CHUNK - 1];
            cols.start_timestamp = F::from_canonical_u32(init_timestamp_u32);

            cols.inner.inputs = p2_input;
            simple_cols.pc = F::from_canonical_u32(*state.pc);
            simple_cols.is_compress = F::from_bool(instruction.opcode == COMP_POS2.global_opcode());
            simple_cols.output_register = output_register;
            simple_cols.input_register_1 = input_register_1;
            simple_cols.input_register_2 = input_register_2;
            simple_cols.output_pointer = output_pointer;
            simple_cols.input_pointer_1 = input_pointer_1;
            simple_cols.input_pointer_2 = input_pointer_2;
        } else if instruction.opcode == VERIFY_BATCH.global_opcode() {
            let init_timestamp = F::from_canonical_u32(init_timestamp_u32);
            let mut col_buffer = vec![F::ZERO; NativePoseidon2Cols::<F, SBOX_REGISTERS>::width()];
            let last_top_level_cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> =
                col_buffer.as_mut_slice().borrow_mut();
            let ltl_specific_cols: &mut TopLevelSpecificCols<F> =
                last_top_level_cols.specific[..TopLevelSpecificCols::<u8>::width()].borrow_mut();
            let &Instruction {
                a: dim_register,
                b: opened_register,
                c: opened_length_register,
                d: proof_id_ptr,
                e: index_register,
                f: commit_register,
                g: opened_element_size_inv,
                ..
            } = instruction;
            // calc inverse fast assuming opened_element_size in {1, 4}
            let mut opened_element_size = F::ONE;
            while opened_element_size * opened_element_size_inv != F::ONE {
                opened_element_size += F::ONE;
            }

            let [proof_id]: [F; 1] =
                memory_read_native(state.memory.data(), proof_id_ptr.as_canonical_u32());
            let [dim_base_pointer]: [F; 1] = tracing_read_native_helper(
                state.memory,
                dim_register.as_canonical_u32(),
                ltl_specific_cols.dim_base_pointer_read.as_mut(),
            );
            let dim_base_pointer_u32 = dim_base_pointer.as_canonical_u32();
            let [opened_base_pointer]: [F; 1] = tracing_read_native_helper(
                state.memory,
                opened_register.as_canonical_u32(),
                ltl_specific_cols.opened_base_pointer_read.as_mut(),
            );
            let opened_base_pointer_u32 = opened_base_pointer.as_canonical_u32();
            let [opened_length]: [F; 1] = tracing_read_native_helper(
                state.memory,
                opened_length_register.as_canonical_u32(),
                ltl_specific_cols.opened_length_read.as_mut(),
            );
            let [index_base_pointer]: [F; 1] = tracing_read_native_helper(
                state.memory,
                index_register.as_canonical_u32(),
                ltl_specific_cols.index_base_pointer_read.as_mut(),
            );
            let index_base_pointer_u32 = index_base_pointer.as_canonical_u32();
            let [commit_pointer]: [F; 1] = tracing_read_native_helper(
                state.memory,
                commit_register.as_canonical_u32(),
                ltl_specific_cols.commit_pointer_read.as_mut(),
            );
            // In E3, the proof is assumed to be valid. The verification during execution is
            // skipped.
            let commit: [F; CHUNK] = tracing_read_native_helper(
                state.memory,
                commit_pointer.as_canonical_u32(),
                ltl_specific_cols.commit_read.as_mut(),
            );

            let opened_length = opened_length.as_canonical_u32() as usize;
            let [initial_log_height]: [F; 1] =
                memory_read_native(state.memory.data(), dim_base_pointer_u32);
            let initial_log_height_u32 = initial_log_height.as_canonical_u32();
            let mut log_height = initial_log_height_u32 as i32;

            // Number of non-inside rows, this is used to compute the offset of the inside row
            // section.
            let (num_inside_rows, num_non_inside_rows) = {
                let opened_element_size_u32 = opened_element_size.as_canonical_u32();
                let mut num_non_inside_rows = initial_log_height_u32 as usize;
                let mut num_inside_rows = 0;
                let mut log_height = initial_log_height_u32;
                let mut opened_index = 0;
                loop {
                    let mut total_len = 0;
                    while opened_index < opened_length {
                        let [height]: [F; 1] = memory_read_native(
                            state.memory.data(),
                            dim_base_pointer_u32 + opened_index as u32,
                        );
                        if height.as_canonical_u32() != log_height {
                            break;
                        }
                        let [row_len]: [F; 1] = memory_read_native(
                            state.memory.data(),
                            opened_base_pointer_u32 + 2 * opened_index as u32 + 1,
                        );
                        total_len += row_len.as_canonical_u32() * opened_element_size_u32;
                        opened_index += 1;
                    }
                    if total_len != 0 {
                        num_non_inside_rows += 1;
                        num_inside_rows += (total_len as usize).div_ceil(CHUNK);
                    }
                    if log_height == 0 {
                        break;
                    }
                    log_height -= 1;
                }
                (num_inside_rows, num_non_inside_rows)
            };
            let mut proof_index = 0;
            let mut opened_index = 0;

            let mut root = [F::ZERO; CHUNK];
            let sibling_proof: Vec<[F; CHUNK]> = {
                let proof_idx = proof_id.as_canonical_u32() as usize;
                state.streams.hint_space[proof_idx]
                    .par_chunks(CHUNK)
                    .map(|c| c.try_into().unwrap())
                    .collect()
            };

            let total_num_row = num_inside_rows + num_non_inside_rows;
            let allocated_rows = arena
                .alloc(MultiRowLayout::new(NativePoseidon2Metadata {
                    num_rows: total_num_row,
                }))
                .0;
            allocated_rows[0].inner.export = F::from_canonical_u32(num_non_inside_rows as u32);
            let mut inside_row_idx = num_non_inside_rows;
            let mut non_inside_row_idx = 0;

            while log_height >= 0 {
                if opened_index < opened_length
                    && memory_read_native::<F, 1>(
                        state.memory.data(),
                        dim_base_pointer_u32 + opened_index as u32,
                    )[0] == F::from_canonical_u32(log_height as u32)
                {
                    state
                        .memory
                        .increment_timestamp_by(NUM_INITIAL_READS as u32);
                    let incorporate_start_timestamp = state.memory.timestamp;
                    let initial_opened_index = opened_index;
                    let mut row_pointer = 0;
                    let mut row_end = 0;
                    let mut rolling_hash = [F::ZERO; 2 * CHUNK];
                    let mut is_first_in_segment = true;

                    loop {
                        if inside_row_idx == total_num_row {
                            opened_index += 1;
                            break;
                        }
                        let inside_cols = &mut allocated_rows[inside_row_idx];
                        let inside_specific_cols: &mut InsideRowSpecificCols<F> = inside_cols
                            .specific[..InsideRowSpecificCols::<u8>::width()]
                            .borrow_mut();
                        let start_timestamp_u32 = state.memory.timestamp;

                        let mut cells_idx = 0;
                        for chunk_elem in rolling_hash.iter_mut().take(CHUNK) {
                            let cell_cols = &mut inside_specific_cols.cells[cells_idx];
                            if is_first_in_segment || row_pointer == row_end {
                                if is_first_in_segment {
                                    is_first_in_segment = false;
                                } else {
                                    opened_index += 1;
                                    if opened_index == opened_length
                                        || memory_read_native::<F, 1>(
                                            state.memory.data(),
                                            dim_base_pointer_u32 + opened_index as u32,
                                        )[0] != F::from_canonical_u32(log_height as u32)
                                    {
                                        break;
                                    }
                                }
                                let [new_row_pointer, row_len]: [F; 2] = tracing_read_native_helper(
                                    state.memory,
                                    opened_base_pointer_u32 + 2 * opened_index as u32,
                                    cell_cols.read_row_pointer_and_length.as_mut(),
                                );
                                row_pointer = new_row_pointer.as_canonical_u32() as usize;
                                row_end = row_pointer
                                    + (opened_element_size * row_len).as_canonical_u32() as usize;
                                cell_cols.is_first_in_row = F::ONE;
                            } else {
                                state.memory.increment_timestamp();
                            }
                            let [value]: [F; 1] = tracing_read_native_helper(
                                state.memory,
                                row_pointer as u32,
                                cell_cols.read.as_mut(),
                            );

                            cell_cols.opened_index = F::from_canonical_usize(opened_index);
                            cell_cols.row_pointer = F::from_canonical_usize(row_pointer);
                            cell_cols.row_end = F::from_canonical_usize(row_end);

                            *chunk_elem = value;
                            row_pointer += 1;
                            cells_idx += 1;
                        }
                        if cells_idx == 0 {
                            break;
                        }
                        inside_cols.inner.inputs[..CHUNK].copy_from_slice(&rolling_hash[..CHUNK]);
                        if !self.optimistic {
                            self.subchip.permute_mut(&mut rolling_hash);
                        }
                        if cells_idx < CHUNK {
                            state
                                .memory
                                .increment_timestamp_by(2 * (CHUNK - cells_idx) as u32);
                        }

                        inside_row_idx += 1;
                        // left
                        inside_cols.incorporate_row = F::ZERO;
                        inside_cols.incorporate_sibling = F::ZERO;
                        inside_cols.inside_row = F::ONE;
                        inside_cols.simple = F::ZERO;
                        // `end_inside_row` of the last row will be set to 1 after this loop.
                        inside_cols.end_inside_row = F::ZERO;
                        inside_cols.end_top_level = F::ZERO;
                        inside_cols.opened_element_size_inv = opened_element_size_inv;
                        inside_cols.very_first_timestamp =
                            F::from_canonical_u32(incorporate_start_timestamp);
                        inside_cols.start_timestamp = F::from_canonical_u32(start_timestamp_u32);

                        inside_cols.initial_opened_index =
                            F::from_canonical_usize(initial_opened_index);
                        inside_cols.opened_base_pointer = opened_base_pointer;
                        if cells_idx < CHUNK {
                            let exhausted_opened_idx = F::from_canonical_usize(opened_index - 1);
                            for exhausted_idx in cells_idx..CHUNK {
                                inside_cols.is_exhausted[exhausted_idx - 1] = F::ONE;
                                inside_specific_cols.cells[exhausted_idx].opened_index =
                                    exhausted_opened_idx;
                            }
                            break;
                        }
                    }
                    {
                        let inside_cols = &mut allocated_rows[inside_row_idx - 1];
                        inside_cols.end_inside_row = F::ONE;
                    }

                    let incorporate_cols = &mut allocated_rows[non_inside_row_idx];
                    let top_level_specific_cols: &mut TopLevelSpecificCols<F> = incorporate_cols
                        .specific[..TopLevelSpecificCols::<u8>::width()]
                        .borrow_mut();

                    let final_opened_index = opened_index - 1;
                    let [height_check]: [F; 1] = tracing_read_native_helper(
                        state.memory,
                        dim_base_pointer_u32 + initial_opened_index as u32,
                        top_level_specific_cols
                            .read_initial_height_or_sibling_is_on_right
                            .as_mut(),
                    );
                    assert_eq!(height_check, F::from_canonical_u32(log_height as u32));
                    let final_height_read_timestamp = state.memory.timestamp;
                    let [height_check]: [F; 1] = tracing_read_native_helper(
                        state.memory,
                        dim_base_pointer_u32 + final_opened_index as u32,
                        top_level_specific_cols.read_final_height.as_mut(),
                    );
                    assert_eq!(height_check, F::from_canonical_u32(log_height as u32));

                    if !self.optimistic {
                        let hash: [F; CHUNK] = std::array::from_fn(|i| rolling_hash[i]);
                        root = if log_height as u32 == initial_log_height_u32 {
                            hash
                        } else {
                            compress(&self.subchip, root, hash).1
                        };
                    }
                    non_inside_row_idx += 1;

                    incorporate_cols.incorporate_row = F::ONE;
                    incorporate_cols.incorporate_sibling = F::ZERO;
                    incorporate_cols.inside_row = F::ZERO;
                    incorporate_cols.simple = F::ZERO;
                    incorporate_cols.end_inside_row = F::ZERO;
                    incorporate_cols.end_top_level = F::ZERO;
                    incorporate_cols.start_top_level = F::from_bool(proof_index == 0);
                    incorporate_cols.opened_element_size_inv = opened_element_size_inv;
                    incorporate_cols.very_first_timestamp = init_timestamp;
                    incorporate_cols.start_timestamp = F::from_canonical_u32(
                        incorporate_start_timestamp - NUM_INITIAL_READS as u32,
                    );
                    top_level_specific_cols.end_timestamp =
                        F::from_canonical_u32(final_height_read_timestamp + 1);

                    incorporate_cols.initial_opened_index =
                        F::from_canonical_usize(initial_opened_index);
                    top_level_specific_cols.final_opened_index =
                        F::from_canonical_usize(final_opened_index);
                    top_level_specific_cols.log_height = F::from_canonical_u32(log_height as u32);
                    top_level_specific_cols.opened_length = F::from_canonical_usize(opened_length);
                    top_level_specific_cols.dim_base_pointer = dim_base_pointer;
                    incorporate_cols.opened_base_pointer = opened_base_pointer;
                    top_level_specific_cols.index_base_pointer = index_base_pointer;
                    top_level_specific_cols.proof_index = F::from_canonical_usize(proof_index);
                }

                if log_height != 0 {
                    let row_start_timestamp = state.memory.timestamp;
                    state
                        .memory
                        .increment_timestamp_by(NUM_INITIAL_READS as u32);

                    let sibling_cols = &mut allocated_rows[non_inside_row_idx];
                    let top_level_specific_cols: &mut TopLevelSpecificCols<F> =
                        sibling_cols.specific[..TopLevelSpecificCols::<u8>::width()].borrow_mut();

                    let read_sibling_is_on_right_timestamp = state.memory.timestamp;
                    let [sibling_is_on_right]: [F; 1] = tracing_read_native_helper(
                        state.memory,
                        index_base_pointer_u32 + proof_index as u32,
                        top_level_specific_cols
                            .read_initial_height_or_sibling_is_on_right
                            .as_mut(),
                    );
                    let sibling = sibling_proof[proof_index];
                    if !self.optimistic {
                        root = if sibling_is_on_right == F::ONE {
                            compress(&self.subchip, sibling, root).1
                        } else {
                            compress(&self.subchip, root, sibling).1
                        };
                    }

                    non_inside_row_idx += 1;

                    sibling_cols.inner.inputs[..CHUNK].copy_from_slice(&sibling);

                    sibling_cols.incorporate_row = F::ZERO;
                    sibling_cols.incorporate_sibling = F::ONE;
                    sibling_cols.inside_row = F::ZERO;
                    sibling_cols.simple = F::ZERO;
                    sibling_cols.end_inside_row = F::ZERO;
                    sibling_cols.end_top_level = F::ZERO;
                    sibling_cols.start_top_level = F::ZERO;
                    sibling_cols.opened_element_size_inv = opened_element_size_inv;
                    sibling_cols.very_first_timestamp = init_timestamp;
                    sibling_cols.start_timestamp = F::from_canonical_u32(row_start_timestamp);

                    top_level_specific_cols.end_timestamp =
                        F::from_canonical_u32(read_sibling_is_on_right_timestamp + 1);
                    sibling_cols.initial_opened_index = F::from_canonical_usize(opened_index);
                    top_level_specific_cols.final_opened_index =
                        F::from_canonical_usize(opened_index - 1);
                    top_level_specific_cols.log_height = F::from_canonical_u32(log_height as u32);
                    top_level_specific_cols.opened_length = F::from_canonical_usize(opened_length);
                    top_level_specific_cols.dim_base_pointer = dim_base_pointer;
                    sibling_cols.opened_base_pointer = opened_base_pointer;
                    top_level_specific_cols.index_base_pointer = index_base_pointer;

                    top_level_specific_cols.proof_index = F::from_canonical_usize(proof_index);
                    top_level_specific_cols.sibling_is_on_right = sibling_is_on_right;
                };

                log_height -= 1;
                proof_index += 1;
            }
            let ltl_trace_cols = &mut allocated_rows[non_inside_row_idx - 1];
            let ltl_trace_specific_cols: &mut TopLevelSpecificCols<F> =
                ltl_trace_cols.specific[..TopLevelSpecificCols::<u8>::width()].borrow_mut();
            ltl_trace_cols.inner.export = F::from_canonical_u32(total_num_row as u32);
            ltl_trace_cols.end_top_level = F::ONE;
            ltl_trace_specific_cols.pc = F::from_canonical_u32(*state.pc);
            ltl_trace_specific_cols.dim_register = dim_register;
            ltl_trace_specific_cols.opened_register = opened_register;
            ltl_trace_specific_cols.opened_length_register = opened_length_register;
            ltl_trace_specific_cols.proof_id = proof_id_ptr;
            ltl_trace_specific_cols.index_register = index_register;
            ltl_trace_specific_cols.commit_register = commit_register;
            ltl_trace_specific_cols.commit_pointer = commit_pointer;
            ltl_trace_specific_cols.dim_base_pointer_read = ltl_specific_cols.dim_base_pointer_read;
            ltl_trace_specific_cols.opened_base_pointer_read =
                ltl_specific_cols.opened_base_pointer_read;
            ltl_trace_specific_cols.opened_length_read = ltl_specific_cols.opened_length_read;
            ltl_trace_specific_cols.index_base_pointer_read =
                ltl_specific_cols.index_base_pointer_read;
            ltl_trace_specific_cols.commit_pointer_read = ltl_specific_cols.commit_pointer_read;
            ltl_trace_specific_cols.commit_read = ltl_specific_cols.commit_read;
            if !self.optimistic {
                assert_eq!(commit, root);
            }
        } else {
            unreachable!()
        }

        *state.pc += DEFAULT_PC_STEP;
        Ok(())
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == VERIFY_BATCH.global_opcode().as_usize() {
            String::from("VERIFY_BATCH")
        } else if opcode == PERM_POS2.global_opcode().as_usize() {
            String::from("PERM_POS2")
        } else if opcode == COMP_POS2.global_opcode().as_usize() {
            String::from("COMP_POS2")
        } else {
            unreachable!("unsupported opcode: {}", opcode)
        }
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> TraceFiller<F>
    for NativePoseidon2Filler<F, SBOX_REGISTERS>
{
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) where
        F: Send + Sync + Clone,
    {
        // Split the trace rows by instruction
        let width = trace.width();
        let mut row_idx = 0;
        let mut row_slice = trace.values.as_mut_slice();
        let mut chunk_start = Vec::new();
        while row_idx < rows_used {
            let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice[..width].borrow();
            let (curr, rest) = if cols.simple.is_one() {
                row_idx += 1;
                row_slice.split_at_mut(width)
            } else {
                let num_non_inside_row = cols.inner.export.as_canonical_u32() as usize;
                let start = (num_non_inside_row - 1) * width;
                let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                    row_slice[start..(start + width)].borrow();
                let total_num_row = cols.inner.export.as_canonical_u32() as usize;
                row_idx += total_num_row;
                row_slice.split_at_mut(total_num_row * width)
            };
            chunk_start.push(curr);
            row_slice = rest;
        }
        chunk_start.into_par_iter().for_each(|chunk_slice| {
            let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = chunk_slice[..width].borrow();
            if cols.simple.is_one() {
                self.fill_simple_chunk(mem_helper, chunk_slice);
            } else {
                self.fill_verify_batch_chunk(mem_helper, chunk_slice);
            }
        });
        // Remaining rows are dummy rows.
        let inner_width = self.subchip.air.width();
        row_slice.par_chunks_exact_mut(width).for_each(|row_slice| {
            row_slice[..inner_width].copy_from_slice(&self.empty_poseidon2_sub_cols);
        });
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2Filler<F, SBOX_REGISTERS> {
    fn fill_simple_chunk(&self, mem_helper: &MemoryAuxColsFactory<F>, chunk_slice: &mut [F]) {
        {
            let inner_width = self.subchip.air.width();
            let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = chunk_slice.as_ref().borrow();
            let inner_cols = &self.subchip.generate_trace(vec![cols.inner.inputs]).values;
            chunk_slice[..inner_width].copy_from_slice(inner_cols);
        }

        let cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = chunk_slice.borrow_mut();
        // Simple poseidon2 row
        let simple_cols: &mut SimplePoseidonSpecificCols<F> =
            cols.specific[..SimplePoseidonSpecificCols::<u8>::width()].borrow_mut();
        let start_timestamp_u32 = cols.start_timestamp.as_canonical_u32();
        mem_fill_helper(
            mem_helper,
            start_timestamp_u32,
            simple_cols.read_output_pointer.as_mut(),
        );
        mem_fill_helper(
            mem_helper,
            start_timestamp_u32 + 1,
            simple_cols.read_input_pointer_1.as_mut(),
        );
        if simple_cols.is_compress.is_one() {
            mem_fill_helper(
                mem_helper,
                start_timestamp_u32 + 2,
                simple_cols.read_input_pointer_2.as_mut(),
            );
        }
        mem_fill_helper(
            mem_helper,
            start_timestamp_u32 + 3,
            simple_cols.read_data_1.as_mut(),
        );
        mem_fill_helper(
            mem_helper,
            start_timestamp_u32 + 4,
            simple_cols.read_data_2.as_mut(),
        );
        mem_fill_helper(
            mem_helper,
            start_timestamp_u32 + 5,
            simple_cols.write_data_1.as_mut(),
        );
        if simple_cols.is_compress.is_zero() {
            mem_fill_helper(
                mem_helper,
                start_timestamp_u32 + 6,
                simple_cols.write_data_2.as_mut(),
            );
        }
    }

    fn fill_verify_batch_chunk(&self, mem_helper: &MemoryAuxColsFactory<F>, chunk_slice: &mut [F]) {
        let inner_width = self.subchip.air.width();
        let width = NativePoseidon2Cols::<F, SBOX_REGISTERS>::width();
        let num_non_inside_rows = {
            let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = chunk_slice[..width].borrow();
            cols.inner.export.as_canonical_u32() as usize
        };
        let total_num_rows = {
            let start = (num_non_inside_rows - 1) * width;
            let last_cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                chunk_slice[start..(start + width)].borrow();
            // During execution, this field hasn't been filled with meaningful data. So we use this
            // field to store the number of inside rows.
            last_cols.inner.export.as_canonical_u32() as usize
        };
        let mut first_round = true;
        let mut root = [F::ZERO; CHUNK];
        let mut inside_idx = num_non_inside_rows;
        let mut non_inside_idx = 0;
        while inside_idx < total_num_rows || non_inside_idx < num_non_inside_rows {
            debug_assert!(non_inside_idx < num_non_inside_rows);
            let incorporate_sibling = {
                let start = non_inside_idx * width;
                let row_slice = &mut chunk_slice[start..(start + width)];
                let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.as_ref().borrow();
                cols.incorporate_sibling.is_one()
            };
            if !incorporate_sibling {
                let mut prev_rolling_hash: [F; 2 * CHUNK];
                let mut rolling_hash = [F::ZERO; 2 * CHUNK];
                loop {
                    let start = inside_idx * width;
                    let row_slice = &mut chunk_slice[start..(start + width)];
                    let mut input_len = 0;
                    {
                        let cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> =
                            row_slice.borrow_mut();
                        let inside_row_specific_cols: &mut InsideRowSpecificCols<F> =
                            cols.specific[..InsideRowSpecificCols::<u8>::width()].borrow_mut();
                        let start_timestamp_u32 = cols.start_timestamp.as_canonical_u32();
                        for (i, cell) in inside_row_specific_cols.cells.iter_mut().enumerate() {
                            if i > 0 && cols.is_exhausted[i - 1].is_one() {
                                break;
                            }
                            input_len += 1;
                            if cell.is_first_in_row.is_one() {
                                mem_fill_helper(
                                    mem_helper,
                                    start_timestamp_u32 + 2 * i as u32,
                                    cell.read_row_pointer_and_length.as_mut(),
                                );
                            }
                            mem_fill_helper(
                                mem_helper,
                                start_timestamp_u32 + 2 * i as u32 + 1,
                                cell.read.as_mut(),
                            );
                        }
                    }
                    {
                        let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                            row_slice.as_ref().borrow();
                        rolling_hash[..input_len].copy_from_slice(&cols.inner.inputs[..input_len]);
                    }
                    prev_rolling_hash = rolling_hash;

                    let inner_cols = &self.subchip.generate_trace(vec![rolling_hash]).values;
                    row_slice[..inner_width].copy_from_slice(inner_cols);
                    let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.as_ref().borrow();
                    rolling_hash = *Self::poseidon2_output_from_trace(&cols.inner);
                    inside_idx += 1;
                    if cols.end_inside_row.is_one() {
                        break;
                    }
                }

                let start = non_inside_idx * width;
                let row_slice = &mut chunk_slice[start..(start + width)];
                let mut p2_input = [F::ZERO; 2 * CHUNK];
                if first_round {
                    p2_input.copy_from_slice(&prev_rolling_hash);
                } else {
                    p2_input[..CHUNK].copy_from_slice(&root);
                    p2_input[CHUNK..].copy_from_slice(&rolling_hash[..CHUNK]);
                }

                first_round = false;
                let inner_cols = &self.subchip.generate_trace(vec![p2_input]).values;
                row_slice[..inner_width].copy_from_slice(inner_cols);
                let cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.borrow_mut();
                Self::fill_timestamp_for_top_level(mem_helper, cols);
                root.copy_from_slice(&Self::poseidon2_output_from_trace(&cols.inner)[..CHUNK]);
                non_inside_idx += 1;
            }

            if non_inside_idx < num_non_inside_rows {
                let start = non_inside_idx * width;
                let row_slice = &mut chunk_slice[start..(start + width)];
                let p2_input = {
                    let cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.borrow_mut();
                    Self::fill_timestamp_for_top_level(mem_helper, cols);
                    let sibling = &cols.inner.inputs[..CHUNK];
                    let top_level_specific_cols: &TopLevelSpecificCols<F> =
                        cols.specific[..TopLevelSpecificCols::<F>::width()].borrow();
                    let sibling_is_on_right = top_level_specific_cols.sibling_is_on_right.is_one();
                    let mut p2_input = [F::ZERO; 2 * CHUNK];
                    if sibling_is_on_right {
                        p2_input[..CHUNK].copy_from_slice(sibling);
                        p2_input[CHUNK..].copy_from_slice(&root);
                    } else {
                        p2_input[..CHUNK].copy_from_slice(&root);
                        p2_input[CHUNK..].copy_from_slice(sibling);
                    };
                    p2_input
                };
                let inner_cols = &self.subchip.generate_trace(vec![p2_input]).values;
                row_slice[..inner_width].copy_from_slice(inner_cols);
                let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.as_ref().borrow();
                root.copy_from_slice(&Self::poseidon2_output_from_trace(&cols.inner)[..CHUNK]);
                non_inside_idx += 1;
            }
        }
    }
    fn fill_timestamp_for_top_level(
        mem_helper: &MemoryAuxColsFactory<F>,
        cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS>,
    ) {
        let top_level_specific_cols: &mut TopLevelSpecificCols<F> =
            cols.specific[..TopLevelSpecificCols::<u8>::width()].borrow_mut();
        let start_timestamp_u32 = cols.start_timestamp.as_canonical_u32();
        if cols.end_top_level.is_one() {
            let very_start_timestamp_u32 = cols.very_first_timestamp.as_canonical_u32();
            mem_fill_helper(
                mem_helper,
                very_start_timestamp_u32,
                top_level_specific_cols.dim_base_pointer_read.as_mut(),
            );
            mem_fill_helper(
                mem_helper,
                very_start_timestamp_u32 + 1,
                top_level_specific_cols.opened_base_pointer_read.as_mut(),
            );
            mem_fill_helper(
                mem_helper,
                very_start_timestamp_u32 + 2,
                top_level_specific_cols.opened_length_read.as_mut(),
            );
            mem_fill_helper(
                mem_helper,
                very_start_timestamp_u32 + 3,
                top_level_specific_cols.index_base_pointer_read.as_mut(),
            );
            mem_fill_helper(
                mem_helper,
                very_start_timestamp_u32 + 4,
                top_level_specific_cols.commit_pointer_read.as_mut(),
            );
            mem_fill_helper(
                mem_helper,
                very_start_timestamp_u32 + 5,
                top_level_specific_cols.commit_read.as_mut(),
            );
        }
        if cols.incorporate_row.is_one() {
            let end_timestamp = top_level_specific_cols.end_timestamp.as_canonical_u32();
            mem_fill_helper(
                mem_helper,
                end_timestamp - 2,
                top_level_specific_cols
                    .read_initial_height_or_sibling_is_on_right
                    .as_mut(),
            );
            mem_fill_helper(
                mem_helper,
                end_timestamp - 1,
                top_level_specific_cols.read_final_height.as_mut(),
            );
        } else if cols.incorporate_sibling.is_one() {
            mem_fill_helper(
                mem_helper,
                start_timestamp_u32 + NUM_INITIAL_READS as u32,
                top_level_specific_cols
                    .read_initial_height_or_sibling_is_on_right
                    .as_mut(),
            );
        } else {
            unreachable!()
        }
    }

    #[inline(always)]
    fn poseidon2_output_from_trace(inner: &Poseidon2SubCols<F, SBOX_REGISTERS>) -> &[F; 2 * CHUNK] {
        &inner.ending_full_rounds.last().unwrap().post
    }
}

fn tracing_read_native_helper<F: PrimeField32, const BLOCK_SIZE: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    base_aux: &mut MemoryBaseAuxCols<F>,
) -> [F; BLOCK_SIZE] {
    let mut prev_ts = 0;
    let ret = tracing_read_native(memory, ptr, &mut prev_ts);
    base_aux.set_prev(F::from_canonical_u32(prev_ts));
    ret
}

/// Fill `MemoryBaseAuxCols`, assuming that the `prev_timestamp` is already set in `base_aux`.
fn mem_fill_helper<F: PrimeField32>(
    mem_helper: &MemoryAuxColsFactory<F>,
    timestamp: u32,
    base_aux: &mut MemoryBaseAuxCols<F>,
) {
    let prev_ts = base_aux.prev_timestamp.as_canonical_u32();
    mem_helper.fill(prev_ts, timestamp, base_aux);
}
