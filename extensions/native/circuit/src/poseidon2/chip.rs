use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        CustomBorrow, E2PreCompute, ExecuteFunc, ExecutionError, InsExecutorE1, InsExecutorE2,
        InstructionExecutor, MultiRowLayout, MultiRowMetadata, RecordArena, SizedRecord,
        TraceFiller, VmSegmentState, VmStateMut,
    },
    system::{
        memory::{
            offline_checker::MemoryBaseAuxCols,
            online::{GuestMemory, TracingMemory},
            MemoryAuxColsFactory,
        },
        native_adapter::util::{
            memory_read_native, tracing_read_native, tracing_write_native_inplace,
        },
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS,
    Poseidon2Opcode::{COMP_POS2, PERM_POS2},
    VerifyBatchOpcode::VERIFY_BATCH,
};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice},
};

use crate::poseidon2::{
    columns::{
        InsideRowSpecificCols, NativePoseidon2Cols, SimplePoseidonSpecificCols,
        TopLevelSpecificCols,
    },
    CHUNK,
};

#[derive(Clone)]
pub struct NativePoseidon2Step<F: Field, const SBOX_REGISTERS: usize> {
    pub(super) subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
}

pub struct NativePoseidon2Filler<F: Field, const SBOX_REGISTERS: usize> {
    // pre-computed Poseidon2 sub cols for dummy rows.
    empty_poseidon2_sub_cols: Vec<F>,
    pub(super) subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2Step<F, SBOX_REGISTERS> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        Self { subchip }
    }
}

fn compress<F: PrimeField32, const SBOX_REGISTERS: usize>(
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

impl<F: PrimeField32, RA, const SBOX_REGISTERS: usize> InstructionExecutor<F, RA>
    for NativePoseidon2Step<F, SBOX_REGISTERS>
where
    for<'buf> RA: RecordArena<
        'buf,
        MultiRowLayout<NativePoseidon2Metadata>,
        NativePoseidon2RecordMut<'buf, F, SBOX_REGISTERS>,
    >,
{
    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> openvm_circuit::arch::Result<()> {
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
            let commit = tracing_read_native_helper(
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

                    let mut prev_rolling_hash: Option<[F; 2 * CHUNK]> = None;
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
                        let p2_input = rolling_hash;
                        prev_rolling_hash = Some(rolling_hash);
                        self.subchip.permute_mut(&mut rolling_hash);
                        if cells_idx < CHUNK {
                            state
                                .memory
                                .increment_timestamp_by(2 * (CHUNK - cells_idx) as u32);
                        }

                        inside_row_idx += 1;
                        inside_cols.inner.inputs = p2_input;
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

                    let hash: [F; CHUNK] = std::array::from_fn(|i| rolling_hash[i]);
                    let (p2_input, new_root) = if log_height as u32 == initial_log_height_u32 {
                        (prev_rolling_hash.unwrap(), hash)
                    } else {
                        compress(&self.subchip, root, hash)
                    };
                    root = new_root;
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

                    incorporate_cols.inner.inputs = p2_input;
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
                    let (p2_input, new_root) = if sibling_is_on_right == F::ONE {
                        compress(&self.subchip, sibling, root)
                    } else {
                        compress(&self.subchip, root, sibling)
                    };
                    root = new_root;

                    non_inside_row_idx += 1;

                    sibling_cols.inner.inputs = p2_input;

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
            assert_eq!(commit, root);
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
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let inner_cols = {
            let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.as_ref().borrow();
            &self.subchip.generate_trace(vec![cols.inner.inputs]).values
        };
        let inner_width = self.subchip.air.width();
        row_slice[..inner_width].copy_from_slice(inner_cols);
        let cols: &mut NativePoseidon2Cols<F, SBOX_REGISTERS> = row_slice.borrow_mut();

        // Simple poseidon2 row
        if cols.simple.is_one() {
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
        } else if cols.inside_row.is_one() {
            let inside_row_specific_cols: &mut InsideRowSpecificCols<F> =
                cols.specific[..InsideRowSpecificCols::<u8>::width()].borrow_mut();
            let start_timestamp_u32 = cols.start_timestamp.as_canonical_u32();
            for (i, cell) in inside_row_specific_cols.cells.iter_mut().enumerate() {
                if i > 0 && cols.is_exhausted[i - 1].is_one() {
                    break;
                }
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
        } else {
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
    }

    fn fill_dummy_trace_row(&self, row_slice: &mut [F]) {
        let width = self.subchip.air.width();
        row_slice[..width].copy_from_slice(&self.empty_poseidon2_sub_cols);
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

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Pos2PreCompute<'a, F: Field, const SBOX_REGISTERS: usize> {
    subchip: &'a Poseidon2SubChip<F, SBOX_REGISTERS>,
    output_register: u32,
    input_register_1: u32,
    input_register_2: u32,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct VerifyBatchPreCompute<'a, F: Field, const SBOX_REGISTERS: usize> {
    subchip: &'a Poseidon2SubChip<F, SBOX_REGISTERS>,
    dim_register: u32,
    opened_register: u32,
    opened_length_register: u32,
    proof_id_ptr: u32,
    index_register: u32,
    commit_register: u32,
    opened_element_size: F,
}

impl<'a, F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2Step<F, SBOX_REGISTERS> {
    #[inline(always)]
    fn pre_compute_pos2_impl(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        pos2_data: &mut Pos2PreCompute<'a, F, SBOX_REGISTERS>,
    ) -> Result<(), ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        if opcode != PERM_POS2.global_opcode() && opcode != COMP_POS2.global_opcode() {
            return Err(ExecutionError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();

        if d != AS::Native as u32 {
            return Err(ExecutionError::InvalidInstruction(pc));
        }
        if e != AS::Native as u32 {
            return Err(ExecutionError::InvalidInstruction(pc));
        }

        *pos2_data = Pos2PreCompute {
            subchip: &self.subchip,
            output_register: a,
            input_register_1: b,
            input_register_2: c,
        };

        Ok(())
    }

    #[inline(always)]
    fn pre_compute_verify_batch_impl(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        verify_batch_data: &mut VerifyBatchPreCompute<'a, F, SBOX_REGISTERS>,
    ) -> Result<(), ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;

        if opcode != VERIFY_BATCH.global_opcode() {
            return Err(ExecutionError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        let f = f.as_canonical_u32();

        let opened_element_size_inv = g;
        // calc inverse fast assuming opened_element_size in {1, 4}
        let mut opened_element_size = F::ONE;
        while opened_element_size * opened_element_size_inv != F::ONE {
            opened_element_size += F::ONE;
        }

        *verify_batch_data = VerifyBatchPreCompute {
            subchip: &self.subchip,
            dim_register: a,
            opened_register: b,
            opened_length_register: c,
            proof_id_ptr: d,
            index_register: e,
            commit_register: f,
            opened_element_size,
        };

        Ok(())
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> InsExecutorE1<F>
    for NativePoseidon2Step<F, SBOX_REGISTERS>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::cmp::max(
            size_of::<Pos2PreCompute<F, SBOX_REGISTERS>>(),
            size_of::<VerifyBatchPreCompute<F, SBOX_REGISTERS>>(),
        )
    }

    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, ExecutionError> {
        let &Instruction { opcode, .. } = inst;

        let is_pos2 = opcode == PERM_POS2.global_opcode() || opcode == COMP_POS2.global_opcode();

        if is_pos2 {
            let pos2_data: &mut Pos2PreCompute<F, SBOX_REGISTERS> = data.borrow_mut();
            self.pre_compute_pos2_impl(pc, inst, pos2_data)?;
            if opcode == PERM_POS2.global_opcode() {
                Ok(execute_pos2_e1_impl::<_, _, SBOX_REGISTERS, true>)
            } else {
                Ok(execute_pos2_e1_impl::<_, _, SBOX_REGISTERS, false>)
            }
        } else {
            let verify_batch_data: &mut VerifyBatchPreCompute<F, SBOX_REGISTERS> =
                data.borrow_mut();
            self.pre_compute_verify_batch_impl(pc, inst, verify_batch_data)?;
            Ok(execute_verify_batch_e1_impl::<_, _, SBOX_REGISTERS>)
        }
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> InsExecutorE2<F>
    for NativePoseidon2Step<F, SBOX_REGISTERS>
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        std::cmp::max(
            size_of::<E2PreCompute<Pos2PreCompute<F, SBOX_REGISTERS>>>(),
            size_of::<E2PreCompute<VerifyBatchPreCompute<F, SBOX_REGISTERS>>>(),
        )
    }

    #[inline(always)]
    fn pre_compute_e2<Ctx: E2ExecutionCtx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, ExecutionError> {
        let &Instruction { opcode, .. } = inst;

        let is_pos2 = opcode == PERM_POS2.global_opcode() || opcode == COMP_POS2.global_opcode();

        if is_pos2 {
            let pre_compute: &mut E2PreCompute<Pos2PreCompute<F, SBOX_REGISTERS>> =
                data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_pos2_impl(pc, inst, &mut pre_compute.data)?;
            if opcode == PERM_POS2.global_opcode() {
                Ok(execute_pos2_e2_impl::<_, _, SBOX_REGISTERS, true>)
            } else {
                Ok(execute_pos2_e2_impl::<_, _, SBOX_REGISTERS, false>)
            }
        } else {
            let pre_compute: &mut E2PreCompute<VerifyBatchPreCompute<F, SBOX_REGISTERS>> =
                data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_verify_batch_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_verify_batch_e2_impl::<_, _, SBOX_REGISTERS>)
        }
    }
}

unsafe fn execute_pos2_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const SBOX_REGISTERS: usize,
    const IS_PERM: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Pos2PreCompute<F, SBOX_REGISTERS> = pre_compute.borrow();
    execute_pos2_e12_impl::<_, _, SBOX_REGISTERS, IS_PERM>(pre_compute, vm_state);
}

unsafe fn execute_pos2_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const SBOX_REGISTERS: usize,
    const IS_PERM: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Pos2PreCompute<F, SBOX_REGISTERS>> = pre_compute.borrow();
    let height =
        execute_pos2_e12_impl::<_, _, SBOX_REGISTERS, IS_PERM>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

unsafe fn execute_verify_batch_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const SBOX_REGISTERS: usize,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &VerifyBatchPreCompute<F, SBOX_REGISTERS> = pre_compute.borrow();
    execute_verify_batch_e12_impl::<_, _, SBOX_REGISTERS>(pre_compute, vm_state);
}

unsafe fn execute_verify_batch_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const SBOX_REGISTERS: usize,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<VerifyBatchPreCompute<F, SBOX_REGISTERS>> = pre_compute.borrow();
    let height = execute_verify_batch_e12_impl::<_, _, SBOX_REGISTERS>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

#[inline(always)]
unsafe fn execute_pos2_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const SBOX_REGISTERS: usize,
    const IS_PERM: bool,
>(
    pre_compute: &Pos2PreCompute<F, SBOX_REGISTERS>,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) -> u32 {
    let subchip = pre_compute.subchip;

    let [output_pointer]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.output_register);
    let [input_pointer_1]: [F; 1] =
        vm_state.vm_read(AS::Native as u32, pre_compute.input_register_1);
    let [input_pointer_2] = if IS_PERM {
        [input_pointer_1 + F::from_canonical_usize(CHUNK)]
    } else {
        vm_state.vm_read(AS::Native as u32, pre_compute.input_register_2)
    };

    let data_1: [F; CHUNK] =
        vm_state.vm_read(AS::Native as u32, input_pointer_1.as_canonical_u32());
    let data_2: [F; CHUNK] =
        vm_state.vm_read(AS::Native as u32, input_pointer_2.as_canonical_u32());

    let p2_input = std::array::from_fn(|i| {
        if i < CHUNK {
            data_1[i]
        } else {
            data_2[i - CHUNK]
        }
    });
    let output = subchip.permute(p2_input);
    let output_pointer_u32 = output_pointer.as_canonical_u32();

    vm_state.vm_write::<F, CHUNK>(
        AS::Native as u32,
        output_pointer_u32,
        &std::array::from_fn(|i| output[i]),
    );
    if IS_PERM {
        vm_state.vm_write::<F, CHUNK>(
            AS::Native as u32,
            output_pointer_u32 + CHUNK as u32,
            &std::array::from_fn(|i| output[i + CHUNK]),
        );
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    1
}

#[inline(always)]
unsafe fn execute_verify_batch_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const SBOX_REGISTERS: usize,
>(
    pre_compute: &VerifyBatchPreCompute<F, SBOX_REGISTERS>,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) -> u32 {
    // TODO: Add a flag `optimistic_execution`. When the flag is true, we trust all inputs
    // and skip all input validation computation during E1 execution.

    let subchip = pre_compute.subchip;
    let opened_element_size = pre_compute.opened_element_size;

    let [proof_id]: [F; 1] = vm_state.host_read(AS::Native as u32, pre_compute.proof_id_ptr);
    let [dim_base_pointer]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.dim_register);
    let dim_base_pointer_u32 = dim_base_pointer.as_canonical_u32();
    let [opened_base_pointer]: [F; 1] =
        vm_state.vm_read(AS::Native as u32, pre_compute.opened_register);
    let opened_base_pointer_u32 = opened_base_pointer.as_canonical_u32();
    let [opened_length]: [F; 1] =
        vm_state.vm_read(AS::Native as u32, pre_compute.opened_length_register);
    let [index_base_pointer]: [F; 1] =
        vm_state.vm_read(AS::Native as u32, pre_compute.index_register);
    let index_base_pointer_u32 = index_base_pointer.as_canonical_u32();
    let [commit_pointer]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.commit_register);
    let commit: [F; CHUNK] = vm_state.vm_read(AS::Native as u32, commit_pointer.as_canonical_u32());

    let opened_length = opened_length.as_canonical_u32() as usize;

    let initial_log_height = {
        let [height]: [F; 1] = vm_state.host_read(AS::Native as u32, dim_base_pointer_u32);
        height.as_canonical_u32()
    };

    let mut log_height = initial_log_height as i32;
    let mut sibling_index = 0;
    let mut opened_index = 0;
    let mut height = 0;

    let mut root = [F::ZERO; CHUNK];
    let sibling_proof: Vec<[F; CHUNK]> = {
        let proof_idx = proof_id.as_canonical_u32() as usize;
        vm_state.streams.hint_space[proof_idx]
            .par_chunks(CHUNK)
            .map(|c| c.try_into().unwrap())
            .collect()
    };

    while log_height >= 0 {
        if opened_index < opened_length
            && vm_state.host_read::<F, 1>(
                AS::Native as u32,
                dim_base_pointer_u32 + opened_index as u32,
            )[0] == F::from_canonical_u32(log_height as u32)
        {
            let initial_opened_index = opened_index;

            let mut row_pointer = 0;
            let mut row_end = 0;

            let mut rolling_hash = [F::ZERO; 2 * CHUNK];

            let mut is_first_in_segment = true;

            loop {
                let mut cells_len = 0;
                for chunk_elem in rolling_hash.iter_mut().take(CHUNK) {
                    if is_first_in_segment || row_pointer == row_end {
                        if is_first_in_segment {
                            is_first_in_segment = false;
                        } else {
                            opened_index += 1;
                            if opened_index == opened_length
                                || vm_state.host_read::<F, 1>(
                                    AS::Native as u32,
                                    dim_base_pointer_u32 + opened_index as u32,
                                )[0] != F::from_canonical_u32(log_height as u32)
                            {
                                break;
                            }
                        }
                        let [new_row_pointer, row_len]: [F; 2] = vm_state.vm_read(
                            AS::Native as u32,
                            opened_base_pointer_u32 + 2 * opened_index as u32,
                        );
                        row_pointer = new_row_pointer.as_canonical_u32() as usize;
                        row_end = row_pointer
                            + (opened_element_size * row_len).as_canonical_u32() as usize;
                    }
                    let [value]: [F; 1] = vm_state.vm_read(AS::Native as u32, row_pointer as u32);
                    cells_len += 1;
                    *chunk_elem = value;
                    row_pointer += 1;
                }
                if cells_len == 0 {
                    break;
                }
                height += 1;
                subchip.permute_mut(&mut rolling_hash);
                if cells_len < CHUNK {
                    break;
                }
            }

            let final_opened_index = opened_index - 1;
            let [height_check]: [F; 1] = vm_state.host_read(
                AS::Native as u32,
                dim_base_pointer_u32 + initial_opened_index as u32,
            );
            assert_eq!(height_check, F::from_canonical_u32(log_height as u32));
            let [height_check]: [F; 1] = vm_state.host_read(
                AS::Native as u32,
                dim_base_pointer_u32 + final_opened_index as u32,
            );
            assert_eq!(height_check, F::from_canonical_u32(log_height as u32));

            let hash: [F; CHUNK] = std::array::from_fn(|i| rolling_hash[i]);

            let new_root = if log_height as u32 == initial_log_height {
                hash
            } else {
                let (_, new_root) = compress(subchip, root, hash);
                new_root
            };
            root = new_root;
            height += 1;
        }

        if log_height != 0 {
            let [sibling_is_on_right]: [F; 1] = vm_state.vm_read(
                AS::Native as u32,
                index_base_pointer_u32 + sibling_index as u32,
            );
            let sibling_is_on_right = sibling_is_on_right == F::ONE;
            let sibling = sibling_proof[sibling_index];
            let (_, new_root) = if sibling_is_on_right {
                compress(subchip, sibling, root)
            } else {
                compress(subchip, root, sibling)
            };
            root = new_root;
            height += 1;
        }

        log_height -= 1;
        sibling_index += 1;
    }

    assert_eq!(commit, root);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    height
}
