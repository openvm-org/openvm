use std::sync::{Arc, Mutex};

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor},
    system::{
        memory::{offline_checker::MemoryBridge, MemoryController, OfflineMemory, RecordId},
        program::ProgramBus,
    },
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, Poseidon2Opcode::PERM_POS2, VmOpcode,
};
use openvm_native_compiler::VerifyBatchOpcode::VERIFY_BATCH;
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubAir, Poseidon2SubChip};
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    Stateful,
};
use serde::{Deserialize, Serialize};

use crate::verify_batch::{
    air::{VerifyBatchAir, VerifyBatchBus},
    CHUNK,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct VerifyBatchRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub instruction: Instruction<F>,

    pub dim_base_pointer: F,
    pub opened_base_pointer: F,
    pub opened_length: usize,
    pub sibling_base_pointer: F,
    pub index_base_pointer: F,
    pub commit_pointer: F,

    pub dim_base_pointer_read: RecordId,
    pub opened_base_pointer_read: RecordId,
    pub opened_length_read: RecordId,
    pub sibling_base_pointer_read: RecordId,
    pub index_base_pointer_read: RecordId,
    pub commit_pointer_read: RecordId,

    pub commit_read: RecordId,
    pub initial_height: usize,
    pub top_level: Vec<TopLevelRecord<F>>,
}

impl<F: PrimeField32> VerifyBatchRecord<F> {
    pub fn opened_element_size_inv(&self) -> F {
        self.instruction.g
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct TopLevelRecord<F: Field> {
    // must be present in first record
    pub incorporate_row: Option<IncorporateRowRecord<F>>,
    // must be present in all bust last record
    pub incorporate_sibling: Option<IncorporateSiblingRecord<F>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct IncorporateSiblingRecord<F: Field> {
    pub read_sibling_array_start: RecordId,
    pub read_root_is_on_right: RecordId,
    pub root_is_on_right: bool,
    pub reads: [RecordId; CHUNK],
    pub p2_input: [F; 2 * CHUNK],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct IncorporateRowRecord<F: Field> {
    pub chunks: Vec<InsideRowRecord<F>>,
    pub initial_opened_index: usize,
    pub final_opened_index: usize,
    pub initial_height_read: RecordId,
    pub final_height_read: RecordId,
    pub p2_input: [F; 2 * CHUNK],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct InsideRowRecord<F: Field> {
    pub cells: Vec<CellRecord>,
    pub p2_input: [F; 2 * CHUNK],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRecord {
    pub read: RecordId,
    pub opened_index: usize,
    pub read_row_pointer_and_length: Option<RecordId>,
    pub row_pointer: usize,
    pub row_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct SimplePermuteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub instruction: Instruction<F>,

    pub read_input_pointer: RecordId,
    pub read_output_pointer: RecordId,
    pub read_data: RecordId,
    pub write_data: RecordId,

    pub input_pointer: F,
    pub output_pointer: F,
    pub p2_input: [F; 2 * CHUNK],
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(bound = "F: Field")]
pub struct VerifyBatchRecordSet<F: Field> {
    pub verify_batch_records: Vec<VerifyBatchRecord<F>>,
    pub simple_permute_records: Vec<SimplePermuteRecord<F>>,
}

pub struct VerifyBatchChip<F: Field, const SBOX_REGISTERS: usize> {
    pub(super) air: VerifyBatchAir<F, SBOX_REGISTERS>,
    pub(super) record_set: VerifyBatchRecordSet<F>,
    pub(super) height: usize,
    pub(super) offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    pub(super) subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> Stateful<Vec<u8>>
    for VerifyBatchChip<F, SBOX_REGISTERS>
{
    fn load_state(&mut self, state: Vec<u8>) {
        self.record_set = bitcode::deserialize(&state).unwrap();
        self.height = self.record_set.simple_permute_records.len();
        for record in self.record_set.verify_batch_records.iter() {
            for top_level in record.top_level.iter() {
                if let Some(incorporate_row) = &top_level.incorporate_row {
                    self.height += 1 + incorporate_row.chunks.len();
                }
                if let Some(_) = &top_level.incorporate_sibling {
                    self.height += 1;
                }
            }
        }
    }

    fn store_state(&self) -> Vec<u8> {
        bitcode::serialize(&self.record_set).unwrap()
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> VerifyBatchChip<F, SBOX_REGISTERS> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        verify_batch_offset: usize,
        perm_pos2_offset: usize,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
        poseidon2_config: Poseidon2Config<F>,
    ) -> Self {
        let air = VerifyBatchAir {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            internal_bus: VerifyBatchBus(7),
            subair: Arc::new(Poseidon2SubAir::new(poseidon2_config.constants.into())),
            verify_batch_offset,
            perm_pos2_offset,
            address_space: F::from_canonical_u32(5),
        };
        Self {
            record_set: Default::default(),
            air,
            height: 0,
            offline_memory,
            subchip: Poseidon2SubChip::new(poseidon2_config.constants),
        }
    }

    fn compress(&self, left: [F; CHUNK], right: [F; CHUNK]) -> ([F; 2 * CHUNK], [F; CHUNK]) {
        let concatenated =
            std::array::from_fn(|i| if i < CHUNK { left[i] } else { right[i - CHUNK] });
        let permuted = self.subchip.permute(concatenated);
        (concatenated, std::array::from_fn(|i| permuted[i]))
    }
}

pub(super) const NUM_INITIAL_READS: usize = 7;

impl<F: PrimeField32, const SBOX_REGISTERS: usize> InstructionExecutor<F>
    for VerifyBatchChip<F, SBOX_REGISTERS>
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        if instruction.opcode == VmOpcode::with_default_offset(PERM_POS2) {
            let &Instruction {
                a: output_register,
                b: input_register,
                d: register_address_space,
                e: data_address_space,
                ..
            } = instruction;

            let (read_output_pointer, output_pointer) =
                memory.read_cell(register_address_space, output_register);
            let (read_input_pointer, input_pointer) =
                memory.read_cell(register_address_space, input_register);
            let (read_data, data) = memory.read(data_address_space, input_pointer);
            let output = self.subchip.permute(data);
            let (write_data, _) = memory.write(data_address_space, output_pointer, output);

            self.record_set
                .simple_permute_records
                .push(SimplePermuteRecord {
                    from_state,
                    instruction: instruction.clone(),
                    read_input_pointer,
                    read_output_pointer,
                    read_data,
                    write_data,
                    input_pointer,
                    output_pointer,
                    p2_input: data,
                });
            self.height += 1;
        } else if instruction.opcode == VmOpcode::with_default_offset(VERIFY_BATCH) {
            let &Instruction {
                a: dim_register,
                b: opened_register,
                c: opened_length_register,
                d: sibling_register,
                e: index_register,
                f: commit_register,
                g: opened_element_size_inv,
                ..
            } = instruction;
            let address_space = self.air.address_space;
            // calc inverse fast assuming opened_element_size in {1, 4}
            let mut opened_element_size = F::ONE;
            while opened_element_size * opened_element_size_inv != F::ONE {
                opened_element_size += F::ONE;
            }

            let (dim_base_pointer_read, dim_base_pointer) =
                memory.read_cell(address_space, dim_register);
            let (opened_base_pointer_read, opened_base_pointer) =
                memory.read_cell(address_space, opened_register);
            let (opened_length_read, opened_length) =
                memory.read_cell(address_space, opened_length_register);
            let (sibling_base_pointer_read, sibling_base_pointer) =
                memory.read_cell(address_space, sibling_register);
            let (index_base_pointer_read, index_base_pointer) =
                memory.read_cell(address_space, index_register);
            let (commit_pointer_read, commit_pointer) =
                memory.read_cell(address_space, commit_register);
            let (commit_read, commit) = memory.read(address_space, commit_pointer);

            let opened_length = opened_length.as_canonical_u32() as usize;

            let initial_height = memory
                .unsafe_read_cell(address_space, dim_base_pointer)
                .as_canonical_u32();
            let mut height = initial_height;
            let mut proof_index = 0;
            let mut opened_index = 0;
            let mut top_level = vec![];

            let mut root = [F::ZERO; CHUNK];

            while height >= 1 {
                let incorporate_row = if opened_index < opened_length
                    && memory.unsafe_read_cell(
                        address_space,
                        dim_base_pointer + F::from_canonical_usize(opened_index),
                    ) == F::from_canonical_u32(height)
                {
                    let initial_opened_index = opened_index;
                    for _ in 0..NUM_INITIAL_READS {
                        memory.increment_timestamp();
                    }
                    let mut chunks = vec![];

                    let mut row_pointer = 0;
                    let mut row_end = 0;

                    let mut prev_rolling_hash: Option<[F; 2 * CHUNK]> = None;
                    let mut rolling_hash = [F::ZERO; 2 * CHUNK];

                    let mut is_first_in_segment = true;

                    loop {
                        let mut cells = vec![];
                        for i in 0..CHUNK {
                            let read_row_pointer_and_length = if is_first_in_segment
                                || row_pointer == row_end
                            {
                                if is_first_in_segment {
                                    is_first_in_segment = false;
                                } else {
                                    opened_index += 1;
                                    if opened_index == opened_length
                                        || memory.unsafe_read_cell(
                                            address_space,
                                            dim_base_pointer
                                                + F::from_canonical_usize(opened_index),
                                        ) != F::from_canonical_u32(height)
                                    {
                                        break;
                                    }
                                }
                                let (result, [new_row_pointer, row_len]) = memory.read(
                                    address_space,
                                    opened_base_pointer + F::from_canonical_usize(2 * opened_index),
                                );
                                row_pointer = new_row_pointer.as_canonical_u32() as usize;
                                row_end = row_pointer
                                    + (opened_element_size * row_len).as_canonical_u32() as usize;
                                Some(result)
                            } else {
                                memory.increment_timestamp();
                                None
                            };
                            let (read, value) = memory
                                .read_cell(address_space, F::from_canonical_usize(row_pointer));
                            cells.push(CellRecord {
                                read,
                                opened_index,
                                read_row_pointer_and_length,
                                row_pointer,
                                row_end,
                            });
                            rolling_hash[i] = value;
                            row_pointer += 1;
                        }
                        if cells.is_empty() {
                            break;
                        }
                        let cells_len = cells.len();
                        chunks.push(InsideRowRecord {
                            cells,
                            p2_input: rolling_hash,
                        });
                        self.height += 1;
                        prev_rolling_hash = Some(rolling_hash);
                        self.subchip.permute_mut(&mut rolling_hash);
                        if cells_len < CHUNK {
                            for _ in 0..CHUNK - cells_len {
                                memory.increment_timestamp();
                                memory.increment_timestamp();
                            }
                            break;
                        }
                    }
                    let final_opened_index = opened_index - 1;
                    let (initial_height_read, height_check) = memory.read_cell(
                        address_space,
                        dim_base_pointer + F::from_canonical_usize(initial_opened_index),
                    );
                    assert_eq!(height_check, F::from_canonical_u32(height));
                    let (final_height_read, height_check) = memory.read_cell(
                        address_space,
                        dim_base_pointer + F::from_canonical_usize(final_opened_index),
                    );
                    assert_eq!(height_check, F::from_canonical_u32(height));

                    let hash: [F; CHUNK] = std::array::from_fn(|i| rolling_hash[i]);

                    let (p2_input, new_root) = if height == initial_height {
                        (prev_rolling_hash.unwrap(), hash)
                    } else {
                        self.compress(root, hash)
                    };
                    root = new_root;

                    self.height += 1;
                    Some(IncorporateRowRecord {
                        chunks,
                        initial_opened_index,
                        final_opened_index,
                        initial_height_read,
                        final_height_read,
                        p2_input,
                    })
                } else {
                    None
                };

                let incorporate_sibling = if height == 1 {
                    None
                } else {
                    for _ in 0..NUM_INITIAL_READS {
                        memory.increment_timestamp();
                    }

                    let (read_root_is_on_right, root_is_on_right) = memory.read_cell(
                        address_space,
                        index_base_pointer + F::from_canonical_usize(proof_index),
                    );
                    let root_is_on_right = root_is_on_right == F::ONE;

                    let (read_sibling_array_start, sibling_array_start) = memory.read_cell(
                        address_space,
                        sibling_base_pointer + F::from_canonical_usize(2 * proof_index),
                    );
                    let sibling_array_start = sibling_array_start.as_canonical_u32() as usize;

                    let mut sibling = [F::ZERO; CHUNK];
                    let mut reads = vec![];
                    for i in 0..CHUNK {
                        let (read, value) = memory.read_cell(
                            address_space,
                            F::from_canonical_usize(sibling_array_start + i),
                        );
                        sibling[i] = value;
                        reads.push(read);
                    }

                    let (p2_input, new_root) = if root_is_on_right {
                        self.compress(sibling, root)
                    } else {
                        self.compress(root, sibling)
                    };
                    root = new_root;

                    self.height += 1;
                    Some(IncorporateSiblingRecord {
                        read_sibling_array_start,
                        read_root_is_on_right,
                        root_is_on_right,
                        reads: std::array::from_fn(|i| reads[i]),
                        p2_input,
                    })
                };

                top_level.push(TopLevelRecord {
                    incorporate_row,
                    incorporate_sibling,
                });

                height /= 2;
                proof_index += 1;
            }

            assert_eq!(commit, root);
            self.record_set
                .verify_batch_records
                .push(VerifyBatchRecord {
                    from_state,
                    instruction: instruction.clone(),
                    dim_base_pointer,
                    opened_base_pointer,
                    opened_length,
                    sibling_base_pointer,
                    index_base_pointer,
                    commit_pointer,
                    dim_base_pointer_read,
                    opened_base_pointer_read,
                    opened_length_read,
                    sibling_base_pointer_read,
                    index_base_pointer_read,
                    commit_pointer_read,
                    commit_read,
                    initial_height: initial_height as usize,
                    top_level,
                });
        } else {
            unreachable!()
        }
        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == (VERIFY_BATCH as usize) + self.air.verify_batch_offset {
            return String::from("VERIFY_BATCH");
        } else if opcode == (PERM_POS2 as usize) + self.air.perm_pos2_offset {
            return String::from("PERM_POS2");
        } else {
            unreachable!()
        }
    }
}
