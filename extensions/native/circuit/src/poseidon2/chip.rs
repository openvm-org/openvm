use std::{
    array::from_fn,
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor},
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, Poseidon2Opcode, UsizeOpcode,
};
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::p3_field::{Field, PrimeField32};

use super::{
    NativePoseidon2Air, NativePoseidon2MemoryCols, NATIVE_POSEIDON2_CHUNK_SIZE,
    NATIVE_POSEIDON2_WIDTH,
};

#[derive(Debug)]
pub struct NativePoseidon2BaseChip<F: Field, const SBOX_REGISTERS: usize> {
    pub air: Arc<NativePoseidon2Air<F, SBOX_REGISTERS>>,
    pub subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    pub records: Vec<Option<NativePoseidon2ChipRecord<F>>>,
    pub offline_memory: Arc<Mutex<OfflineMemory<F>>>,
}

#[derive(Clone, Debug)]
pub struct NativePoseidon2ChipRecord<F> {
    pub from_state: ExecutionState<u32>,
    pub opcode: Poseidon2Opcode,
    pub input: [F; NATIVE_POSEIDON2_WIDTH],
    pub c: F,
    pub rd: RecordId,
    pub rs1: RecordId,
    pub rs2: Option<RecordId>,

    pub read1: RecordId,
    pub read2: RecordId,

    pub write1: RecordId,
    pub write2: Option<RecordId>,
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2BaseChip<F, SBOX_REGISTERS> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        poseidon2_config: Poseidon2Config<F>,
        offset: usize,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        Self {
            air: Arc::new(NativePoseidon2Air::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                subchip.air.clone(),
                offset,
            )),
            subchip,
            records: vec![],
            offline_memory,
        }
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> InstructionExecutor<F>
    for NativePoseidon2BaseChip<F, SBOX_REGISTERS>
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let local_opcode = Poseidon2Opcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let rd = memory.read_cell(d, a);
        let rs1 = memory.read_cell(d, b);
        let rs2 = match local_opcode {
            Poseidon2Opcode::PERM_POS2 => {
                memory.increment_timestamp();
                None
            }
            Poseidon2Opcode::COMP_POS2 => Some(memory.read_cell(d, c)),
        };

        let read1 = memory.read::<NATIVE_POSEIDON2_CHUNK_SIZE>(e, rs1.1);
        let read2 = memory.read::<NATIVE_POSEIDON2_CHUNK_SIZE>(
            e,
            match rs2 {
                Some(rs2) => rs2.1,
                None => rs1.1 + F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE),
            },
        );

        let mut input_state: [F; NATIVE_POSEIDON2_WIDTH] = [F::ZERO; NATIVE_POSEIDON2_WIDTH];
        input_state[..NATIVE_POSEIDON2_CHUNK_SIZE].copy_from_slice(&read1.1);
        input_state[NATIVE_POSEIDON2_CHUNK_SIZE..].copy_from_slice(&read2.1);

        let output_state = self.subchip.permute(input_state);

        let output1: [F; NATIVE_POSEIDON2_CHUNK_SIZE] = from_fn(|i| output_state[i]);
        let output2: [F; NATIVE_POSEIDON2_CHUNK_SIZE] =
            from_fn(|i| output_state[NATIVE_POSEIDON2_CHUNK_SIZE + i]);

        let (write1, _) = memory.write::<NATIVE_POSEIDON2_CHUNK_SIZE>(e, rd.1, output1);
        let write2 = match local_opcode {
            Poseidon2Opcode::PERM_POS2 => Some(
                memory
                    .write::<NATIVE_POSEIDON2_CHUNK_SIZE>(
                        e,
                        rd.1 + F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE),
                        output2,
                    )
                    .0,
            ),
            Poseidon2Opcode::COMP_POS2 => {
                memory.increment_timestamp();
                None
            }
        };

        self.records.push(Some(NativePoseidon2ChipRecord {
            from_state,
            opcode: local_opcode,
            input: input_state,
            c,
            rd: rd.0,
            rs1: rs1.0,
            rs2: rs2.map(|rs2| rs2.0),
            read1: read1.0,
            read2: read2.0,
            write1,
            write2,
        }));

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Poseidon2Opcode::from_usize(opcode - self.air.offset)
        )
    }
}

impl<F: PrimeField32 + Sync> NativePoseidon2ChipRecord<F> {
    pub fn to_memory_cols(&self, memory: &OfflineMemory<F>) -> NativePoseidon2MemoryCols<F> {
        let aux_cols_factory = memory.aux_cols_factory();
        let rs1 = memory.record_by_id(self.rs1);
        let (rs2_ptr, rs2_val, rs2_read_aux) = match self.rs2 {
            Some(rs2) => {
                let rs2 = memory.record_by_id(rs2);
                (
                    rs2.pointer,
                    rs2.data[0],
                    aux_cols_factory.make_read_aux_cols(rs2),
                )
            }
            None => (
                F::ZERO,
                rs1.data[0] + F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE),
                MemoryReadAuxCols::disabled(),
            ),
        };
        let rd = memory.record_by_id(self.rd);
        let read1 = memory.record_by_id(self.read1);
        let read2 = memory.record_by_id(self.read2);
        NativePoseidon2MemoryCols {
            from_state: self.from_state.map(F::from_canonical_u32),
            opcode_flag: match self.opcode {
                Poseidon2Opcode::PERM_POS2 => F::ONE,
                Poseidon2Opcode::COMP_POS2 => F::TWO,
            },
            ptr_as: rd.address_space,
            chunk_as: read1.address_space,
            c: self.c,
            rs_ptr: [rs1.pointer, rs2_ptr],
            rd_ptr: rd.pointer,
            rs_val: [rs1.data[0], rs2_val],
            rd_val: rd.data[0],
            rs_read_aux: [aux_cols_factory.make_read_aux_cols(rs1), rs2_read_aux],
            rd_read_aux: aux_cols_factory.make_read_aux_cols(rd),
            chunk_read_aux: [
                aux_cols_factory.make_read_aux_cols(read1),
                aux_cols_factory.make_read_aux_cols(read2),
            ],
            chunk_write_aux: [
                aux_cols_factory.make_write_aux_cols(memory.record_by_id(self.write1)),
                self.write2.map_or(MemoryWriteAuxCols::disabled(), |w| {
                    aux_cols_factory.make_write_aux_cols(memory.record_by_id(w))
                }),
            ],
        }
    }
}
