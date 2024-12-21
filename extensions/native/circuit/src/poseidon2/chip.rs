use std::{array::from_fn, sync::Arc};

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor},
    system::{
        memory::{
            offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAuxColsFactory, MemoryControllerRef, MemoryReadRecord, MemoryWriteRecord,
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
    pub memory_controller: MemoryControllerRef<F>,
    pub records: Vec<Option<NativePoseidon2ChipRecord<F>>>,
}

#[derive(Clone, Debug)]
pub struct NativePoseidon2ChipRecord<F> {
    pub from_state: ExecutionState<u32>,
    pub opcode: Poseidon2Opcode,
    pub input: [F; NATIVE_POSEIDON2_WIDTH],
    pub c: F,
    pub rd: MemoryReadRecord<F, 1>,
    pub rs1: MemoryReadRecord<F, 1>,
    pub rs2: Option<MemoryReadRecord<F, 1>>,

    pub read1: MemoryReadRecord<F, NATIVE_POSEIDON2_CHUNK_SIZE>,
    pub read2: MemoryReadRecord<F, NATIVE_POSEIDON2_CHUNK_SIZE>,

    pub write1: MemoryWriteRecord<F, NATIVE_POSEIDON2_CHUNK_SIZE>,
    pub write2: Option<MemoryWriteRecord<F, NATIVE_POSEIDON2_CHUNK_SIZE>>,
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2BaseChip<F, SBOX_REGISTERS> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        poseidon2_config: Poseidon2Config<F>,
        offset: usize,
    ) -> Self {
        let memory_bridge = memory_controller.borrow().memory_bridge();
        let subchip = Poseidon2SubChip::new(poseidon2_config);
        Self {
            air: Arc::new(NativePoseidon2Air::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                subchip.air.clone(),
                offset,
            )),
            subchip,
            memory_controller,
            records: vec![],
        }
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> InstructionExecutor<F>
    for NativePoseidon2BaseChip<F, SBOX_REGISTERS>
{
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let local_opcode = Poseidon2Opcode::from_usize(opcode.local_opcode_idx(self.air.offset));
        let mut memory = self.memory_controller.borrow_mut();

        let rd = memory.read_cell(d, a);
        let rs1 = memory.read_cell(d, b);
        let rs2 = match local_opcode {
            Poseidon2Opcode::PERM_POS2 => {
                memory.increment_timestamp();
                None
            }
            Poseidon2Opcode::COMP_POS2 => Some(memory.read_cell(d, c)),
        };

        let read1 = memory.read::<NATIVE_POSEIDON2_CHUNK_SIZE>(e, rs1.data[0]);
        let read2 = memory.read::<NATIVE_POSEIDON2_CHUNK_SIZE>(
            e,
            match rs2 {
                Some(rs2) => rs2.data[0],
                None => rs1.data[0] + F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE),
            },
        );

        let mut input_state: [F; NATIVE_POSEIDON2_WIDTH] = [F::ZERO; NATIVE_POSEIDON2_WIDTH];
        input_state[..NATIVE_POSEIDON2_CHUNK_SIZE].copy_from_slice(&read1.data);
        input_state[NATIVE_POSEIDON2_CHUNK_SIZE..].copy_from_slice(&read2.data);

        let output_state = self.subchip.permute(input_state);

        let output1: [F; NATIVE_POSEIDON2_CHUNK_SIZE] = from_fn(|i| output_state[i]);
        let output2: [F; NATIVE_POSEIDON2_CHUNK_SIZE] =
            from_fn(|i| output_state[NATIVE_POSEIDON2_CHUNK_SIZE + i]);

        let write1 = memory.write::<NATIVE_POSEIDON2_CHUNK_SIZE>(e, rd.data[0], output1);
        let write2 = match local_opcode {
            Poseidon2Opcode::PERM_POS2 => Some(memory.write::<NATIVE_POSEIDON2_CHUNK_SIZE>(
                e,
                rd.data[0] + F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE),
                output2,
            )),
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
            rd,
            rs1,
            rs2,
            read1,
            read2,
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
    pub fn to_memory_cols(
        &self,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) -> NativePoseidon2MemoryCols<F> {
        let (rs2_ptr, rs2_val, rs2_read_aux) = match self.rs2 {
            Some(rs2) => (
                rs2.pointer,
                rs2.data[0],
                aux_cols_factory.make_read_aux_cols(rs2),
            ),
            None => (
                F::ZERO,
                self.rs1.data[0] + F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE),
                MemoryReadAuxCols::disabled(),
            ),
        };
        NativePoseidon2MemoryCols {
            from_state: self.from_state.map(F::from_canonical_u32),
            opcode_flag: match self.opcode {
                Poseidon2Opcode::PERM_POS2 => F::ONE,
                Poseidon2Opcode::COMP_POS2 => F::TWO,
            },
            ptr_as: self.rd.address_space,
            chunk_as: self.read1.address_space,
            c: self.c,
            rs_ptr: [self.rs1.pointer, rs2_ptr],
            rd_ptr: self.rd.pointer,
            rs_val: [self.rs1.data[0], rs2_val],
            rd_val: self.rd.data[0],
            rs_read_aux: [aux_cols_factory.make_read_aux_cols(self.rs1), rs2_read_aux],
            rd_read_aux: aux_cols_factory.make_read_aux_cols(self.rd),
            chunk_read_aux: [
                aux_cols_factory.make_read_aux_cols(self.read1),
                aux_cols_factory.make_read_aux_cols(self.read2),
            ],
            chunk_write_aux: [
                aux_cols_factory.make_write_aux_cols(self.write1),
                self.write2.map_or(MemoryWriteAuxCols::disabled(), |w| {
                    aux_cols_factory.make_write_aux_cols(w)
                }),
            ],
        }
    }
}
