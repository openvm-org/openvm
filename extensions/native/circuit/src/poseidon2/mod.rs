use std::sync::{Arc, Mutex};

pub use columns::*;
use openvm_circuit::{
    arch::{ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor},
    system::{
        memory::{offline_checker::MemoryBridge, MemoryController, OfflineMemory},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::BytesStateful;
use openvm_instructions::instruction::Instruction;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::{Field, PrimeField32},
    prover::types::AirProofInput,
    rap::AnyRap,
    Chip, ChipUsageGetter,
};

mod air;
pub use air::*;
mod chip;
pub use chip::*;
mod columns;

mod trace;

#[cfg(test)]
mod tests;

pub const NATIVE_POSEIDON2_WIDTH: usize = 16;
pub const NATIVE_POSEIDON2_CHUNK_SIZE: usize = 8;

#[derive(BytesStateful)]
pub enum NativePoseidon2Chip<F: Field> {
    Register0(NativePoseidon2BaseChip<F, 0>),
    Register1(NativePoseidon2BaseChip<F, 1>),
}

impl<F: PrimeField32> NativePoseidon2Chip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        poseidon2_config: Poseidon2Config<F>,
        offset: usize,
        max_constraint_degree: usize,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        if max_constraint_degree >= 7 {
            Self::Register0(NativePoseidon2BaseChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                poseidon2_config,
                offset,
                offline_memory,
            ))
        } else {
            Self::Register1(NativePoseidon2BaseChip::new(
                execution_bus,
                program_bus,
                memory_bridge,
                poseidon2_config,
                offset,
                offline_memory,
            ))
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for NativePoseidon2Chip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.execute(memory, instruction, from_state),
            NativePoseidon2Chip::Register1(chip) => chip.execute(memory, instruction, from_state),
        }
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.get_opcode_name(opcode),
            NativePoseidon2Chip::Register1(chip) => chip.get_opcode_name(opcode),
        }
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for NativePoseidon2Chip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.air(),
            NativePoseidon2Chip::Register1(chip) => chip.air(),
        }
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.generate_air_proof_input(),
            NativePoseidon2Chip::Register1(chip) => chip.generate_air_proof_input(),
        }
    }
}

impl<F: PrimeField32> ChipUsageGetter for NativePoseidon2Chip<F> {
    fn air_name(&self) -> String {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.air_name(),
            NativePoseidon2Chip::Register1(chip) => chip.air_name(),
        }
    }

    fn current_trace_height(&self) -> usize {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.current_trace_height(),
            NativePoseidon2Chip::Register1(chip) => chip.current_trace_height(),
        }
    }

    fn trace_width(&self) -> usize {
        match self {
            NativePoseidon2Chip::Register0(chip) => chip.trace_width(),
            NativePoseidon2Chip::Register1(chip) => chip.trace_width(),
        }
    }
}
