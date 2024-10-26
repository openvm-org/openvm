use std::{borrow::Borrow, cell::RefCell, iter, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use axvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, CommonOpcode, PhantomInstruction,
    UsizeOpcode,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use parking_lot::Mutex;

use crate::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor, PcIncOrSet},
    system::{
        memory::MemoryControllerRef,
        program::{ExecutionError, ProgramBus},
        vm::Streams,
    },
};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct PhantomAir {
    pub execution_bridge: ExecutionBridge,
    /// Global opcode for PhantomOpcode
    pub phantom_opcode: usize,
}

#[derive(AlignedBorrow)]
pub struct PhantomCols<T> {
    pub pc: T,
    pub timestamp: T,
    pub is_valid: T,
}

impl<F: Field> BaseAir<F> for PhantomAir {
    fn width(&self) -> usize {
        PhantomCols::<F>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for PhantomAir {}
impl<F: Field> BaseAirWithPublicValues<F> for PhantomAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for PhantomAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let &PhantomCols {
            pc,
            timestamp,
            is_valid,
        } = (*local).borrow();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                AB::Expr::from_canonical_usize(self.phantom_opcode),
                iter::empty::<AB::Expr>(),
                ExecutionState::<AB::Expr>::new(pc, timestamp),
                AB::Expr::one(),
                PcIncOrSet::Inc(AB::Expr::from_canonical_u32(DEFAULT_PC_STEP)),
            )
            .eval(builder, is_valid);
    }
}

pub struct PhantomChip<F: Field> {
    pub air: PhantomAir,
    pub rows: Vec<PhantomCols<F>>,
    memory: MemoryControllerRef<F>,
    streams: Arc<Mutex<Streams<F>>>,
}

impl<F: Field> PhantomChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        streams: Arc<Mutex<Streams<F>>>,
        offset: usize,
    ) -> Self {
        Self {
            air: PhantomAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                phantom_opcode: offset + CommonOpcode::PHANTOM.as_usize(),
            },
            rows: vec![],
            memory: memory_controller,
            streams,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for PhantomChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction {
            opcode, a, b, c, d, ..
        } = instruction;
        assert_eq!(opcode, self.air.phantom_opcode);

        let phantom = PhantomInstruction::from_repr(c.as_canonical_u32() as usize).ok_or(
            ExecutionError::InvalidPhantomInstruction(from_state.pc, c.as_canonical_u32() as usize),
        )?;
        match phantom {
            PhantomInstruction::Nop => {}
            PhantomInstruction::PrintF => {
                let value = RefCell::borrow(&self.memory).unsafe_read_cell(d, a);
                println!("{}", value);
            }
            PhantomInstruction::HintInput => {
                let mut streams = self.streams.lock();
                let hint = match streams.input_stream.pop_front() {
                    Some(hint) => hint,
                    None => {
                        return Err(ExecutionError::EndOfInputStream(from_state.pc));
                    }
                };
                streams.hint_stream.clear();
                streams
                    .hint_stream
                    .push_back(F::from_canonical_usize(hint.len()));
                streams.hint_stream.extend(hint);
                drop(streams);
            }
            PhantomInstruction::HintBits => {
                let mut streams = self.streams.lock();
                let val = RefCell::borrow(&self.memory).unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = b.as_canonical_u32();
                streams.hint_stream.clear();
                for _ in 0..len {
                    streams
                        .hint_stream
                        .push_back(F::from_canonical_u32(val & 1));
                    val >>= 1;
                }
                drop(streams);
            }
            PhantomInstruction::HintBytes => {
                let mut streams = self.streams.lock();
                let val = RefCell::borrow(&self.memory).unsafe_read_cell(d, a);
                let mut val = val.as_canonical_u32();

                let len = b.as_canonical_u32();
                streams.hint_stream.clear();
                for _ in 0..len {
                    streams
                        .hint_stream
                        .push_back(F::from_canonical_u32(val & 0xff));
                    val >>= 8;
                }
                drop(streams);
            }
            _ => {}
        };

        self.rows.push(PhantomCols {
            pc: F::from_canonical_u32(from_state.pc),
            timestamp: F::from_canonical_u32(from_state.timestamp),
            is_valid: F::one(),
        });
        self.memory.borrow_mut().increment_timestamp();
        Ok(ExecutionState::new(
            from_state.pc + DEFAULT_PC_STEP,
            from_state.timestamp + 1,
        ))
    }

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", CommonOpcode::PHANTOM)
    }
}

impl<F: PrimeField32> ChipUsageGetter for PhantomChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.rows.len()
    }
    fn trace_width(&self) -> usize {
        PhantomCols::<F>::width()
    }
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for PhantomChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let curr_height = self.rows.len();
        let correct_height = self.rows.len().next_power_of_two();
        let width = PhantomCols::<Val<SC>>::width();

        let trace = RowMajorMatrix::new(
            self.rows
                .iter()
                .flat_map(|row| vec![row.pc, row.timestamp, row.is_valid])
                .chain(iter::repeat(Val::<SC>::zero()).take((correct_height - curr_height) * width))
                .collect::<Vec<_>>(),
            width,
        );
        AirProofInput::simple(self.air(), trace, vec![])
    }
}
