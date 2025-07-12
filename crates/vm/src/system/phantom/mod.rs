use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant,
    SystemOpcode, VmOpcode,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use super::memory::MemoryController;
use crate::{
    arch::{
        ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor,
        PcIncOrSet, PhantomSubExecutor, Streams,
    },
    system::{
        phantom::execution::{execute_impl, PhantomOperands, PhantomStateMut},
        program::ProgramBus,
    },
};

mod execution;

/// PhantomAir still needs columns for each nonzero operand in a phantom instruction.
/// We currently allow `a,b,c` where the lower 16 bits of `c` are used as the [PhantomInstruction]
/// discriminant.
const NUM_PHANTOM_OPERANDS: usize = 3;

#[derive(Clone, Debug)]
pub struct PhantomAir {
    pub execution_bridge: ExecutionBridge,
    /// Global opcode for PhantomOpcode
    pub phantom_opcode: VmOpcode,
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Serialize, Deserialize)]
pub struct PhantomCols<T> {
    pub pc: T,
    #[serde(with = "BigArray")]
    pub operands: [T; NUM_PHANTOM_OPERANDS],
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
            operands,
            timestamp,
            is_valid,
        } = (*local).borrow();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                self.phantom_opcode.to_field::<AB::F>(),
                operands,
                ExecutionState::<AB::Expr>::new(pc, timestamp),
                AB::Expr::ONE,
                PcIncOrSet::Inc(AB::Expr::from_canonical_u32(DEFAULT_PC_STEP)),
            )
            .eval(builder, is_valid);
    }
}

pub struct PhantomChip<F> {
    pub air: PhantomAir,
    pub rows: Vec<PhantomCols<F>>,
    phantom_executors: FxHashMap<PhantomDiscriminant, Box<dyn PhantomSubExecutor<F>>>,
}

impl<F> PhantomChip<F> {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus, offset: usize) -> Self {
        Self {
            air: PhantomAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                phantom_opcode: VmOpcode::from_usize(offset + SystemOpcode::PHANTOM.local_usize()),
            },
            rows: vec![],
            phantom_executors: FxHashMap::default(),
        }
    }

    pub(crate) fn add_sub_executor<P: PhantomSubExecutor<F> + 'static>(
        &mut self,
        sub_executor: P,
        discriminant: PhantomDiscriminant,
    ) -> Option<Box<dyn PhantomSubExecutor<F>>> {
        self.phantom_executors
            .insert(discriminant, Box::new(sub_executor))
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for PhantomChip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let mut pc = from_state.pc;
        self.rows.push(PhantomCols {
            pc: F::from_canonical_u32(pc),
            operands: [instruction.a, instruction.b, instruction.c],
            timestamp: F::from_canonical_u32(memory.memory.timestamp),
            is_valid: F::ONE,
        });

        let c_u32 = instruction.c.as_canonical_u32();
        let sub_executor = self
            .phantom_executors
            .get(&PhantomDiscriminant(c_u32 as u16))
            .unwrap();
        execute_impl(
            PhantomStateMut {
                pc: &mut pc,
                memory: &mut memory.memory.data,
                streams,
                rng,
            },
            &PhantomOperands {
                a: instruction.a.as_canonical_u32(),
                b: instruction.b.as_canonical_u32(),
                c: instruction.c.as_canonical_u32(),
            },
            sub_executor,
        )?;
        pc += DEFAULT_PC_STEP;
        memory.increment_timestamp();

        Ok(ExecutionState {
            pc,
            timestamp: memory.memory.timestamp,
        })
    }

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", SystemOpcode::PHANTOM)
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
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let correct_height = self.rows.len().next_power_of_two();
        let width = PhantomCols::<Val<SC>>::width();
        let mut rows = Val::<SC>::zero_vec(width * correct_height);
        rows.par_chunks_mut(width)
            .zip(&self.rows)
            .for_each(|(row, row_record)| {
                let row: &mut PhantomCols<_> = row.borrow_mut();
                *row = *row_record;
            });
        let trace = RowMajorMatrix::new(rows, width);

        AirProofInput::simple(trace, vec![])
    }
}
