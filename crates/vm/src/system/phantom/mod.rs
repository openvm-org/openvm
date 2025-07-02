use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, PhantomDiscriminant, SysPhantom,
    SystemOpcode, VmOpcode,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use super::memory::online::{GuestMemory, TracingMemory};
use crate::arch::{
    execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
    get_record_from_slice, EmptyMultiRowLayout, ExecutionBridge, ExecutionError, ExecutionState,
    InsExecutorE1, PcIncOrSet, PhantomSubExecutor, RecordArena, TraceFiller, TraceStep, VmStateMut,
};

#[cfg(test)]
mod tests;

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct PhantomRecord {
    pub pc: u32,
    pub operands: [u32; NUM_PHANTOM_OPERANDS],
    pub timestamp: u32,
}

/// `PhantomChip` is a special executor because it is stateful and stores all the phantom
/// sub-executors.
#[derive(derive_new::new)]
pub struct PhantomChip<F> {
    phantom_executors: FxHashMap<PhantomDiscriminant, Box<dyn PhantomSubExecutor<F>>>,
    phantom_opcode: VmOpcode,
}

impl<F> PhantomChip<F> {
    pub(crate) fn add_sub_executor<P: PhantomSubExecutor<F> + 'static>(
        &mut self,
        sub_executor: P,
        discriminant: PhantomDiscriminant,
    ) -> Option<Box<dyn PhantomSubExecutor<F>>> {
        self.phantom_executors
            .insert(discriminant, Box::new(sub_executor))
    }
}

impl<F> InsExecutorE1<F> for PhantomChip<F>
where
    F: PrimeField32,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction {
            opcode, a, b, c, ..
        } = instruction;
        assert_eq!(opcode, self.phantom_opcode);

        let c_u32 = c.as_canonical_u32();
        let discriminant = PhantomDiscriminant(c_u32 as u16);
        // If not a system phantom sub-instruction (which is handled in
        // ExecutionSegment), look for a phantom sub-executor to handle it.
        if SysPhantom::from_repr(discriminant.0).is_none() {
            let sub_executor = self.phantom_executors.get(&discriminant).ok_or_else(|| {
                ExecutionError::PhantomNotFound {
                    pc: *state.pc,
                    discriminant,
                }
            })?;
            // TODO(ayush): implement phantom subexecutor for new traits
            sub_executor
                .as_ref()
                .phantom_execute(
                    state.memory,
                    state.streams,
                    state.rng,
                    discriminant,
                    a.as_canonical_u32(),
                    b.as_canonical_u32(),
                    (c_u32 >> 16) as u16,
                )
                .map_err(|e| ExecutionError::Phantom {
                    pc: *state.pc,
                    discriminant,
                    inner: e,
                })?;
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<(), ExecutionError> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

impl<F> TraceStep<F> for PhantomChip<F>
where
    F: PrimeField32,
{
    type RecordLayout = EmptyMultiRowLayout;
    type RecordMut<'a> = &'a mut PhantomRecord;

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<'buf, F, TracingMemory<F>, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let record: &mut PhantomRecord = state.ctx.alloc(EmptyMultiRowLayout::default());
        record.pc = *state.pc;
        record.timestamp = state.memory.timestamp;
        let [a, b, c] = [instruction.a, instruction.b, instruction.c].map(|x| x.as_canonical_u32());
        record.operands = [a, b, c];

        let opcode = instruction.opcode;
        assert_eq!(opcode, self.phantom_opcode);

        let discriminant = PhantomDiscriminant(c as u16);
        // If not a system phantom sub-instruction (which is handled in
        // ExecutionSegment), look for a phantom sub-executor to handle it.
        if SysPhantom::from_repr(discriminant.0).is_none() {
            let sub_executor = self.phantom_executors.get(&discriminant).ok_or_else(|| {
                ExecutionError::PhantomNotFound {
                    pc: *state.pc,
                    discriminant,
                }
            })?;
            // TODO(ayush): implement phantom subexecutor for new traits
            sub_executor
                .as_ref()
                .phantom_execute(
                    &state.memory.data,
                    state.streams,
                    state.rng,
                    discriminant,
                    a,
                    b,
                    (c >> 16) as u16,
                )
                .map_err(|e| ExecutionError::Phantom {
                    pc: *state.pc,
                    discriminant,
                    inner: e,
                })?;
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        state.memory.increment_timestamp();

        Ok(())
    }

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", SystemOpcode::PHANTOM)
    }
}

impl<F> TraceFiller<F> for PhantomChip<F>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mut row_slice: &mut [F]) {
        // SAFETY: assume that row has size PhantomCols::<F>::width()
        let record: &PhantomRecord = unsafe { get_record_from_slice(&mut row_slice, ()) };
        let row: &mut PhantomCols<F> = row_slice.borrow_mut();
        // SAFETY: must assign in reverse order of column struct to prevent overwriting
        // borrowed data
        row.is_valid = F::ONE;
        row.timestamp = F::from_canonical_u32(record.timestamp);
        row.operands[2] = F::from_canonical_u32(record.operands[2]);
        row.operands[1] = F::from_canonical_u32(record.operands[1]);
        row.operands[0] = F::from_canonical_u32(record.operands[0]);
        row.pc = F::from_canonical_u32(record.pc)
    }
}
