use std::{borrow::Borrow, iter, marker::PhantomData, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::Rv32DoNothingAdapterInterface;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::MemoryController,
        program::{bridge::ProgramBus, Instruction},
    },
};

#[derive(Debug, Clone)]
pub struct Rv32TerminateNopAdapterChip<F: Field> {
    _marker: PhantomData<F>,
    pub air: Rv32TerminateNopAdapterAir,
}

impl<F: PrimeField32> Rv32TerminateNopAdapterChip<F> {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus) -> Self {
        Self {
            _marker: PhantomData,
            air: Rv32TerminateNopAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32TerminateNopProcessedInstruction<T> {
    pub flag: T,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32TerminateNopAdapterCols<T> {
    pub from_state: ExecutionState<T>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32TerminateNopAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for Rv32TerminateNopAdapterAir {
    fn width(&self) -> usize {
        size_of::<Rv32TerminateNopAdapterCols<u8>>()
    }
}

impl<AB: InteractionBuilder> Air<AB> for Rv32TerminateNopAdapterAir {
    fn eval(&self, _builder: &mut AB) {}
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32TerminateNopAdapterAir {
    type Interface = Rv32DoNothingAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32TerminateNopAdapterCols<AB::Var> = (*local).borrow();
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                iter::empty::<AB::Expr>(),
                local_cols.from_state,
                ExecutionState {
                    pc: ctx.to_pc.unwrap(),
                    timestamp: local_cols.from_state.timestamp + AB::Expr::one(),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, _local: &[AB::Var]) -> AB::Var {
        todo!()
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32TerminateNopAdapterChip<F> {
    type ReadRecord = ();
    type WriteRecord = ();
    type Air = Rv32TerminateNopAdapterAir;
    type Interface = Rv32DoNothingAdapterInterface<F>;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        Ok(([], ()))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        _from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap(),
                timestamp: memory.timestamp(),
            },
            (),
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
