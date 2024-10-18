use std::{borrow::Borrow, marker::PhantomData};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{Rv32TerminateNopOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::Rv32TerminateNopProcessedInstruction,
    system::program::Instruction,
};

#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32TerminateNopCols<T> {
    pub flag: T, // [invalid, terminate, nop]
}

#[derive(Debug, Clone)]
pub struct Rv32TerminateNopCoreAir<F: Field> {
    pub _marker: PhantomData<F>,
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for Rv32TerminateNopCoreAir<F> {
    fn width(&self) -> usize {
        Rv32TerminateNopCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32TerminateNopCoreAir<F> {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for Rv32TerminateNopCoreAir<AB::F>
where
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<()>,
    I::Writes: From<()>,
    I::ProcessedInstruction: From<Rv32TerminateNopProcessedInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32TerminateNopCols<AB::Var> = (*local_core).borrow();
        let flag = cols.flag;
        builder.assert_zero(flag * (flag - AB::Expr::one()) * (flag - AB::Expr::two()));
        AdapterAirContext {
            to_pc: Some(from_pc + (flag - AB::Expr::one()) * AB::Expr::from_canonical_u32(4)),
            reads: ().into(),
            writes: ().into(),
            instruction: Rv32TerminateNopProcessedInstruction { flag: flag.into() }.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32TerminateNopCoreChip<F: Field> {
    pub air: Rv32TerminateNopCoreAir<F>,
}

impl<F: Field> Rv32TerminateNopCoreChip<F> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: Rv32TerminateNopCoreAir::<F> {
                _marker: PhantomData,
                offset,
            },
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for Rv32TerminateNopCoreChip<F>
where
    I::Writes: From<()>,
{
    type Record = Rv32TerminateNopOpcode;
    type Air = Rv32TerminateNopCoreAir<F>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let opcode = Rv32TerminateNopOpcode::from_usize(instruction.opcode - self.air.offset);
        let next_pc = solve_terminate_nop(opcode, from_pc);
        Ok((
            AdapterRuntimeContext {
                to_pc: next_pc,
                writes: ().into(),
            },
            opcode,
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32TerminateNopOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        row_slice[0] = if record == Rv32TerminateNopOpcode::TERMINATE {
            F::one()
        } else {
            F::two()
        };
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns Some(next pc) or None if terminate
pub(super) fn solve_terminate_nop(opcode: Rv32TerminateNopOpcode, pc: u32) -> Option<u32> {
    if opcode == Rv32TerminateNopOpcode::NOP {
        Some(pc + 4)
    } else {
        None
    }
}
