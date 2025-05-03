use std::{
    borrow::{Borrow, BorrowMut},
    collections::VecDeque,
    fmt::Debug,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, DynAdapterInterface, DynArray, ExecutionBridge,
        ExecutionState, MinimalInstruction, Result, VmAdapterAir,
    },
    system::memory::MemoryController,
};

// Replaces A: VmAdapterChip while testing VmCoreChip functionality, as it has no
// constraints and thus cannot cause a failure.
pub struct TestAdapterChip<F> {
    /// List of the return values of `preprocess` this chip should provide on each sequential call.
    pub prank_reads: VecDeque<Vec<F>>,
    /// List of `pc_inc` to use in `postprocess` on each sequential call.
    /// Defaults to `4` if not provided.
    pub prank_pc_inc: VecDeque<Option<u32>>,

    pub air: TestAdapterAir,
}

impl<F> TestAdapterChip<F> {
    pub fn new(
        prank_reads: Vec<Vec<F>>,
        prank_pc_inc: Vec<Option<u32>>,
        execution_bridge: ExecutionBridge,
    ) -> Self {
        Self {
            prank_reads: prank_reads.into(),
            prank_pc_inc: prank_pc_inc.into(),
            air: TestAdapterAir { execution_bridge },
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TestAdapterRecord<T> {
    pub from_pc: u32,
    pub operands: [T; 7],
}

#[derive(Clone, Copy, Debug)]
pub struct TestAdapterAir {
    pub execution_bridge: ExecutionBridge,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct TestAdapterCols<T> {
    pub from_pc: T,
    pub operands: [T; 7],
}

impl<F: Field> BaseAir<F> for TestAdapterAir {
    fn width(&self) -> usize {
        TestAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for TestAdapterAir {
    type Interface = DynAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let processed_instruction: MinimalInstruction<AB::Expr> = ctx.instruction.into();
        let cols: &TestAdapterCols<AB::Var> = local.borrow();
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                processed_instruction.opcode,
                cols.operands.to_vec(),
                ExecutionState {
                    pc: cols.from_pc.into(),
                    timestamp: AB::Expr::ONE,
                },
                AB::Expr::ZERO,
                (4, ctx.to_pc),
            )
            .eval(builder, processed_instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &TestAdapterCols<AB::Var> = local.borrow();
        cols.from_pc
    }
}
