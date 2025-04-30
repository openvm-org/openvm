use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, BasicAdapterInterface,
        ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

use super::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{memory_read, tracing_read};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BranchAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BranchAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for Rv32BranchAdapterAir {
    fn width(&self) -> usize {
        Rv32BranchAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32BranchAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 2, 0, RV32_REGISTER_NUM_LIMBS, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32BranchAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs2_ptr),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BranchAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

/// Reads instructions of the form OP a, b, c, d, e where if(\[a:4\]_d op \[b:4\]_e) pc += c.
/// Operands d and e can only be 1.
#[derive(derive_new::new)]
pub struct Rv32BranchAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32BranchAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32BranchAdapterCols<u8>>();
    type ReadData = ([u8; RV32_REGISTER_NUM_LIMBS], [u8; RV32_REGISTER_NUM_LIMBS]);
    type WriteData = ();
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let &Instruction { a, b, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);

        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.rs1_ptr = a;
        let rs1 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            a.as_canonical_u32(),
            &mut adapter_row.reads_aux[0],
        );
        adapter_row.rs2_ptr = b;
        let rs2 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            b.as_canonical_u32(),
            &mut adapter_row.reads_aux[1],
        );

        (rs1, rs2)
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory<F>,
        _instruction: &Instruction<F>,
        _adapter_row: &mut [F],
        _data: &Self::WriteData,
    ) {
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[0].as_mut());
        timestamp += 1;

        mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[1].as_mut());
    }
}

impl<F> AdapterExecutorE1<F> for Rv32BranchAdapterStep
where
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = ([u8; RV32_REGISTER_NUM_LIMBS], [u8; RV32_REGISTER_NUM_LIMBS]);
    type WriteData = ();

    #[inline(always)]
    fn read<Mem>(&self, memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData
    where
        Mem: GuestMemory,
    {
        let Instruction { a, b, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
            memory_read(memory, RV32_REGISTER_AS, a.as_canonical_u32());
        let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
            memory_read(memory, RV32_REGISTER_AS, b.as_canonical_u32());

        (rs1, rs2)
    }

    #[inline(always)]
    fn write<Mem>(&self, _memory: &mut Mem, _instruction: &Instruction<F>, _data: &Self::WriteData)
    where
        Mem: GuestMemory,
    {
    }
}
