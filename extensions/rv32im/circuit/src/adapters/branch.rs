use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, get_record_from_slice, AdapterAirContext,
        AdapterExecutorE1, AdapterTraceFiller, AdapterTraceStep, BasicAdapterInterface,
        ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir, VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{memory_read_from_state, tracing_read};

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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32BranchAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub reads_aux: [MemoryReadAuxRecord; 2],
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
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = ();
    type RecordMut<'a> = &'a mut Rv32BranchAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, record: &mut &mut Rv32BranchAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        record: &mut &mut Rv32BranchAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { a, b, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);

        record.rs1_ptr = a.as_canonical_u32();
        let rs1 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            a.as_canonical_u32(),
            &mut record.reads_aux[0].prev_timestamp,
        );
        record.rs2_ptr = b.as_canonical_u32();
        let rs2 = tracing_read(
            memory,
            RV32_REGISTER_AS,
            b.as_canonical_u32(),
            &mut record.reads_aux[1].prev_timestamp,
        );

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory<F>,
        _instruction: &Instruction<F>,
        _data: &Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // This function is intentionally left empty
    }
}
impl<F: PrimeField32, CTX> AdapterTraceFiller<F, CTX> for Rv32BranchAdapterStep {
    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32BranchAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32BranchAdapterCols<F> = adapter_row.borrow_mut();

        // We must assign in reverse
        let timestamp = record.from_timestamp;

        mem_helper.fill(
            record.reads_aux[1].prev_timestamp,
            timestamp + 1,
            adapter_row.reads_aux[1].as_mut(),
        );

        mem_helper.fill(
            record.reads_aux[0].prev_timestamp,
            timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );

        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);
        adapter_row.rs2_ptr = F::from_canonical_u32(record.rs2_ptr);
    }
}

impl<F> AdapterExecutorE1<F> for Rv32BranchAdapterStep
where
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 2];
    type WriteData = ();

    #[inline(always)]
    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { a, b, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
            memory_read_from_state(state, RV32_REGISTER_AS, a.as_canonical_u32());
        let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
            memory_read_from_state(state, RV32_REGISTER_AS, b.as_canonical_u32());

        [rs1, rs2]
    }

    #[inline(always)]
    fn write<Ctx>(
        &self,
        _state: &mut VmStateMut<GuestMemory, Ctx>,
        _instruction: &Instruction<F>,
        _data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
    }
}
