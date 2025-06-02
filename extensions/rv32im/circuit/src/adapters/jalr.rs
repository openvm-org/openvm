use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, AdapterAirContext, AdapterExecutorE1, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, SignedImmInstruction, VmAdapterAir,
        VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use super::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::{
    memory_read_from_state, memory_write_from_state, tracing_read, tracing_write,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32JalrAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// Only writes if `needs_write`.
    /// Sets `needs_write` to 0 iff `rd == x0`
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32JalrAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for Rv32JalrAdapterAir {
    fn width(&self) -> usize {
        Rv32JalrAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32JalrAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        SignedImmInstruction<AB::Expr>,
        1,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32JalrAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let write_count = local_cols.needs_write;

        builder.assert_bool(write_count);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs1_ptr,
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd_ptr,
                ),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));

        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                    write_count.into(),
                    ctx.instruction.imm_sign,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32JalrAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// This adapter reads from [b:4]_d (rs1) and writes to [a:4]_d (rd)
#[derive(derive_new::new)]
pub struct Rv32JalrAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for Rv32JalrAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32JalrAdapterCols<u8>>();
    type ReadData = [u8; RV32_REGISTER_NUM_LIMBS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type TraceContext<'a> = ();

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();
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
        let &Instruction { b, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.rs1_ptr = b;
        tracing_read(
            memory,
            RV32_REGISTER_AS,
            b.as_canonical_u32(),
            &mut adapter_row.rs1_aux_cols,
        )
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let &Instruction {
            a, d, f: enabled, ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        if enabled != F::ZERO {
            let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();

            adapter_row.needs_write = F::ONE;

            adapter_row.rd_ptr = a;
            tracing_write(
                memory,
                RV32_REGISTER_AS,
                a.as_canonical_u32(),
                data,
                &mut adapter_row.rd_aux_cols,
            );
        } else {
            memory.increment_timestamp();
        }
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        _trace_ctx: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut Rv32JalrAdapterCols<F> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        mem_helper.fill_from_prev(timestamp, adapter_row.rs1_aux_cols.as_mut());
        timestamp += 1;

        if adapter_row.needs_write.is_one() {
            mem_helper.fill_from_prev(timestamp, adapter_row.rd_aux_cols.as_mut());
        }
    }
}

impl<F> AdapterExecutorE1<F> for Rv32JalrAdapterStep
where
    F: PrimeField32,
{
    // TODO(ayush): directly use u32
    type ReadData = [u8; RV32_REGISTER_NUM_LIMBS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];

    #[inline(always)]
    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { b, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
            memory_read_from_state(state, RV32_REGISTER_AS, b.as_canonical_u32());

        rs1
    }

    #[inline(always)]
    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction {
            a, d, f: enabled, ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        if *enabled != F::ZERO {
            memory_write_from_state(state, RV32_REGISTER_AS, a.as_canonical_u32(), data);
        }
    }
}
