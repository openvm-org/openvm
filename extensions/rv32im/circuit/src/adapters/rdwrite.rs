use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{utils::not, AlignedBytesBorrow};
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
use crate::adapters::tracing_write;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32RdWriteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32CondRdWriteAdapterCols<T> {
    pub inner: Rv32RdWriteAdapterCols<T>,
    pub needs_write: T,
}

/// This adapter doesn't read anything, and writes to \[a:4\]_d, where d == 1
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32RdWriteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

/// This adapter doesn't read anything, and **maybe** writes to \[a:4\]_d, where d == 1
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32CondRdWriteAdapterAir {
    inner: Rv32RdWriteAdapterAir,
}

impl<F: Field> BaseAir<F> for Rv32RdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv32RdWriteAdapterCols::<F>::width()
    }
}

impl<F: Field> BaseAir<F> for Rv32CondRdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv32CondRdWriteAdapterCols::<F>::width()
    }
}

impl Rv32RdWriteAdapterAir {
    /// If `needs_write` is provided:
    /// - Only writes if `needs_write`.
    /// - Sets operand `f = needs_write` in the instruction.
    /// - Does not put any other constraints on `needs_write`
    ///
    /// Otherwise:
    /// - Writes if `ctx.instruction.is_valid`.
    /// - Sets operand `f` to default value of `0` in the instruction.
    #[allow(clippy::type_complexity)]
    fn conditional_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_cols: &Rv32RdWriteAdapterCols<AB::Var>,
        ctx: AdapterAirContext<
            AB::Expr,
            BasicAdapterInterface<
                AB::Expr,
                ImmInstruction<AB::Expr>,
                0,
                1,
                0,
                RV32_REGISTER_NUM_LIMBS,
            >,
        >,
        needs_write: Option<AB::Expr>,
    ) {
        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let timestamp_delta = 1;
        let (write_count, f) = if let Some(needs_write) = needs_write {
            (needs_write.clone(), needs_write)
        } else {
            (ctx.instruction.is_valid.clone(), AB::Expr::ZERO)
        };
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd_ptr,
                ),
                ctx.writes[0].clone(),
                timestamp,
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
                    AB::Expr::ZERO,
                    ctx.instruction.immediate,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::ZERO,
                    f,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32RdWriteAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, RV32_REGISTER_NUM_LIMBS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32RdWriteAdapterCols<AB::Var> = (*local).borrow();
        self.conditional_eval(builder, local_cols, ctx, None);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32RdWriteAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32CondRdWriteAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, RV32_REGISTER_NUM_LIMBS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32CondRdWriteAdapterCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local_cols.needs_write);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(local_cols.needs_write);

        self.inner.conditional_eval(
            builder,
            &local_cols.inner,
            ctx,
            Some(local_cols.needs_write.into()),
        );
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32CondRdWriteAdapterCols<_> = local.borrow();
        cols.inner.from_state.pc
    }
}

/// This adapter doesn't read anything, and writes to \[a:4\]_d, where d == 1
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Rv32RdWriteAdapterRecord {
    pub from_pc: u32,
    pub from_timestamp: u32,

    // Will use u32::MAX to indicate no write
    pub rd_ptr: u32,
    pub rd_aux_record: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32RdWriteAdapterExecutor;

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32RdWriteAdapterFiller;

impl<F> AdapterTraceExecutor<F> for Rv32RdWriteAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32RdWriteAdapterCols<u8>>();
    type ReadData = ();
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type RecordMut<'a> = &'a mut Rv32RdWriteAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        // Rv32RdWriteAdapter doesn't read anything
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();
        tracing_write(
            memory,
            RV32_REGISTER_AS,
            record.rd_ptr,
            data,
            &mut record.rd_aux_record.prev_timestamp,
            &mut record.rd_aux_record.prev_data,
        );
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32RdWriteAdapterFiller {
    const WIDTH: usize = size_of::<Rv32RdWriteAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as Rv32RdWriteAdapterRecord
        let record: &Rv32RdWriteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut Rv32RdWriteAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row
            .rd_aux_cols
            .set_prev_data(record.rd_aux_record.prev_data.map(F::from_canonical_u8));
        mem_helper.fill(
            record.rd_aux_record.prev_timestamp,
            record.from_timestamp,
            adapter_row.rd_aux_cols.as_mut(),
        );
        adapter_row.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        adapter_row.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}

/// This adapter doesn't read anything, and **maybe** writes to \[a:4\]_d, where d == 1
#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32CondRdWriteAdapterExecutor {
    inner: Rv32RdWriteAdapterExecutor,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32CondRdWriteAdapterFiller {
    inner: Rv32RdWriteAdapterFiller,
}

impl<F> AdapterTraceExecutor<F> for Rv32CondRdWriteAdapterExecutor
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<Rv32CondRdWriteAdapterCols<u8>>();
    type ReadData = ();
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type RecordMut<'a> = &'a mut Rv32RdWriteAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        <Rv32RdWriteAdapterExecutor as AdapterTraceExecutor<F>>::read(
            &self.inner,
            memory,
            instruction,
            record,
        )
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let Instruction { f: enabled, .. } = instruction;

        if enabled.is_one() {
            <Rv32RdWriteAdapterExecutor as AdapterTraceExecutor<F>>::write(
                &self.inner,
                memory,
                instruction,
                data,
                record,
            );
        } else {
            memory.increment_timestamp();
            record.rd_ptr = u32::MAX;
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for Rv32CondRdWriteAdapterFiller {
    const WIDTH: usize = size_of::<Rv32CondRdWriteAdapterCols<u8>>();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        // - get_record_from_slice correctly interprets the bytes as Rv32RdWriteAdapterRecord
        let record: &Rv32RdWriteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_cols: &mut Rv32CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();

        adapter_cols.needs_write = F::from_bool(record.rd_ptr != u32::MAX);

        if record.rd_ptr != u32::MAX {
            // SAFETY:
            // - adapter_row has sufficient length for the split
            // - size_of::<Rv32RdWriteAdapterCols<u8>>() is the correct split point
            unsafe {
                self.inner.fill_trace_row(
                    mem_helper,
                    adapter_row
                        .split_at_mut_unchecked(size_of::<Rv32RdWriteAdapterCols<u8>>())
                        .0,
                )
            };
        } else {
            adapter_cols.inner.rd_ptr = F::ZERO;
            mem_helper.fill_zero(adapter_cols.inner.rd_aux_cols.as_mut());
            adapter_cols.inner.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
            adapter_cols.inner.from_state.pc = F::from_canonical_u32(record.from_pc);
        }
    }
}
