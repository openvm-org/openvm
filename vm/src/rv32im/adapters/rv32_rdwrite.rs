use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
};

use afs_derive::AlignedBorrow;
use afs_primitives::utils::not;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JumpUiProcessedInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub immediate: T,
}

/// This adapter doesn't read anything, and writes to [a:4]_d, where d == 1
#[derive(Debug, Clone)]
pub struct Rv32RdWriteAdapterChip<F: Field> {
    pub air: Rv32RdWriteAdapterAir,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

/// This adapter doesn't read anything, and **maybe** writes to [a:4]_d, where d == 1
#[derive(Debug, Clone)]
pub struct Rv32CondRdWriteAdapterChip<F: Field> {
    /// Do not use the inner air directly, use `air` instead.
    inner: Rv32RdWriteAdapterChip<F>,
    pub air: Rv32CondRdWriteAdapterAir,
}

impl<F: PrimeField32> Rv32RdWriteAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: Rv32RdWriteAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}

impl<F: PrimeField32> Rv32CondRdWriteAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let inner = Rv32RdWriteAdapterChip::new(execution_bus, program_bus, memory_controller);
        let air = Rv32CondRdWriteAdapterAir { inner: inner.air };
        Self { inner, air }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32RdWriteWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub rd: Option<MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>>,
}

type Rv32RdWriteAdapterInterface<T> =
    BasicAdapterInterface<T, JumpUiProcessedInstruction<T>, 0, 1, 0, RV32_REGISTER_NUM_LANES>;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32RdWriteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LANES>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32CondRdWriteAdapterCols<T> {
    inner: Rv32RdWriteAdapterCols<T>,
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32RdWriteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

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
    /// Otherwise:
    /// - Writes if `ctx.instruction.is_valid`.
    /// - Sets operand `f` to default value of `0` in the instruction.
    fn condition_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_cols: &Rv32RdWriteAdapterCols<AB::Var>,
        ctx: AdapterAirContext<AB::Expr, Rv32RdWriteAdapterInterface<AB::Expr>>,
        needs_write: Option<AB::Expr>,
    ) {
        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let timestamp_delta = 1;
        let (write_count, op_f) = if let Some(needs_write) = needs_write {
            (needs_write.clone(), needs_write)
        } else {
            (ctx.instruction.is_valid.clone(), AB::Expr::zero())
        };
        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), local_cols.rd_ptr),
                ctx.writes[0].clone(),
                timestamp,
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(4));
        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    AB::Expr::zero(),
                    ctx.instruction.immediate,
                    AB::Expr::one(),
                    AB::Expr::zero(),
                    op_f,
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
    type Interface = Rv32RdWriteAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv32RdWriteAdapterCols<AB::Var> = (*local).borrow();
        self.condition_eval(builder, local_cols, ctx, None);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32RdWriteAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32CondRdWriteAdapterAir {
    type Interface = Rv32RdWriteAdapterInterface<AB::Expr>;

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

        self.inner.condition_eval(
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

impl<F: PrimeField32> VmAdapterChip<F> for Rv32RdWriteAdapterChip<F> {
    type ReadRecord = ();
    type WriteRecord = Rv32RdWriteWriteRecord<F>;
    type Air = Rv32RdWriteAdapterAir;
    type Interface = Rv32RdWriteAdapterInterface<F>;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let d = instruction.d;
        debug_assert_eq!(d.as_canonical_u32(), 1);

        Ok(([], ()))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction { op_a: a, d, .. } = *instruction;
        let rd = memory.write(d, a, output.writes[0]);

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state,
                rd: Some(rd),
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let adapter_cols: &mut Rv32RdWriteAdapterCols<F> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        let rd = write_record.rd.unwrap();
        adapter_cols.rd_ptr = rd.pointer;
        adapter_cols.rd_aux_cols = self.aux_cols_factory.make_write_aux_cols(rd);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32CondRdWriteAdapterChip<F> {
    type ReadRecord = ();
    type WriteRecord = Rv32RdWriteWriteRecord<F>;
    type Air = Rv32CondRdWriteAdapterAir;
    type Interface = Rv32RdWriteAdapterInterface<F>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        self.inner.preprocess(memory, instruction)
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction { op_a: a, d, .. } = *instruction;
        let rd = if instruction.op_f != F::zero() {
            Some(memory.write(d, a, output.writes[0]))
        } else {
            memory.increment_timestamp();
            None
        };

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state, rd },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let adapter_cols: &mut Rv32CondRdWriteAdapterCols<F> = row_slice.borrow_mut();
        adapter_cols.inner.from_state = write_record.from_state.map(F::from_canonical_u32);
        if let Some(rd) = write_record.rd {
            adapter_cols.inner.rd_ptr = rd.pointer;
            adapter_cols.inner.rd_aux_cols = self.inner.aux_cols_factory.make_write_aux_cols(rd);
            adapter_cols.needs_write = F::one();
        } else {
            adapter_cols.needs_write = F::zero();
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
