use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::RV32_REGISTER_NUM_LANES;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        HasFromPc, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryWriteAuxCols}, MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef, MemoryWriteRecord
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

// This adapter doesn't read anything, and writes to [a:4]_d, where d == 1
#[derive(Debug, Clone)]
pub struct Rv32RdWriteAdapter<F: Field> {
    pub air: Rv32RdWriteAdapterAir,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32> Rv32RdWriteAdapter<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controler: MemoryControllerRef<F>,
    ) -> Self {
        Self {
            air: Rv32RdWriteAdapterAir {
                memory_bridge: RefCell::borrow(&memory_controler).memory_bridge(),
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            },
            aux_cols_factory: RefCell::borrow(&memory_controler).aux_cols_factory(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rv32RdWriteWriteRecord<F: Field> {
    pub from_state: ExecutionState<usize>,
    pub rd: MemoryWriteRecord<F, RV32_REGISTER_NUM_LANES>,
}

#[derive(Debug, Clone)]
pub struct Rv32RdWriteProcessedInstruction<T> {
    pub is_valid: T,
    pub expected_opcode: T,
    pub c: T,
}

// This is used by the CoreAir to pass the necessary fields to AdapterAir
impl<T> From<(T, T, T)> for Rv32RdWriteProcessedInstruction<T> {
    fn from(tuple: (T, T, T)) -> Self {
        Rv32RdWriteProcessedInstruction {
            is_valid: tuple.0,
            expected_opcode: tuple.1,
            c: tuple.2,
        }
    }
}

pub struct Rv32RdWriteAdapterInterface<T>(PhantomData<T>);
impl<T> VmAdapterInterface<T> for Rv32RdWriteAdapterInterface<T> {
    type Reads = ();
    type Writes = [T; RV32_REGISTER_NUM_LANES];
    type ProcessedInstruction = Rv32RdWriteProcessedInstruction<T>;
}

impl<T: Clone> HasFromPc<T> for Rv32RdWriteAdapterInterface<T> {
    fn get_from_pc(local_adapter: &[T]) -> T {
        let adapter_cols: &Rv32RdWriteAdapterCols<T> = (*local_adapter).borrow();
        return adapter_cols.from_state.pc.clone()
    }
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv32RdWriteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub a: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LANES>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32RdWriteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for Rv32RdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv32RdWriteAdapterCols::<u8>::width()
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

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };
        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), local_cols.a),
                ctx.writes,
                timestamp_pp(),
                &local_cols.rd_aux_cols,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(4));
        self.execution_bridge
            .execute(
                ctx.instruction.expected_opcode,
                [
                    local_cols.a.into(),
                    AB::Expr::zero(),
                    ctx.instruction.c.into(),
                    AB::Expr::one(),
                    AB::Expr::zero(),
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: local_cols.from_state.timestamp
                        + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for Rv32RdWriteAdapter<F> {
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

        Ok(((), ()))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        let Instruction { op_a: a, d, .. } = *instruction;
        let rd = memory.write(d, a, output.writes);

        let to_pc = output
            .to_pc
            .unwrap_or(F::from_canonical_usize(from_state.pc + 4));
        Ok((
            ExecutionState {
                pc: to_pc.as_canonical_u32() as usize,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
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
        let adapter_cols: &mut Rv32RdWriteAdapterCols<F> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_usize);
        adapter_cols.a = write_record.rd.pointer;
        adapter_cols.rd_aux_cols = self
            .aux_cols_factory
            .make_write_aux_cols(write_record.rd.clone());
        println!("{:?}", adapter_cols);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
