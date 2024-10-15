use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::NativeAdapterInterface;
use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, ExecutionBridge, ExecutionBus, ExecutionState,
        Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NativeAdapterChip<F: Field> {
    pub air: NativeAdapterAir,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32> NativeAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: NativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}

#[derive(Debug)]
pub struct NativeReadRecord<F: Field> {
    pub b: MemoryReadRecord<F, 1>,
    pub c: MemoryReadRecord<F, 1>,
}

#[derive(Debug)]
pub struct NativeWriteRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub a: MemoryWriteRecord<F, 1>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub a_idx: T,
    pub a_as: T,
    pub b_idx: T,
    pub b_as: T,
    pub c_idx: T,
    pub c_as: T,
    pub reads_aux: [MemoryReadOrImmediateAuxCols<T>; 2],
    pub writes_aux: [MemoryWriteAuxCols<T, 1>; 1],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for NativeAdapterAir {
    fn width(&self) -> usize {
        NativeAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for NativeAdapterAir {
    type Interface = NativeAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.b_as, cols.b_idx),
                ctx.reads[0][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.c_as, cols.c_idx),
                ctx.reads[1][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(cols.a_as, cols.a_idx),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_pc(
                ctx.instruction.opcode,
                [
                    cols.a_idx, cols.b_idx, cols.c_idx, cols.a_as, cols.b_as, cols.c_as,
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, ctx.instruction.is_valid);
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for NativeAdapterChip<F> {
    type ReadRecord = NativeReadRecord<F>;
    type WriteRecord = NativeWriteRecord<F>;
    type Air = NativeAdapterAir;
    type Interface = NativeAdapterInterface<F>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_b: b,
            op_c: c,
            e,
            op_f: f,
            ..
        } = *instruction;

        let b_val = memory.read::<1>(e, b);
        let c_val = memory.read::<1>(f, c);

        Ok((
            [b_val.data, c_val.data],
            Self::ReadRecord { b: b_val, c: c_val },
        ))
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
        let a_val = memory.write(d, a, output.writes[0]);

        Ok((
            ExecutionState {
                pc: from_state.pc + 1,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state,
                a: a_val,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let row_slice: &mut NativeAdapterCols<_> = row_slice.borrow_mut();
        let aux_cols_factory = &self.aux_cols_factory;

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.a_idx = write_record.a.pointer;
        row_slice.a_as = write_record.a.address_space;
        row_slice.b_idx = read_record.b.pointer;
        row_slice.b_as = read_record.b.address_space;
        row_slice.c_idx = read_record.c.pointer;
        row_slice.c_as = read_record.c.address_space;

        row_slice.reads_aux = [
            aux_cols_factory.make_read_or_immediate_aux_cols(read_record.b),
            aux_cols_factory.make_read_or_immediate_aux_cols(read_record.c),
        ];
        row_slice.writes_aux = [aux_cols_factory.make_write_aux_cols(write_record.a)];
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
