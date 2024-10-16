use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, Instruction},
    },
};

#[derive(Debug)]
pub struct VectorReadRecord<F: Field, const NUM_READS: usize, const READ_SIZE: usize> {
    pub reads: [MemoryReadRecord<F, READ_SIZE>; NUM_READS],
}

#[derive(Debug)]
pub struct VectorWriteRecord<F: Field, const WRITE_SIZE: usize> {
    pub from_state: ExecutionState<u32>,
    pub writes: [MemoryWriteRecord<F, WRITE_SIZE>; 1],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NativeBasicAdapterChip<F: Field, const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub air: NativeBasicAdapterAir<READ_SIZE, WRITE_SIZE>,
    aux_cols_factory: MemoryAuxColsFactory<F>,
}

impl<F: PrimeField32, const READ_SIZE: usize, const WRITE_SIZE: usize>
    NativeBasicAdapterChip<F, READ_SIZE, WRITE_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        let aux_cols_factory = memory_controller.aux_cols_factory();
        Self {
            air: NativeBasicAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            aux_cols_factory,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeBasicAdapterCols<T, const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub from_state: ExecutionState<T>,
    pub a_idx: T,
    pub b_idx: T,
    pub c_idx: T,
    pub a_as: T,
    pub b_as: T,
    pub c_as: T,
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; 1],
    pub reads_aux: [MemoryReadAuxCols<T, READ_SIZE>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeBasicAdapterAir<const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const READ_SIZE: usize, const WRITE_SIZE: usize> BaseAir<F>
    for NativeBasicAdapterAir<READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        NativeBasicAdapterCols::<F, READ_SIZE, WRITE_SIZE>::width()
    }
}

impl<AB: InteractionBuilder, const READ_SIZE: usize, const WRITE_SIZE: usize> VmAdapterAir<AB>
    for NativeBasicAdapterAir<READ_SIZE, WRITE_SIZE>
{
    type Interface = BasicAdapterInterface<AB::Expr, 2, 1, READ_SIZE, WRITE_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeBasicAdapterCols<_, READ_SIZE, WRITE_SIZE> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(cols.b_as, cols.b_idx),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(cols.c_as, cols.c_idx),
                ctx.reads[1].clone(),
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

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeBasicAdapterCols<_, READ_SIZE, WRITE_SIZE> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32, const READ_SIZE: usize, const WRITE_SIZE: usize> VmAdapterChip<F>
    for NativeBasicAdapterChip<F, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = VectorReadRecord<F, 2, READ_SIZE>;
    type WriteRecord = VectorWriteRecord<F, WRITE_SIZE>;
    type Air = NativeBasicAdapterAir<READ_SIZE, WRITE_SIZE>;
    type Interface = BasicAdapterInterface<F, 2, 1, READ_SIZE, WRITE_SIZE>;

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

        let y_val = memory.read::<READ_SIZE>(e, b);
        let z_val = memory.read::<READ_SIZE>(f, c);

        Ok((
            [y_val.data, z_val.data],
            Self::ReadRecord {
                reads: [y_val, z_val],
            },
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
        let a_val = memory.write::<WRITE_SIZE>(d, a, output.writes[0]);

        Ok((
            ExecutionState {
                pc: from_state.pc + 1,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state,
                writes: [a_val],
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    ) {
        let row_slice: &mut NativeBasicAdapterCols<_, READ_SIZE, WRITE_SIZE> =
            row_slice.borrow_mut();
        let aux_cols_factory = &self.aux_cols_factory;

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.a_idx = write_record.writes[0].pointer;
        row_slice.a_as = write_record.writes[0].address_space;
        row_slice.b_idx = read_record.reads[0].pointer;
        row_slice.b_as = read_record.reads[0].address_space;
        row_slice.c_idx = read_record.reads[1].pointer;
        row_slice.c_as = read_record.reads[1].address_space;

        row_slice.reads_aux = [
            aux_cols_factory.make_read_aux_cols(read_record.reads[0]),
            aux_cols_factory.make_read_aux_cols(read_record.reads[1]),
        ];
        row_slice.writes_aux = [aux_cols_factory.make_write_aux_cols(write_record.writes[0])];
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
