use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use ax_circuit_derive::AlignedBorrow;
use ax_stark_backend::interaction::InteractionBuilder;
use axvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct NativeVectorizedAdapterChip<F: Field, const N: usize> {
    pub air: NativeVectorizedAdapterAir<N>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const N: usize> NativeVectorizedAdapterChip<F, N> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: NativeVectorizedAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct NativeVectorizedReadRecord<F: Field, const N: usize> {
    pub b: MemoryReadRecord<F, N>,
    pub c: MemoryReadRecord<F, N>,
}

#[derive(Debug)]
pub struct NativeVectorizedWriteRecord<F: Field, const N: usize> {
    pub from_state: ExecutionState<u32>,
    pub a: MemoryWriteRecord<F, N>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeVectorizedAdapterCols<T, const N: usize> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub ab_as: T,
    pub b_pointer: T,
    pub c_pointer: T,
    pub c_as: T,
    pub reads_aux: [MemoryReadAuxCols<T, N>; 2],
    pub writes_aux: [MemoryWriteAuxCols<T, N>; 1],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeVectorizedAdapterAir<const N: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const N: usize> BaseAir<F> for NativeVectorizedAdapterAir<N> {
    fn width(&self) -> usize {
        NativeVectorizedAdapterCols::<F, N>::width()
    }
}

impl<AB: InteractionBuilder, const N: usize> VmAdapterAir<AB> for NativeVectorizedAdapterAir<N> {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, N, N>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeVectorizedAdapterCols<_, N> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(cols.ab_as, cols.b_pointer),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(cols.c_as, cols.c_pointer),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &cols.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(cols.ab_as, cols.a_pointer),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a_pointer,
                    cols.b_pointer,
                    cols.c_pointer,
                    cols.ab_as,
                    cols.c_as,
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeVectorizedAdapterCols<_, N> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32, const N: usize> VmAdapterChip<F> for NativeVectorizedAdapterChip<F, N> {
    type ReadRecord = NativeVectorizedReadRecord<F, N>;
    type WriteRecord = NativeVectorizedWriteRecord<F, N>;
    type Air = NativeVectorizedAdapterAir<N>;
    type Interface = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, N, N>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, d, e, .. } = *instruction;

        let y_val = memory.read::<N>(d, b);
        let z_val = memory.read::<N>(e, c);

        Ok((
            [y_val.data, z_val.data],
            Self::ReadRecord { b: y_val, c: z_val },
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
        let Instruction { a, d, .. } = *instruction;
        let a_val = memory.write(d, a, output.writes[0]);

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
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
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let row_slice: &mut NativeVectorizedAdapterCols<_, N> = row_slice.borrow_mut();

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.a_pointer = write_record.a.pointer;
        row_slice.ab_as = write_record.a.address_space;
        row_slice.b_pointer = read_record.b.pointer;
        row_slice.c_pointer = read_record.c.pointer;
        row_slice.c_as = read_record.c.address_space;

        row_slice.reads_aux = [
            aux_cols_factory.make_read_aux_cols(read_record.b),
            aux_cols_factory.make_read_aux_cols(read_record.c),
        ];
        row_slice.writes_aux = [aux_cols_factory.make_write_aux_cols(write_record.a)];
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
