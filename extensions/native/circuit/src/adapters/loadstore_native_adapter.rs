use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::utils;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{AbstractField, Field, PrimeField32},
};
use axvm_circuit::{
    arch::{
        instructions::UsizeOpcode, AdapterAirContext, AdapterRuntimeContext, ExecutionBridge,
        ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
            MemoryReadRecord, MemoryWriteRecord,
        },
        program::ProgramBus,
    },
};
use axvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use axvm_native_compiler::NativeLoadStoreOpcode::{self, *};

pub struct NativeLoadStoreInstruction<T> {
    pub is_valid: T,
    // Absolute opcode number
    pub opcode: T,
    pub is_loadw: T,
    pub is_loadw2: T,
    pub is_storew: T,
    pub is_storew2: T,
    pub is_shintw: T,
}

pub struct NativeLoadStoreAdapterInterface<T, const NUM_CELLS: usize>(PhantomData<T>);

impl<T, const NUM_CELLS: usize> VmAdapterInterface<T>
    for NativeLoadStoreAdapterInterface<T, NUM_CELLS>
{
    // TODO[yi]: Fix when vectorizing
    type Reads = ([T; 2], T);
    type Writes = [T; NUM_CELLS];
    type ProcessedInstruction = NativeLoadStoreInstruction<T>;
}

#[derive(Debug)]
pub struct NativeLoadStoreAdapterChip<F: Field, const NUM_CELLS: usize> {
    pub air: NativeLoadStoreAdapterAir<NUM_CELLS>,
    offset: usize,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const NUM_CELLS: usize> NativeLoadStoreAdapterChip<F, NUM_CELLS> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        offset: usize,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: NativeLoadStoreAdapterAir {
                memory_bridge,
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            },
            offset,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeLoadStoreReadRecord<F: Field, const NUM_CELLS: usize> {
    pub pointer1_read: MemoryReadRecord<F, 1>,
    pub pointer2_read: Option<MemoryReadRecord<F, 1>>,
    pub data_read: Option<MemoryReadRecord<F, NUM_CELLS>>,
    pub write_as: F,
    pub write_ptr: F,

    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
    pub e: F,
    pub f: F,
    pub g: F,
}

#[derive(Clone, Debug)]
pub struct NativeLoadStoreWriteRecord<F: Field, const NUM_CELLS: usize> {
    pub from_state: ExecutionState<F>,
    pub write: MemoryWriteRecord<F, NUM_CELLS>,
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct NativeLoadStoreAdapterCols<T, const NUM_CELLS: usize> {
    pub from_state: ExecutionState<T>,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    pub g: T,

    pub data_read_as: T,
    pub data_read_pointer: T,

    pub data_write_as: T,
    pub data_write_pointer: T,

    pub pointer_read_aux_cols: [MemoryReadOrImmediateAuxCols<T>; 2],
    pub data_read_aux_cols: MemoryReadOrImmediateAuxCols<T>,
    // TODO[yi]: Fix when vectorizing
    // pub data_read_aux_cols: MemoryReadAuxCols<T, NUM_CELLS>,
    pub data_write_aux_cols: MemoryWriteAuxCols<T, NUM_CELLS>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeLoadStoreAdapterAir<const NUM_CELLS: usize> {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for NativeLoadStoreAdapterAir<NUM_CELLS> {
    fn width(&self) -> usize {
        NativeLoadStoreAdapterCols::<F, NUM_CELLS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_CELLS: usize> VmAdapterAir<AB>
    for NativeLoadStoreAdapterAir<NUM_CELLS>
{
    type Interface = NativeLoadStoreAdapterInterface<AB::Expr, NUM_CELLS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // TODO[yi]: Remove when vectorizing
        assert_eq!(NUM_CELLS, 1);

        let cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = AB::Expr::from_canonical_usize(0);

        let is_valid = ctx.instruction.is_valid;
        let is_loadw = ctx.instruction.is_loadw;
        let is_storew = ctx.instruction.is_storew;
        let is_loadw2 = ctx.instruction.is_loadw2;
        let is_storew2 = ctx.instruction.is_storew2;
        let is_shintw = ctx.instruction.is_shintw;

        // first pointer read is always [c]_d
        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.d, cols.c),
                ctx.reads.0[0].clone(),
                timestamp + timestamp_delta.clone(),
                &cols.pointer_read_aux_cols[0],
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        // second pointer read is [f]_d if loadw2 or storew2, otherwise disabled
        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.d, cols.f),
                ctx.reads.0[1].clone(),
                timestamp + timestamp_delta.clone(),
                &cols.pointer_read_aux_cols[1],
            )
            .eval(
                builder,
                is_valid.clone() - is_shintw.clone() - is_loadw.clone() - is_storew.clone(),
            );
        timestamp_delta +=
            is_valid.clone() - is_shintw.clone() - is_loadw.clone() - is_storew.clone();

        // TODO[yi]: Remove when vectorizing
        // read data, disabled if SHINTW
        // data pointer = [c]_d + [f]_d * g + b, degree 2
        builder
            .when(is_valid.clone() - is_shintw.clone())
            .assert_eq(
                cols.data_read_as,
                utils::select::<AB::Expr>(is_loadw.clone() + is_loadw2.clone(), cols.e, cols.d),
            );
        // TODO[yi]: Do we need to check for overflow?
        builder.assert_eq(
            (is_valid.clone() - is_shintw.clone()) * cols.data_read_pointer,
            (is_storew.clone() + is_storew2.clone()) * cols.a
                + (is_loadw.clone() + is_loadw2.clone())
                    * (ctx.reads.0[0].clone() + cols.b + ctx.reads.0[1].clone() * cols.g),
        );
        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.data_read_as, cols.data_read_pointer),
                ctx.reads.1.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_read_aux_cols,
            )
            .eval(builder, is_valid.clone() - is_shintw.clone());
        timestamp_delta += is_valid.clone() - is_shintw.clone();

        // data write
        builder.when(is_valid.clone()).assert_eq(
            cols.data_write_as,
            utils::select::<AB::Expr>(is_loadw.clone() + is_loadw2.clone(), cols.d, cols.e),
        );
        // TODO[yi]: Do we need to check for overflow?
        builder.assert_eq(
            is_valid.clone() * cols.data_write_pointer,
            (is_loadw.clone() + is_loadw2.clone()) * cols.a
                + (is_storew.clone() + is_storew2.clone() + is_shintw.clone())
                    * (ctx.reads.0[0].clone() + cols.b + ctx.reads.0[1].clone() * cols.g),
        );
        self.memory_bridge
            .write(
                MemoryAddress::new(cols.data_write_as, cols.data_write_pointer),
                ctx.writes.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_write_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [cols.a, cols.b, cols.c, cols.d, cols.e, cols.f, cols.g],
                cols.from_state,
                timestamp_delta.clone(),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let local_cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        local_cols.from_state.pc
    }
}

impl<F: PrimeField32, const NUM_CELLS: usize> VmAdapterChip<F>
    for NativeLoadStoreAdapterChip<F, NUM_CELLS>
{
    // TODO[yi]: Fix when vectorizing
    type ReadRecord = NativeLoadStoreReadRecord<F, 1>;
    type WriteRecord = NativeLoadStoreWriteRecord<F, NUM_CELLS>;
    type Air = NativeLoadStoreAdapterAir<NUM_CELLS>;
    type Interface = NativeLoadStoreAdapterInterface<F, NUM_CELLS>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = *instruction;
        let local_opcode_index = NativeLoadStoreOpcode::from_usize(opcode - self.offset);

        let read1_as = d;
        let read1_ptr = c;
        let read2_as = d;
        let read2_ptr = f;

        let read1_cell = memory.read_cell(read1_as, read1_ptr);
        let read2_cell = match local_opcode_index {
            LOADW2 | STOREW2 => Some(memory.read_cell(read2_as, read2_ptr)),
            _ => None,
        };

        let (data_read_as, data_write_as) = {
            match local_opcode_index {
                LOADW | LOADW2 => (e, d),
                STOREW | STOREW2 | SHINTW => (d, e),
            }
        };
        let (data_read_ptr, data_write_ptr) = {
            match local_opcode_index {
                LOADW => (read1_cell.data[0] + b, a),
                LOADW2 => (read1_cell.data[0] + b + read2_cell.unwrap().data[0] * g, a),
                STOREW => (a, read1_cell.data[0] + b),
                STOREW2 => (a, read1_cell.data[0] + b + read2_cell.unwrap().data[0] * g),
                SHINTW => (a, read1_cell.data[0] + b),
            }
        };

        // TODO[yi]: Fix when vectorizing
        let data_read = match local_opcode_index {
            SHINTW => None,
            _ => Some(memory.read::<1>(data_read_as, data_read_ptr)),
        };
        let record = NativeLoadStoreReadRecord {
            pointer1_read: read1_cell,
            pointer2_read: read2_cell,
            data_read,
            write_as: data_write_as,
            write_ptr: data_write_ptr,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        };

        Ok((
            (
                [
                    read1_cell.data[0],
                    read2_cell.map_or(F::ZERO, |x| x.data[0]),
                ],
                data_read.map_or(F::ZERO, |x| x.data[0]),
            ),
            record,
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let write =
            memory.write::<NUM_CELLS>(read_record.write_as, read_record.write_ptr, output.writes);
        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state: from_state.map(F::from_canonical_u32),
                write,
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
        let cols: &mut NativeLoadStoreAdapterCols<_, NUM_CELLS> = row_slice.borrow_mut();
        cols.from_state = write_record.from_state;
        cols.a = read_record.a;
        cols.b = read_record.b;
        cols.c = read_record.c;
        cols.d = read_record.d;
        cols.e = read_record.e;
        cols.f = read_record.f;
        cols.g = read_record.g;

        cols.data_read_as = read_record
            .data_read
            .map_or(F::ZERO, |read| read.address_space);
        cols.data_read_pointer = read_record.data_read.map_or(F::ZERO, |read| read.pointer);

        cols.data_write_as = write_record.write.address_space;
        cols.data_write_pointer = write_record.write.pointer;

        cols.pointer_read_aux_cols[0] =
            aux_cols_factory.make_read_or_immediate_aux_cols(read_record.pointer1_read);
        cols.pointer_read_aux_cols[1] = read_record
            .pointer2_read
            .map_or_else(MemoryReadOrImmediateAuxCols::disabled, |read| {
                aux_cols_factory.make_read_or_immediate_aux_cols(read)
            });
        cols.data_read_aux_cols = read_record
            .data_read
            .map_or_else(MemoryReadOrImmediateAuxCols::disabled, |read| {
                aux_cols_factory.make_read_or_immediate_aux_cols(read)
            });
        cols.data_write_aux_cols = aux_cols_factory.make_write_aux_cols(write_record.write);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
