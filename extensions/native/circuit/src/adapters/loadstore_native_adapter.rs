use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, AdapterAirContext, AdapterExecutorE1, AdapterTraceStep,
        ExecutionBridge, ExecutionState, VmAdapterAir, VmAdapterInterface, VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS,
    NativeLoadStoreOpcode::{self, *},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

use crate::adapters::{
    memory_read_native, memory_read_native_from_state, memory_write_native_from_state,
    tracing_read_native, tracing_write_native,
};

pub struct NativeLoadStoreInstruction<T> {
    pub is_valid: T,
    // Absolute opcode number
    pub opcode: T,
    pub is_loadw: T,
    pub is_storew: T,
    pub is_hint_storew: T,
}

pub struct NativeLoadStoreAdapterInterface<T, const NUM_CELLS: usize>(PhantomData<T>);

impl<T, const NUM_CELLS: usize> VmAdapterInterface<T>
    for NativeLoadStoreAdapterInterface<T, NUM_CELLS>
{
    type Reads = (T, [T; NUM_CELLS]);
    type Writes = [T; NUM_CELLS];
    type ProcessedInstruction = NativeLoadStoreInstruction<T>;
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct NativeLoadStoreAdapterCols<T, const NUM_CELLS: usize> {
    pub from_state: ExecutionState<T>,
    pub a: T,
    pub b: T,
    pub c: T,

    pub data_write_pointer: T,

    pub pointer_read_aux_cols: MemoryReadAuxCols<T>,
    pub data_read_aux_cols: MemoryReadAuxCols<T>,
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
        let cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = AB::Expr::from_canonical_usize(0);

        let is_valid = ctx.instruction.is_valid;
        let is_loadw = ctx.instruction.is_loadw;
        let is_storew = ctx.instruction.is_storew;
        let is_hint_storew = ctx.instruction.is_hint_storew;

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        let ptr = ctx.reads.0;
        // Here we ignore ctx.reads.1 and we use `ctx.writes` as the data for both the write and the
        // second read (in the case of load/store when it exists).
        let data = ctx.writes;

        // first pointer read is always [c]_d
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.c),
                [ptr.clone()],
                timestamp + timestamp_delta.clone(),
                &cols.pointer_read_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    native_as.clone(),
                    is_storew.clone() * cols.a + is_loadw.clone() * (ptr.clone() + cols.b),
                ),
                data.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_read_aux_cols,
            )
            .eval(builder, is_valid.clone() - is_hint_storew.clone());
        timestamp_delta += is_valid.clone() - is_hint_storew.clone();

        builder.assert_eq(
            is_valid.clone() * cols.data_write_pointer,
            is_loadw.clone() * cols.a
                + (is_storew.clone() + is_hint_storew.clone()) * (ptr.clone() + cols.b),
        );

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.data_write_pointer),
                data.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_write_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a.into(),
                    cols.b.into(),
                    cols.c.into(),
                    native_as.clone(),
                    native_as.clone(),
                ],
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

#[derive(derive_new::new)]
pub struct NativeLoadStoreAdapterStep<const NUM_CELLS: usize> {
    offset: usize,
}

impl<F, CTX, const NUM_CELLS: usize> AdapterTraceStep<F, CTX>
    for NativeLoadStoreAdapterStep<NUM_CELLS>
where
    F: PrimeField32,
{
    const WIDTH: usize = std::mem::size_of::<NativeLoadStoreAdapterCols<u8, NUM_CELLS>>();
    type ReadData = (F, [F; NUM_CELLS]);
    type WriteData = [F; NUM_CELLS];
    type TraceContext<'a> = F;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut NativeLoadStoreAdapterCols<F, NUM_CELLS> = adapter_row.borrow_mut();

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
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);

        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let adapter_row: &mut NativeLoadStoreAdapterCols<F, NUM_CELLS> = adapter_row.borrow_mut();
        adapter_row.a = a;
        adapter_row.b = b;
        adapter_row.c = c;

        // Read the pointer value from memory
        let [read_cell] = tracing_read_native::<F, 1>(
            memory,
            c.as_canonical_u32(),
            adapter_row.pointer_read_aux_cols.as_mut(),
        );

        let (data_read_as, _) = match local_opcode {
            LOADW => (e.as_canonical_u32(), d.as_canonical_u32()),
            STOREW | HINT_STOREW => (d.as_canonical_u32(), e.as_canonical_u32()),
        };

        debug_assert_eq!(data_read_as, AS::Native as u32);

        let (data_read_ptr, _) = match local_opcode {
            LOADW => ((read_cell + b).as_canonical_u32(), a.as_canonical_u32()),
            STOREW | HINT_STOREW => (a.as_canonical_u32(), (read_cell + b).as_canonical_u32()),
        };

        // Read data based on opcode
        let data_read: [F; NUM_CELLS] = match local_opcode {
            HINT_STOREW => [F::ZERO; NUM_CELLS],
            LOADW | STOREW => tracing_read_native::<F, NUM_CELLS>(
                memory,
                data_read_ptr,
                adapter_row.data_read_aux_cols.as_mut(),
            ),
        };

        (read_cell, data_read)
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
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        // TODO(ayush): remove duplication
        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);

        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let adapter_row: &mut NativeLoadStoreAdapterCols<F, NUM_CELLS> = adapter_row.borrow_mut();

        let [read_cell] = memory_read_native::<F, 1>(memory.data(), c.as_canonical_u32());

        let (_, data_write_as) = match local_opcode {
            LOADW => (e.as_canonical_u32(), d.as_canonical_u32()),
            STOREW | HINT_STOREW => (d.as_canonical_u32(), e.as_canonical_u32()),
        };

        debug_assert_eq!(data_write_as, AS::Native as u32);

        let data_write_ptr = match local_opcode {
            LOADW => a.as_canonical_u32(),
            STOREW | HINT_STOREW => (read_cell + b).as_canonical_u32(),
        };

        adapter_row.data_write_pointer = F::from_canonical_u32(data_write_ptr);

        // Write data to memory
        tracing_write_native(
            memory,
            data_write_ptr,
            data,
            &mut adapter_row.data_write_aux_cols,
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        is_hint_storew: Self::TraceContext<'_>,
        adapter_row: &mut [F],
    ) {
        let adapter_row: &mut NativeLoadStoreAdapterCols<F, NUM_CELLS> = adapter_row.borrow_mut();
        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();

        // Fill auxiliary columns for memory operations
        mem_helper.fill_from_prev(timestamp, adapter_row.pointer_read_aux_cols.as_mut());
        timestamp += 1;

        if is_hint_storew.is_zero() {
            mem_helper.fill_from_prev(timestamp, adapter_row.data_read_aux_cols.as_mut());
            timestamp += 1;
        }

        mem_helper.fill_from_prev(timestamp, adapter_row.data_write_aux_cols.as_mut());
    }
}

impl<F, const NUM_CELLS: usize> AdapterExecutorE1<F> for NativeLoadStoreAdapterStep<NUM_CELLS>
where
    F: PrimeField32,
{
    type ReadData = (F, [F; NUM_CELLS]);
    type WriteData = [F; NUM_CELLS];

    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);

        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [read_cell]: [F; 1] = memory_read_native_from_state(state, c.as_canonical_u32());

        let data_read_as = match local_opcode {
            LOADW => e.as_canonical_u32(),
            STOREW | HINT_STOREW => d.as_canonical_u32(),
        };

        debug_assert_eq!(data_read_as, AS::Native as u32);

        let data_read_ptr = match local_opcode {
            LOADW => (read_cell + b).as_canonical_u32(),
            STOREW | HINT_STOREW => a.as_canonical_u32(),
        };

        let data_read: [F; NUM_CELLS] = match local_opcode {
            HINT_STOREW => [F::ZERO; NUM_CELLS],
            LOADW | STOREW => memory_read_native_from_state(state, data_read_ptr),
        };

        (read_cell, data_read)
    }

    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), AS::Native as u32);

        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [read_cell]: [F; 1] = memory_read_native(state.memory, c.as_canonical_u32());

        let data_write_as = match local_opcode {
            LOADW => d.as_canonical_u32(),
            STOREW | HINT_STOREW => e.as_canonical_u32(),
        };

        debug_assert_eq!(data_write_as, AS::Native as u32);

        let data_write_ptr = match local_opcode {
            LOADW => a.as_canonical_u32(),
            STOREW | HINT_STOREW => (read_cell + b).as_canonical_u32(),
        };

        memory_write_native_from_state(state, data_write_ptr, data);
    }
}
