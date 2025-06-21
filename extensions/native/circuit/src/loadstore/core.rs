use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        get_record_from_slice,
        instructions::LocalOpcode,
        AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller, AdapterTraceStep,
        EmptyAdapterCoreLayout, ExecutionError, RecordArena, Result, StepExecutorE1, TraceFiller,
        TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::NativeLoadStoreOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::adapters::NativeLoadStoreInstruction;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeLoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub is_loadw: T,
    pub is_storew: T,
    pub is_hint_storew: T,

    pub pointer_read: T,
    pub data: [T; NUM_CELLS],
}

#[derive(Clone, Debug, derive_new::new)]
pub struct NativeLoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for NativeLoadStoreCoreAir<NUM_CELLS> {
    fn width(&self) -> usize {
        NativeLoadStoreCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize> BaseAirWithPublicValues<F>
    for NativeLoadStoreCoreAir<NUM_CELLS>
{
}

impl<AB, I, const NUM_CELLS: usize> VmCoreAir<AB, I> for NativeLoadStoreCoreAir<NUM_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<(AB::Expr, [AB::Expr; NUM_CELLS])>,
    I::Writes: From<[AB::Expr; NUM_CELLS]>,
    I::ProcessedInstruction: From<NativeLoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &NativeLoadStoreCoreCols<_, NUM_CELLS> = (*local_core).borrow();
        let flags = [cols.is_loadw, cols.is_storew, cols.is_hint_storew];
        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(NativeLoadStoreOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into()
                        * AB::Expr::from_canonical_usize(local_opcode.local_usize())
                },
            ),
        );

        AdapterAirContext {
            to_pc: None,
            reads: (cols.pointer_read.into(), cols.data.map(Into::into)).into(),
            writes: cols.data.map(Into::into).into(),
            instruction: NativeLoadStoreInstruction {
                is_valid,
                opcode: expected_opcode,
                is_loadw: cols.is_loadw.into(),
                is_storew: cols.is_storew.into(),
                is_hint_storew: cols.is_hint_storew.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct NativeLoadStoreCoreRecord<F, const NUM_CELLS: usize> {
    pub pointer_read: F,
    pub data: [F; NUM_CELLS],
    pub local_opcode: u8,
}

#[derive(Debug)]
pub struct NativeLoadStoreCoreStep<A, const NUM_CELLS: usize> {
    adapter: A,
    offset: usize,
}

impl<A, const NUM_CELLS: usize> NativeLoadStoreCoreStep<A, NUM_CELLS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self { adapter, offset }
    }
}

impl<F, CTX, A, const NUM_CELLS: usize> TraceStep<F, CTX> for NativeLoadStoreCoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceStep<F, CTX, ReadData = (F, [F; NUM_CELLS]), WriteData = [F; NUM_CELLS]>,
{
    type RecordLayout = EmptyAdapterCoreLayout<F, A>;
    type RecordMut<'a> = (
        A::RecordMut<'a>,
        &'a mut NativeLoadStoreCoreRecord<F, NUM_CELLS>,
    );

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeLoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let &Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let (pointer_read, data_read) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;
        let opcode = NativeLoadStoreOpcode::from_usize(core_record.local_opcode as usize);

        let data = if opcode == NativeLoadStoreOpcode::HINT_STOREW {
            if state.streams.hint_stream.len() < NUM_CELLS {
                return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
            }
            array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap())
        } else {
            data_read
        };

        self.adapter
            .write(state.memory, instruction, &data, &mut adapter_record);

        core_record.pointer_read = pointer_read;
        core_record.data = data;

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, CTX, A, const NUM_CELLS: usize> TraceFiller<F, CTX>
    for NativeLoadStoreCoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F, CTX>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &NativeLoadStoreCoreRecord<F, NUM_CELLS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut NativeLoadStoreCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        let opcode = NativeLoadStoreOpcode::from_usize(record.local_opcode as usize);

        // Writing in reverse order to avoid overwriting the `record`
        core_row.data = record.data;
        core_row.pointer_read = record.pointer_read;
        core_row.is_hint_storew = F::from_bool(opcode == NativeLoadStoreOpcode::HINT_STOREW);
        core_row.is_storew = F::from_bool(opcode == NativeLoadStoreOpcode::STOREW);
        core_row.is_loadw = F::from_bool(opcode == NativeLoadStoreOpcode::LOADW);
    }
}

impl<F, A, const NUM_CELLS: usize> StepExecutorE1<F> for NativeLoadStoreCoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<F, ReadData = (F, [F; NUM_CELLS]), WriteData = [F; NUM_CELLS]>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { opcode, .. } = instruction;

        // Get the local opcode for this instruction
        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (_, data_read) = self.adapter.read(state, instruction);

        let data = if local_opcode == NativeLoadStoreOpcode::HINT_STOREW {
            if state.streams.hint_stream.len() < NUM_CELLS {
                return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
            }
            array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap())
        } else {
            data_read
        };

        self.adapter.write(state, instruction, &data);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}
