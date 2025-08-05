use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, NativeLoadStoreOpcode};
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

#[derive(derive_new::new, Debug, Clone, Copy)]
pub struct NativeLoadStoreCoreStep<A, const NUM_CELLS: usize> {
    adapter: A,
    offset: usize,
}

#[derive(derive_new::new)]
pub struct NativeLoadStoreCoreFiller<A, const NUM_CELLS: usize> {
    adapter: A,
}

impl<F, A, RA, const NUM_CELLS: usize> PreflightExecutor<F, RA>
    for NativeLoadStoreCoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceStep<F, ReadData = (F, [F; NUM_CELLS]), WriteData = [F; NUM_CELLS]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut NativeLoadStoreCoreRecord<F, NUM_CELLS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeLoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

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
            .write(state.memory, instruction, data, &mut adapter_record);

        core_record.pointer_read = pointer_read;
        core_record.data = data;

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_CELLS: usize> TraceFiller<F> for NativeLoadStoreCoreFiller<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
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

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct NativeLoadStorePreCompute<F> {
    a: u32,
    b: F,
    c: u32,
}

impl<A, const NUM_CELLS: usize> NativeLoadStoreCoreStep<A, NUM_CELLS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut NativeLoadStorePreCompute<F>,
    ) -> Result<NativeLoadStoreOpcode, StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let a = a.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();

        if d != AS::Native as u32 || e != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = NativeLoadStorePreCompute { a, b, c };

        Ok(local_opcode)
    }
}

impl<F, A, const NUM_CELLS: usize> Executor<F> for NativeLoadStoreCoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<NativeLoadStorePreCompute<F>>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut NativeLoadStorePreCompute<F> = data.borrow_mut();

        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = match local_opcode {
            NativeLoadStoreOpcode::LOADW => execute_e1_loadw::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::STOREW => execute_e1_storew::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::HINT_STOREW => execute_e1_hint_storew::<F, Ctx, NUM_CELLS>,
        };

        Ok(fn_ptr)
    }
}

impl<F, A, const NUM_CELLS: usize> MeteredExecutor<F> for NativeLoadStoreCoreStep<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<NativeLoadStorePreCompute<F>>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: E2ExecutionCtx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<NativeLoadStorePreCompute<F>> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = match local_opcode {
            NativeLoadStoreOpcode::LOADW => execute_e2_loadw::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::STOREW => execute_e2_storew::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::HINT_STOREW => execute_e2_hint_storew::<F, Ctx, NUM_CELLS>,
        };

        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_loadw<F: PrimeField32, CTX: E1ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeLoadStorePreCompute<F> = pre_compute.borrow();
    execute_e12_loadw::<_, _, NUM_CELLS>(pre_compute, vm_state);
}

unsafe fn execute_e1_storew<F: PrimeField32, CTX: E1ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeLoadStorePreCompute<F> = pre_compute.borrow();
    execute_e12_storew::<_, _, NUM_CELLS>(pre_compute, vm_state);
}

unsafe fn execute_e1_hint_storew<F: PrimeField32, CTX: E1ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeLoadStorePreCompute<F> = pre_compute.borrow();
    execute_e12_hint_storew::<_, _, NUM_CELLS>(pre_compute, vm_state);
}

unsafe fn execute_e2_loadw<F: PrimeField32, CTX: E2ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeLoadStorePreCompute<F>> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_loadw::<_, _, NUM_CELLS>(&pre_compute.data, vm_state);
}

unsafe fn execute_e2_storew<F: PrimeField32, CTX: E2ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeLoadStorePreCompute<F>> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_storew::<_, _, NUM_CELLS>(&pre_compute.data, vm_state);
}

unsafe fn execute_e2_hint_storew<F: PrimeField32, CTX: E2ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeLoadStorePreCompute<F>> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_hint_storew::<_, _, NUM_CELLS>(&pre_compute.data, vm_state);
}

#[inline(always)]
unsafe fn execute_e12_loadw<F: PrimeField32, CTX: E1ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &NativeLoadStorePreCompute<F>,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let [read_cell]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.c);

    let data_read_ptr = (read_cell + pre_compute.b).as_canonical_u32();
    let data_read: [F; NUM_CELLS] = vm_state.vm_read(AS::Native as u32, data_read_ptr);

    vm_state.vm_write(AS::Native as u32, pre_compute.a, &data_read);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[inline(always)]
unsafe fn execute_e12_storew<F: PrimeField32, CTX: E1ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &NativeLoadStorePreCompute<F>,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let [read_cell]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.c);
    let data_read: [F; NUM_CELLS] = vm_state.vm_read(AS::Native as u32, pre_compute.a);

    let data_write_ptr = (read_cell + pre_compute.b).as_canonical_u32();
    vm_state.vm_write(AS::Native as u32, data_write_ptr, &data_read);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[inline(always)]
unsafe fn execute_e12_hint_storew<F: PrimeField32, CTX: E1ExecutionCtx, const NUM_CELLS: usize>(
    pre_compute: &NativeLoadStorePreCompute<F>,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let [read_cell]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.c);

    if vm_state.streams.hint_stream.len() < NUM_CELLS {
        vm_state.exit_code = Err(ExecutionError::HintOutOfBounds { pc: vm_state.pc });
        return;
    }
    let data: [F; NUM_CELLS] =
        array::from_fn(|_| vm_state.streams.hint_stream.pop_front().unwrap());

    let data_write_ptr = (read_cell + pre_compute.b).as_canonical_u32();
    vm_state.vm_write(AS::Native as u32, data_write_ptr, &data);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}
