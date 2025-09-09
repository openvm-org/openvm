use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, NativeLoadStoreOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::NativeLoadStoreCoreExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct NativeLoadStorePreCompute<F> {
    a: u32,
    b: F,
    c: u32,
}

impl<A, const NUM_CELLS: usize> NativeLoadStoreCoreExecutor<A, NUM_CELLS> {
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

impl<F, A, const NUM_CELLS: usize> Executor<F> for NativeLoadStoreCoreExecutor<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<NativeLoadStorePreCompute<F>>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut NativeLoadStorePreCompute<F> = data.borrow_mut();

        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = match local_opcode {
            NativeLoadStoreOpcode::LOADW => execute_e1_loadw_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::STOREW => execute_e1_storew_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::HINT_STOREW => {
                execute_e1_hint_storew_handler::<F, Ctx, NUM_CELLS>
            }
        };

        Ok(fn_ptr)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut NativeLoadStorePreCompute<F> = data.borrow_mut();

        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = match local_opcode {
            NativeLoadStoreOpcode::LOADW => execute_e1_loadw_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::STOREW => execute_e1_storew_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::HINT_STOREW => {
                execute_e1_hint_storew_handler::<F, Ctx, NUM_CELLS>
            }
        };

        Ok(fn_ptr)
    }
}

impl<F, A, const NUM_CELLS: usize> MeteredExecutor<F> for NativeLoadStoreCoreExecutor<A, NUM_CELLS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<NativeLoadStorePreCompute<F>>>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
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
            NativeLoadStoreOpcode::LOADW => execute_e2_loadw_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::STOREW => execute_e2_storew_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::HINT_STOREW => {
                execute_e2_hint_storew_handler::<F, Ctx, NUM_CELLS>
            }
        };

        Ok(fn_ptr)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<NativeLoadStorePreCompute<F>> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = match local_opcode {
            NativeLoadStoreOpcode::LOADW => execute_e2_loadw_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::STOREW => execute_e2_storew_handler::<F, Ctx, NUM_CELLS>,
            NativeLoadStoreOpcode::HINT_STOREW => {
                execute_e2_hint_storew_handler::<F, Ctx, NUM_CELLS>
            }
        };

        Ok(fn_ptr)
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_loadw<F: PrimeField32, CTX: ExecutionCtxTrait, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeLoadStorePreCompute<F> = pre_compute.borrow();
    execute_e12_loadw::<_, _, NUM_CELLS>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_storew<F: PrimeField32, CTX: ExecutionCtxTrait, const NUM_CELLS: usize>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeLoadStorePreCompute<F> = pre_compute.borrow();
    execute_e12_storew::<_, _, NUM_CELLS>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_hint_storew<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const NUM_CELLS: usize,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &NativeLoadStorePreCompute<F> = pre_compute.borrow();
    execute_e12_hint_storew::<_, _, NUM_CELLS>(pre_compute, instret, pc, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_loadw<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const NUM_CELLS: usize,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeLoadStorePreCompute<F>> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_loadw::<_, _, NUM_CELLS>(&pre_compute.data, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_storew<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const NUM_CELLS: usize,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeLoadStorePreCompute<F>> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_storew::<_, _, NUM_CELLS>(&pre_compute.data, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_hint_storew<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const NUM_CELLS: usize,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<NativeLoadStorePreCompute<F>> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_hint_storew::<_, _, NUM_CELLS>(&pre_compute.data, instret, pc, exec_state)
}

#[inline(always)]
unsafe fn execute_e12_loadw<F: PrimeField32, CTX: ExecutionCtxTrait, const NUM_CELLS: usize>(
    pre_compute: &NativeLoadStorePreCompute<F>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let [read_cell]: [F; 1] = exec_state.vm_read(AS::Native as u32, pre_compute.c);

    let data_read_ptr = (read_cell + pre_compute.b).as_canonical_u32();
    let data_read: [F; NUM_CELLS] = exec_state.vm_read(AS::Native as u32, data_read_ptr);

    exec_state.vm_write(AS::Native as u32, pre_compute.a, &data_read);

    *pc = (*pc).wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[inline(always)]
unsafe fn execute_e12_storew<F: PrimeField32, CTX: ExecutionCtxTrait, const NUM_CELLS: usize>(
    pre_compute: &NativeLoadStorePreCompute<F>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let [read_cell]: [F; 1] = exec_state.vm_read(AS::Native as u32, pre_compute.c);
    let data_read: [F; NUM_CELLS] = exec_state.vm_read(AS::Native as u32, pre_compute.a);

    let data_write_ptr = (read_cell + pre_compute.b).as_canonical_u32();
    exec_state.vm_write(AS::Native as u32, data_write_ptr, &data_read);

    *pc = (*pc).wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[inline(always)]
unsafe fn execute_e12_hint_storew<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const NUM_CELLS: usize,
>(
    pre_compute: &NativeLoadStorePreCompute<F>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let [read_cell]: [F; 1] = exec_state.vm_read(AS::Native as u32, pre_compute.c);

    if exec_state.streams.hint_stream.len() < NUM_CELLS {
        let err = ExecutionError::HintOutOfBounds { pc: *pc };
        return Err(err);
    }
    let data: [F; NUM_CELLS] =
        array::from_fn(|_| exec_state.streams.hint_stream.pop_front().unwrap());

    let data_write_ptr = (read_cell + pre_compute.b).as_canonical_u32();
    exec_state.vm_write(AS::Native as u32, data_write_ptr, &data);

    *pc = (*pc).wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    Ok(())
}
