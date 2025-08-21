use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{elem_to_ext, FriReducedOpeningExecutor};
use crate::field_extension::{FieldExtension, EXT_DEG};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FriReducedOpeningPreCompute {
    a_ptr_ptr: u32,
    b_ptr_ptr: u32,
    length_ptr: u32,
    alpha_ptr: u32,
    result_ptr: u32,
    hint_id_ptr: u32,
    is_init_ptr: u32,
}

impl FriReducedOpeningExecutor {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut FriReducedOpeningPreCompute,
    ) -> Result<(), StaticProgramError> {
        let &Instruction {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;

        let a_ptr_ptr = a.as_canonical_u32();
        let b_ptr_ptr = b.as_canonical_u32();
        let length_ptr = c.as_canonical_u32();
        let alpha_ptr = d.as_canonical_u32();
        let result_ptr = e.as_canonical_u32();
        let hint_id_ptr = f.as_canonical_u32();
        let is_init_ptr = g.as_canonical_u32();

        *data = FriReducedOpeningPreCompute {
            a_ptr_ptr,
            b_ptr_ptr,
            length_ptr,
            alpha_ptr,
            result_ptr,
            hint_id_ptr,
            is_init_ptr,
        };

        Ok(())
    }
}

impl<F> Executor<F> for FriReducedOpeningExecutor
where
    F: PrimeField32,
{
    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut FriReducedOpeningPreCompute = data.borrow_mut();

        self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = execute_e1_tco_handler;
        Ok(fn_ptr)
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<FriReducedOpeningPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut FriReducedOpeningPreCompute = data.borrow_mut();

        self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = execute_e1_impl;
        Ok(fn_ptr)
    }
}

impl<F> MeteredExecutor<F> for FriReducedOpeningExecutor
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<FriReducedOpeningPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<FriReducedOpeningPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = execute_e2_impl;
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
        let pre_compute: &mut E2PreCompute<FriReducedOpeningPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = execute_e2_tco_handler;
        Ok(fn_ptr)
    }
}

#[create_tco_handler]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FriReducedOpeningPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FriReducedOpeningPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &FriReducedOpeningPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let alpha = vm_state.vm_read(AS::Native as u32, pre_compute.alpha_ptr);

    let [length]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.length_ptr);
    let length = length.as_canonical_u32() as usize;

    let [a_ptr]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.a_ptr_ptr);
    let [b_ptr]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.b_ptr_ptr);

    let [is_init_read]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.is_init_ptr);
    let is_init = is_init_read.as_canonical_u32();

    let [hint_id_f]: [F; 1] = vm_state.host_read(AS::Native as u32, pre_compute.hint_id_ptr);
    let hint_id = hint_id_f.as_canonical_u32() as usize;

    let data = if is_init == 0 {
        let hint_steam = &mut vm_state.streams.hint_space[hint_id];
        hint_steam.drain(0..length).collect()
    } else {
        vec![]
    };

    let mut as_and_bs = Vec::with_capacity(length);
    #[allow(clippy::needless_range_loop)]
    for i in 0..length {
        let a_ptr_i = (a_ptr + F::from_canonical_usize(i)).as_canonical_u32();
        let [a]: [F; 1] = if is_init == 0 {
            vm_state.vm_write(AS::Native as u32, a_ptr_i, &[data[i]]);
            [data[i]]
        } else {
            vm_state.vm_read(AS::Native as u32, a_ptr_i)
        };
        let b_ptr_i = (b_ptr + F::from_canonical_usize(EXT_DEG * i)).as_canonical_u32();
        let b = vm_state.vm_read(AS::Native as u32, b_ptr_i);

        as_and_bs.push((a, b));
    }

    let mut result = [F::ZERO; EXT_DEG];
    for (a, b) in as_and_bs.into_iter().rev() {
        // result = result * alpha + (b - a)
        result = FieldExtension::add(
            FieldExtension::multiply(result, alpha),
            FieldExtension::subtract(b, elem_to_ext(a)),
        );
    }

    vm_state.vm_write(AS::Native as u32, pre_compute.result_ptr, &result);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    length as u32 + 2
}
