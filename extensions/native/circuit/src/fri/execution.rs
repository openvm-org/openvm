use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::p3_field::{
    extension::BinomiallyExtendable, ExtensionField, FieldAlgebra, FieldExtensionAlgebra,
    PackedField, PackedValue, PrimeField32,
};

use super::{ExtPacked, FriReducedOpeningExecutor, EF};
use crate::{
    field_extension::EXT_DEG,
    utils::{transmute_array_to_ext, transmute_ext_to_array},
};

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
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
{
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
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
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
}

unsafe fn execute_e1_impl<
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    CTX: ExecutionCtxTrait,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FriReducedOpeningPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    CTX: MeteredExecutionCtxTrait,
>(
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
unsafe fn execute_e12_impl<
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    CTX: ExecutionCtxTrait,
>(
    pre_compute: &FriReducedOpeningPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let alpha = vm_state.vm_read(AS::Native as u32, pre_compute.alpha_ptr);
    let alpha = transmute_array_to_ext::<F, EF<F>, EXT_DEG>(&alpha);

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

    let mut as_and_bs: Vec<(F, EF<F>)> = Vec::with_capacity(length);
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
        let b = transmute_array_to_ext::<F, EF<F>, EXT_DEG>(&b);

        as_and_bs.push((a, b));
    }

    // Use vectorized polynomial evaluation
    let result = compute_polynomial_evaluation(&as_and_bs, alpha);

    let result = transmute_ext_to_array::<F, EF<F>, EXT_DEG>(&result);
    vm_state.vm_write(AS::Native as u32, pre_compute.result_ptr, &result);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    length as u32 + 2
}

#[inline(always)]
pub(super) fn compute_polynomial_evaluation<F>(as_and_bs: &[(F, EF<F>)], alpha: EF<F>) -> EF<F>
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    F::Packing: PackedField<Scalar = F>,
    EF<F>: ExtensionField<F>,
{
    match F::Packing::WIDTH {
        4 => packed_polynomial_evaluation::<F, 4>(as_and_bs, alpha),
        8 => packed_polynomial_evaluation::<F, 8>(as_and_bs, alpha),
        16 => packed_polynomial_evaluation::<F, 16>(as_and_bs, alpha),
        _ => scalar_polynomial_evaluation(as_and_bs, alpha),
    }
}

#[inline(always)]
fn scalar_polynomial_evaluation<F>(as_and_bs: &[(F, EF<F>)], alpha: EF<F>) -> EF<F>
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
{
    let mut result = EF::<F>::ZERO;
    for &(a, b) in as_and_bs.iter().rev() {
        result = result * alpha + (b - EF::<F>::from_base(a));
    }
    result
}

#[inline(always)]
fn packed_polynomial_evaluation<F, const WIDTH: usize>(
    as_and_bs: &[(F, EF<F>)],
    alpha: EF<F>,
) -> EF<F>
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    F::Packing: PackedField<Scalar = F>,
    EF<F>: ExtensionField<F>,
{
    // Precompute powers of alpha for vectorized operations: [1, α, α², ..., α^(WIDTH-1)]
    let mut alpha_powers = [EF::<F>::ONE; WIDTH];
    for i in 1..WIDTH {
        alpha_powers[i] = alpha_powers[i - 1] * alpha;
    }

    // Pack alpha powers into SIMD format for efficient computation
    let mut alpha_powers_packed = ExtPacked::<F>::from_base_fn(|coeff_idx| {
        F::Packing::from_fn(|lane| alpha_powers[lane].as_base_slice()[coeff_idx])
    });

    // Compute α^WIDTH for updating packed powers between batches
    let alpha_width = alpha_powers[WIDTH - 1] * alpha;
    let alpha_width_packed = ExtPacked::<F>::from_f(alpha_width);

    let mut result_packed = ExtPacked::<F>::from_f(EF::<F>::ZERO);

    // Process full batches of WIDTH elements each
    for batch in as_and_bs.chunks_exact(WIDTH) {
        // Extract and pack coefficients (b - a) for the current batch
        let coeffs_packed = ExtPacked::<F>::from_base_fn(|coeff_idx| {
            F::Packing::from_fn(|lane| {
                let (a, b) = batch[lane];
                if coeff_idx == 0 {
                    b.as_base_slice()[coeff_idx] - a
                } else {
                    b.as_base_slice()[coeff_idx]
                }
            })
        });

        result_packed += coeffs_packed * alpha_powers_packed;
        alpha_powers_packed *= alpha_width_packed;
    }

    // Handle remaining elements that don't fill a complete batch
    let remainder = as_and_bs.chunks_exact(WIDTH).remainder();
    if !remainder.is_empty() {
        let coeffs_packed = ExtPacked::<F>::from_base_fn(|coeff_idx| {
            F::Packing::from_fn(|lane| {
                remainder.get(lane).map_or(F::ZERO, |&(a, b)| {
                    if coeff_idx == 0 {
                        b.as_base_slice()[coeff_idx] - a
                    } else {
                        b.as_base_slice()[coeff_idx]
                    }
                })
            })
        });

        result_packed += coeffs_packed * alpha_powers_packed;
    }

    // Perform horizontal reduction using tree-based summation for optimal performance
    let mut working_packed = result_packed;
    let mut stride = WIDTH / 2;

    while stride > 0 {
        let shifted_packed = ExtPacked::<F>::from_base_fn(|coeff_idx| {
            F::Packing::from_fn(|lane| {
                let shifted_lane = (lane + stride) % WIDTH;
                working_packed.as_base_slice()[coeff_idx].as_slice()[shifted_lane]
            })
        });
        working_packed += shifted_packed;
        stride /= 2;
    }

    // Extract the accumulated result from the first lane
    EF::<F>::from_base_fn(|coeff_idx| working_packed.as_base_slice()[coeff_idx].as_slice()[0])
}
