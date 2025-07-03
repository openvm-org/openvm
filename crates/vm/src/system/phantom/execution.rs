use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, PhantomDiscriminant, SysPhantom,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;

use crate::{
    arch::{
        execution_mode::E1ExecutionCtx, ExecuteFunc, ExecutionError, InsExecutorE1,
        PhantomSubExecutor, Streams, VmSegmentState,
    },
    system::{memory::online::GuestMemory, phantom::PhantomChip},
};

#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
pub(super) struct PhantomOperands {
    pub(super) a: u32,
    pub(super) b: u32,
    pub(super) c: u32,
}

#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
struct PhantomPreCompute<F> {
    operands: PhantomOperands,
    sub_executor: *const Box<dyn PhantomSubExecutor<F>>,
}

impl<F> InsExecutorE1<F> for PhantomChip<F>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<PhantomPreCompute<F>>()
    }
    #[inline(always)]
    fn pre_compute_e1<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> crate::arch::Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let data: &mut PhantomPreCompute<F> = data.borrow_mut();
        let c = inst.c.as_canonical_u32();
        *data = PhantomPreCompute {
            operands: PhantomOperands {
                a: inst.a.as_canonical_u32(),
                b: inst.b.as_canonical_u32(),
                c,
            },
            sub_executor: self
                .phantom_executors
                .get(&PhantomDiscriminant(c as u16))
                .unwrap(),
        };
        Ok(execute_e1_impl)
    }

    fn set_trace_height(&mut self, height: usize) {}
}

pub(super) struct PhantomStateMut<'a, F> {
    pub(super) pc: &'a mut u32,
    pub(super) memory: &'a mut GuestMemory,
    pub(super) streams: &'a mut Streams<F>,
    pub(super) rng: &'a mut StdRng,
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &PhantomPreCompute<F> = pre_compute.borrow();
    let sub_executor = &*pre_compute.sub_executor;
    execute_impl(
        PhantomStateMut {
            pc: &mut vm_state.pc,
            memory: &mut vm_state.memory,
            streams: &mut vm_state.streams,
            rng: &mut vm_state.rng,
        },
        &pre_compute.operands,
        sub_executor,
    )
    .unwrap();
    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}
#[inline(always)]
pub(super) fn execute_impl<F>(
    state: PhantomStateMut<F>,
    operands: &PhantomOperands,
    sub_executor: &Box<dyn PhantomSubExecutor<F>>,
) -> Result<(), ExecutionError>
where
    F: PrimeField32,
{
    let &PhantomOperands { a, b, c } = operands;

    let discriminant = PhantomDiscriminant(c as u16);
    // If not a system phantom sub-instruction (which is handled in
    // ExecutionSegment), look for a phantom sub-executor to handle it.
    if SysPhantom::from_repr(discriminant.0).is_none() {
        sub_executor
            .phantom_execute(
                state.memory,
                state.streams,
                state.rng,
                discriminant,
                a,
                b,
                (c >> 16) as u16,
            )
            .map_err(|e| ExecutionError::Phantom {
                pc: *state.pc,
                discriminant,
                inner: e,
            })?;
    }

    Ok(())
}
