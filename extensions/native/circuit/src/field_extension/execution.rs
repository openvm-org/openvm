use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FieldExtensionOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::{FieldExtension, FieldExtensionCoreExecutor, EXT_DEG};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FieldExtensionPreCompute {
    a: u32,
    b: u32,
    c: u32,
}

impl<A> FieldExtensionCoreExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FieldExtensionPreCompute,
    ) -> Result<u8, StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        let local_opcode = FieldExtensionOpcode::from_usize(
            opcode.local_opcode_idx(FieldExtensionOpcode::CLASS_OFFSET),
        );

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();

        if d != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if e != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = FieldExtensionPreCompute { a, b, c };

        Ok(local_opcode as u8)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $opcode:ident) => {
        match $opcode {
            0 => Ok($execute_impl::<_, _, 0>), // FE4ADD
            1 => Ok($execute_impl::<_, _, 1>), // FE4SUB
            2 => Ok($execute_impl::<_, _, 2>), // BBE4MUL
            3 => Ok($execute_impl::<_, _, 3>), // BBE4DIV
            _ => panic!("Invalid field extension opcode: {}", $opcode),
        }
    };
}

impl<F, A> Executor<F> for FieldExtensionCoreExecutor<A>
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
        let pre_compute: &mut FieldExtensionPreCompute = data.borrow_mut();

        let opcode = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_tco_handler, opcode)
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<FieldExtensionPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut FieldExtensionPreCompute = data.borrow_mut();

        let opcode = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_impl, opcode)
    }
}

impl<F, A> MeteredExecutor<F> for FieldExtensionCoreExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<FieldExtensionPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<FieldExtensionPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        dispatch!(execute_e2_impl, opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<FieldExtensionPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        dispatch!(execute_e2_tco_handler, opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const OPCODE: u8>(
    pre_compute: &FieldExtensionPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let y: [F; EXT_DEG] = vm_state.vm_read::<F, EXT_DEG>(AS::Native as u32, pre_compute.b);
    let z: [F; EXT_DEG] = vm_state.vm_read::<F, EXT_DEG>(AS::Native as u32, pre_compute.c);

    let x = match OPCODE {
        0 => FieldExtension::add(y, z),      // FE4ADD
        1 => FieldExtension::subtract(y, z), // FE4SUB
        2 => FieldExtension::multiply(y, z), // BBE4MUL
        3 => FieldExtension::divide(y, z),   // BBE4DIV
        _ => panic!("Invalid field extension opcode: {OPCODE}"),
    };

    vm_state.vm_write(AS::Native as u32, pre_compute.a, &x);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[create_tco_handler]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const OPCODE: u8>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExtensionPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OPCODE>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const OPCODE: u8>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExtensionPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OPCODE>(&pre_compute.data, vm_state);
}
