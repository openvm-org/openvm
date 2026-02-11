use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::online::{GuestMemory, TracingMemory},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv64im_transpiler::Rv64BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64BranchEqualPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64BranchEqualExecutor {
    pub offset: usize,
}

impl Rv64BranchEqualExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64BranchEqualPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = Rv64BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = Rv64BranchEqualPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(local_opcode == Rv64BranchEqualOpcode::BNE)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_bne:ident) => {
        if $is_bne {
            Ok($execute_impl::<_, _, true>)
        } else {
            Ok($execute_impl::<_, _, false>)
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64BranchEqualExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64BranchEqualPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut Rv64BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_bne)
    }

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
        let data: &mut Rv64BranchEqualPreCompute = data.borrow_mut();
        let is_bne = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_bne)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64BranchEqualExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64BranchEqualPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<Rv64BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_bne)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<Rv64BranchEqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_bne = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_bne)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64BranchEqualExecutor {
    fn get_opcode_name(&self, _opcode: usize) -> String {
        panic!("not yet implemented")
    }

    fn execute(
        &self,
        _state: VmStateMut<F, TracingMemory, RA>,
        _instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        panic!("not yet implemented")
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv64BranchEqualExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64BranchEqualExecutor {
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_metered_asm(
        &self,
        _inst: &Instruction<F>,
        _pc: u32,
        _chip_idx: usize,
        _config: &SystemConfig,
    ) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &Rv64BranchEqualPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let mut pc = exec_state.pc();
    let rs1 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.b as u32);
    if (rs1 == rs2) ^ IS_NE {
        pc = (pc as isize + pre_compute.imm) as u32;
    } else {
        pc = pc.wrapping_add(DEFAULT_PC_STEP);
    }
    exec_state.set_pc(pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64BranchEqualPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64BranchEqualPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_NE>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64BranchEqualPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64BranchEqualPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_NE>(&pre_compute.data, exec_state);
}
