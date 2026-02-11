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
use openvm_rv64im_transpiler::Rv64JalLuiOpcode::{self, JAL};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::base_alu_w::sign_extend_32_to_64;

const RV_J_TYPE_IMM_BITS: usize = 21;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64JalLuiPreCompute {
    signed_imm: i32,
    a: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalLuiExecutor {
    pub offset: usize,
}

fn get_signed_imm<F: PrimeField32>(is_jal: bool, imm: F) -> i32 {
    let imm_f = imm.as_canonical_u32();
    if is_jal {
        if imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)) {
            imm_f as i32
        } else {
            let neg_imm_f = F::ORDER_U32 - imm_f;
            debug_assert!(neg_imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)));
            -(neg_imm_f as i32)
        }
    } else {
        imm_f as i32
    }
}

impl Rv64JalLuiExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64JalLuiPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = Rv64JalLuiOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));
        let is_jal = local_opcode == JAL;
        let signed_imm = get_signed_imm(is_jal, inst.c);

        *data = Rv64JalLuiPreCompute {
            signed_imm,
            a: inst.a.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok((is_jal, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_jal:ident, $enabled:ident) => {
        Ok(match ($is_jal, $enabled) {
            (true, true) => $execute_impl::<_, _, true, true>,
            (true, false) => $execute_impl::<_, _, true, false>,
            (false, true) => $execute_impl::<_, _, false, true>,
            (false, false) => $execute_impl::<_, _, false, false>,
        })
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64JalLuiExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64JalLuiPreCompute>()
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
        let data: &mut Rv64JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_jal, enabled)
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
        let data: &mut Rv64JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_jal, enabled)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64JalLuiExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64JalLuiPreCompute>>()
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
        let data: &mut E2PreCompute<Rv64JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_jal, enabled)
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
        let data: &mut E2PreCompute<Rv64JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_jal, enabled)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64JalLuiExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64JalLuiExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64JalLuiExecutor {
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
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: &Rv64JalLuiPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let Rv64JalLuiPreCompute { a, signed_imm } = *pre_compute;
    let mut pc = exec_state.pc();
    let rd: [u8; 8] = if IS_JAL {
        let rd_val = (pc + DEFAULT_PC_STEP) as u64;
        let next_pc = pc as i32 + signed_imm;
        debug_assert!(next_pc >= 0);
        pc = next_pc as u32;
        rd_val.to_le_bytes()
    } else {
        // LUI: imm << 12, sign-extended to 64 bits
        let imm = signed_imm as u32;
        let rd32 = imm << 12;
        pc += DEFAULT_PC_STEP;
        sign_extend_32_to_64(rd32).to_le_bytes()
    };

    if ENABLED {
        exec_state.vm_write(RV32_REGISTER_AS, a as u32, &rd);
    }
    exec_state.set_pc(pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64JalLuiPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64JalLuiPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_JAL, ENABLED>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64JalLuiPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Rv64JalLuiPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_JAL, ENABLED>(&pre_compute.data, exec_state);
}
