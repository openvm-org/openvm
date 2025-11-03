use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

#[cfg(feature = "aot")]
use crate::common::{gpr_to_rv32_register, rv32_register_to_gpr};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV32_REGISTER_AS,
};
#[cfg(feature = "aot")]
use openvm_instructions::{LocalOpcode, VmOpcode};
#[cfg(feature = "aot")]
use openvm_rv32im_transpiler::Rv32JalrOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::Rv32JalrExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalrPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl<A> Rv32JalrExecutor<A> {
    /// Return true if enabled.
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut JalrPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let imm_extended = inst.c.as_canonical_u32() + inst.g.as_canonical_u32() * 0xffff0000;
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = JalrPreCompute {
            imm_extended,
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok(enabled)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $enabled:ident) => {
        if $enabled {
            Ok($execute_impl::<_, _, true>)
        } else {
            Ok($execute_impl::<_, _, false>)
        }
    };
}

#[cfg(feature = "aot")]
const REG_B_W: &str = "eax";
#[cfg(feature = "aot")]
const REG_A_W: &str = "ecx";
#[cfg(feature = "aot")]
const REG_PC: &str = "r13";
#[cfg(feature = "aot")]
const REG_PC_W: &str = "r13d";
#[cfg(feature = "aot")]
const REG_INSTRET: &str = "r14";
#[cfg(feature = "aot")]
const REG_A: &str = "rcx"; // used when building jump address

impl<F, A> InterpreterExecutor<F> for Rv32JalrExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalrPreCompute>()
    }
    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
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
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for Rv32JalrExecutor<A>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &Instruction<F>) -> bool {
        inst.opcode == Rv32JalrOpcode::JALR.global_opcode()
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        let mut asm_str = String::new();
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let a = to_i16(inst.a);
        let b = to_i16(inst.b);
        if a % 4 != 0 || b % 4 != 0 {
            return Err(AotError::InvalidInstruction);
        }
        let imm_extended = inst.c.as_canonical_u32() + inst.g.as_canonical_u32() * 0xffff0000;
        let write_rd = !inst.f.is_zero();

        asm_str += &rv32_register_to_gpr((b / 4) as u8, REG_B_W);

        asm_str += &format!("   add {}, {}\n", REG_B_W, imm_extended);
        asm_str += &format!("   and {}, -2\n", REG_B_W); // clear bit 0 per RISC-V jalr
        asm_str += &format!("   mov {}, {}\n", REG_PC_W, REG_B_W); // zero-extend into r13

        if write_rd {
            let next_pc = pc.wrapping_add(DEFAULT_PC_STEP);
            asm_str += &format!("   mov {}, {}\n", REG_A_W, next_pc);
            asm_str += &gpr_to_rv32_register(REG_A_W, (a / 4) as u8);
        }

        asm_str += &format!("   add {}, 1\n", REG_INSTRET);
        asm_str += "   lea rdx, [rip + map_pc_base]\n";
        asm_str += &format!("   movsxd {}, [rdx + {}]\n", REG_A, REG_PC);
        asm_str += "   add rcx, rdx\n";
        asm_str += "   jmp rcx\n";
        Ok(asm_str)
    }
}

impl<F, A> MeteredExecutor<F> for Rv32JalrExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalrPreCompute>>()
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
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
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
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: &JalrPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 = u32::from_le_bytes(rs1);
    let to_pc = rs1.wrapping_add(pre_compute.imm_extended);
    let to_pc = to_pc - (to_pc & 1);
    debug_assert!(to_pc < (1 << PC_BITS));
    let rd = (*pc + DEFAULT_PC_STEP).to_le_bytes();

    if ENABLED {
        exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    }

    *pc = to_pc;
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_left: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &JalrPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, ENABLED>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalrPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, ENABLED>(&pre_compute.data, instret, pc, exec_state);
}
