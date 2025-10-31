use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, VmOpcode
};
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::LessThanExecutor;
use crate::adapters::imm_to_bytes;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LessThanPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LessThanPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = LessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();

        *data = LessThanPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok((is_imm, local_opcode == LessThanOpcode::SLTU))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_sltu:ident) => {
        match ($is_imm, $is_sltu) {
            (true, true) => Ok($execute_impl::<_, _, true, true>),
            (true, false) => Ok($execute_impl::<_, _, true, false>),
            (false, true) => Ok($execute_impl::<_, _, false, true>),
            (false, false) => Ok($execute_impl::<_, _, false, false>),
        }
    };
}

// Callee saved
const REG_EXEC_STATE_PTR: &str = "rbx";
const REG_INSNS_PTR: &str = "rbp";
const REG_PC: &str = "r13";
const REG_INSTRET: &str = "r14";
const REG_GUEST_MEM_PTR: &str = "r15";

// Caller saved
const REG_B: &str = "rax";
const REG_B_W: &str = "eax";

const REG_A: &str = "rcx";
const REG_A_W: &str = "ecx";

const REG_FOURTH_ARG: &str = "rcx";
const REG_THIRD_ARG: &str = "rdx";
const REG_SECOND_ARG: &str = "rsi";
const REG_FIRST_ARG: &str = "rdi";
const REG_RETURN_VAL: &str = "rax";

const REG_C: &str = "r10";
const REG_C_W: &str = "r10d";
const REG_C_B: &str = "r10b";
const REG_AUX: &str = "r11";

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LessThanPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LessThanPreCompute = data.borrow_mut();
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, is_imm, is_sltu)
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
        let pre_compute: &mut LessThanPreCompute = data.borrow_mut();
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, is_imm, is_sltu)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &Instruction<F>) -> bool {
        false
    }
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let mut asm_str = String::new();
        let a : i16 = to_i16(inst.a);
        let b : i16 = to_i16(inst.b);
        let c : i16 = to_i16(inst.c);
        let e : i16 = to_i16(inst.e);
        assert!(a % 4 == 0, "instruction.a must be a multiple of 4");
        assert!(b % 4 == 0, "instruction.b must be a multiple of 4");
        let xmm_map_reg_a = if (a/4) % 2 == 0 {
            a/8
        } else {
            ((a/4)-1)/2 // floor((a/4)/2)
        };
        let xmm_map_reg_b = if (b/4) % 2 == 0 {
            b/8
        } else {
            ((b/4)-1)/2
        };
        // [a:4]_1 <- [b:4]_1
        if (b/4)%2 == 0 {
            // get the [0:32) bits of xmm_map_reg_b
            asm_str += &format!("   vmovd {}, xmm{}\n", REG_A, xmm_map_reg_b);
        } else {
            // get the [32:64) bits of xmm_map_reg_b
            asm_str += &format!("   vpextrd {}, xmm{}, 1\n", REG_A_W, xmm_map_reg_b);
        }
        let mut asm_opcode = String::new();
        if inst.opcode == LessThanOpcode::SLT.global_opcode() {
            asm_opcode += "setl";
        } else if inst.opcode == LessThanOpcode::SLTU.global_opcode() {
            asm_opcode += "setb";
        }
        if e == 0 {
            asm_str += &format!("   cmp {}, {}\n", REG_A_W, c);
        } else {
            // [a:4]_1 <- [a:4]_1 + [c:4]_1
            assert_eq!(c % 4, 0);
            let xmm_map_reg_c = if (c/4) % 2 == 0 {
                c/8
            } else {
                ((c/4)-1)/2
            };
            // XMM -> General Register
            if (c/4) % 2 == 0 {
                // get the [0:32) bits of xmm_map_reg_c
                asm_str += &format!("   vmovd {}, xmm{}\n", REG_C, xmm_map_reg_c);
            } else {
                // get the [32:64) bits of xmm_map_reg_b
                asm_str += &format!("   vpextrd {REG_C_W}, xmm{}, 1\n", xmm_map_reg_c);
            }
            asm_str += &format!("   cmp {}, {}\n", REG_A_W, REG_C_W);
        }
        // Set REG_A to 1 if less than (signed), 0 otherwise
        asm_str += &format!("   {} cl\n", asm_opcode);  // setl cl or setb cl
        asm_str += &format!("   movzx {}, cl\n", REG_A_W);  // zero-extend to 32-bit
        // General Register -> XMM
        if (a / 4) % 2 == 0 {
            // make the [0:32) bits of xmm_map_reg_a equal to REG_A_W without modifying the other
            // bits
            asm_str += &format!(
                "   vpinsrd xmm{}, xmm{}, {REG_A_W}, 0\n",
                xmm_map_reg_a, xmm_map_reg_a
            );
        } else {
            // make the [32:64) bits of xmm_map_reg_a equal to REG_A_W without modifying the other
            // bits
            asm_str += &format!(
                "   vpinsrd xmm{}, xmm{}, {REG_A_W}, 1\n",
                xmm_map_reg_a, xmm_map_reg_a
            );
        }
        asm_str += &format!("   add {}, {}\n", REG_PC, 4);
        asm_str += &format!("   add {}, {}\n", REG_INSTRET, 1);
        // let it fall to the next instruction
        Ok(asm_str)
    }
}


impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LessThanPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, is_imm, is_sltu)
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
        let pre_compute: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, is_imm, is_sltu)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
>(
    pre_compute: &LessThanPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if E_IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };
    let cmp_result = if IS_U32 {
        u32::from_le_bytes(rs1) < u32::from_le_bytes(rs2)
    } else {
        i32::from_le_bytes(rs1) < i32::from_le_bytes(rs2)
    };
    let mut rd = [0u8; RV32_REGISTER_NUM_LIMBS];
    rd[0] = cmp_result as u8;
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    *pc += DEFAULT_PC_STEP;
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &LessThanPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<LessThanPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32>(&pre_compute.data, instret, pc, exec_state);
}
