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
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{adapters::imm_to_bytes, BaseAluExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct BaseAluPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    /// Return `is_imm`, true if `e` is RV32_IMM_AS.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BaseAluPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        let e_u32 = e.as_canonical_u32();
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = BaseAluPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(is_imm)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match (
                $is_imm,
                BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)),
            ) {
                (true, BaseAluOpcode::ADD) => $execute_impl::<_, _, true, AddOp>,
                (false, BaseAluOpcode::ADD) => $execute_impl::<_, _, false, AddOp>,
                (true, BaseAluOpcode::SUB) => $execute_impl::<_, _, true, SubOp>,
                (false, BaseAluOpcode::SUB) => $execute_impl::<_, _, false, SubOp>,
                (true, BaseAluOpcode::XOR) => $execute_impl::<_, _, true, XorOp>,
                (false, BaseAluOpcode::XOR) => $execute_impl::<_, _, false, XorOp>,
                (true, BaseAluOpcode::OR) => $execute_impl::<_, _, true, OrOp>,
                (false, BaseAluOpcode::OR) => $execute_impl::<_, _, false, OrOp>,
                (true, BaseAluOpcode::AND) => $execute_impl::<_, _, true, AndOp>,
                (false, BaseAluOpcode::AND) => $execute_impl::<_, _, false, AndOp>,
            },
        )
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
    for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BaseAluPreCompute>()
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
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, inst.opcode, self.offset)
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
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, inst.opcode, self.offset)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BaseAluPreCompute>>()
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
        let data: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, inst.opcode, self.offset)
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
        let data: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, inst.opcode, self.offset)
    }
}


#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F> for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> where F: PrimeField32
{
    fn is_aot_supported(&self, instruction: &Instruction<F>) -> bool {
        true
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let mut asm_str = String::new();
        let a: i16 = to_i16(inst.a);
        let b: i16 = to_i16(inst.b);
        let c: i16 = to_i16(inst.c);
        let e: i16 = to_i16(inst.e);

        let xmm_map_reg_a = if (a / 4) % 2 == 0 {
            a / 8
        } else {
            ((a / 4) - 1) / 2 // floor((a/4)/2)
        };

        let xmm_map_reg_b = if (b / 4) % 2 == 0 {
            b / 8
        } else {
            ((b / 4) - 1) / 2
        };

        // [a:4]_1 <- [b:4]_1
        if (b / 4) % 2 == 0 {
            // get the [0:32) bits of xmm_map_reg_b
            asm_str += &format!("   vmovd {}, xmm{}\n", REG_A, xmm_map_reg_b);
        } else {
            // get the [32:64) bits of xmm_map_reg_b
            asm_str += &format!("   vpextrd {}, xmm{}, 1\n", REG_A_W, xmm_map_reg_b);
        }

        let mut asm_opcode = String::new();
        if inst.opcode == BaseAluOpcode::ADD.global_opcode() {
            asm_opcode += "add";
        } else if inst.opcode == BaseAluOpcode::SUB.global_opcode() {
            asm_opcode += "sub";
        } else if inst.opcode == BaseAluOpcode::AND.global_opcode() {
            asm_opcode += "and";
        } else if inst.opcode == BaseAluOpcode::OR.global_opcode() {
            asm_opcode += "or";
        } else if inst.opcode == BaseAluOpcode::XOR.global_opcode() {
            asm_opcode += "xor";
        }
        if e == 0 {
            // [a:4]_1 <- [a:4]_1 + c
            asm_str += &format!("   {} {}, {}\n", asm_opcode, REG_A, c);
        } else {
            // [a:4]_1 <- [a:4]_1 + [c:4]_1
            assert_eq!(c % 4, 0);
            let xmm_map_reg_c = if (c / 4) % 2 == 0 {
                c / 8
            } else {
                ((c / 4) - 1) / 2
            };

            // XMM -> General Register
            if (c / 4) % 2 == 0 {
                // get the [0:32) bits of xmm_map_reg_c
                asm_str += &format!("   vmovd {}, xmm{}\n", REG_C, xmm_map_reg_c);
            } else {
                // get the [32:64) bits of xmm_map_reg_b
                asm_str += &format!("   vpextrd {REG_C_W}, xmm{}, 1\n", xmm_map_reg_c);
            }

            // reg_a += reg_c
            asm_str += &format!("   {} {}, {}\n", asm_opcode, REG_A, REG_C);
        }

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
        
        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &BaseAluPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };
    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd = <OP as AluOp>::compute(rs1, rs2);
    let rd = rd.to_le_bytes();
    exec_state.vm_write::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BaseAluPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BaseAluPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, instret, pc, exec_state);
}

trait AluOp {
    fn compute(rs1: u32, rs2: u32) -> u32;
}
struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_add(rs2)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_sub(rs2)
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 ^ rs2
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 | rs2
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 & rs2
    }
}
