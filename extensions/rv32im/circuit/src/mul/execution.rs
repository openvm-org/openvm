use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
#[cfg(feature = "aot")]
use openvm_instructions::VmOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::MultiplicationExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MultiPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A, const LIMB_BITS: usize> MultiplicationExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MultiPreCompute,
    ) -> Result<(), StaticProgramError> {
        assert_eq!(
            MulOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset)),
            MulOpcode::MUL
        );
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = MultiPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

// Callee saved registers
#[cfg(feature = "aot")]
const REG_PC: &str = "r13";
#[cfg(feature = "aot")]
const REG_INSTRET: &str = "r14";

// Caller saved registers
#[cfg(feature = "aot")]
const REG_A: &str = "rax";
#[cfg(feature = "aot")]
const REG_A_W: &str = "eax";
#[cfg(feature = "aot")]
const REG_B: &str = "rcx";
#[cfg(feature = "aot")]
const REG_B_W: &str = "ecx";

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for MultiplicationExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<MultiPreCompute>()
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
        let pre_compute: &mut MultiPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_impl)
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
        let pre_compute: &mut MultiPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for MultiplicationExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &Instruction<F>) -> bool {
        inst.opcode == MulOpcode::MUL.global_opcode()
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let map_reg = |reg: i16| -> (i16, u8) {
            let index = reg / 4;
            (index / 2, (index % 2) as u8)
        };
        let a = to_i16(inst.a);
        let b = to_i16(inst.b);
        let c = to_i16(inst.c);

        let (xmm_a, lane_a) = map_reg(a);
        let (xmm_b, lane_b) = map_reg(b);
        let (xmm_c, lane_c) = map_reg(c);

        let mut asm_str = String::new();

        if lane_b == 0 {
            asm_str += &format!("   vmovd {}, xmm{}\n", REG_B_W, xmm_b);
        } else {
            asm_str += &format!("   vpextrd {}, xmm{}, {}\n", REG_B_W, xmm_b, lane_b);
        }

        if lane_c == 0 {
            asm_str += &format!("   vmovd {}, xmm{}\n", REG_A_W, xmm_c);
        } else {
            asm_str += &format!("   vpextrd {}, xmm{}, {}\n", REG_A_W, xmm_c, lane_c);
        }

        asm_str += &format!("   imul {}, {}\n", REG_A_W, REG_B_W);
        asm_str += &format!(
            "   vpinsrd xmm{}, xmm{}, {REG_A_W}, {}\n",
            xmm_a, xmm_a, lane_a
        );
        asm_str += &format!("   add {}, 4\n", REG_PC);
        asm_str += &format!("   add {}, 1\n", REG_INSTRET);

        Ok(asm_str)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for MultiplicationExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MultiPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<MultiPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_impl)
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
        let pre_compute: &mut E2PreCompute<MultiPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &MultiPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd = rs1.wrapping_mul(rs2);
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd.to_le_bytes());

    *pc += DEFAULT_PC_STEP;
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MultiPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MultiPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, instret, pc, exec_state);
}
