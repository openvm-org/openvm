use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::RV64_REGISTER_AS,
    LocalOpcode,
};
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[allow(unused_imports)]
use crate::{common::*, AddSubExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct AddSubPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AddSubExecutor<A, NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut AddSubPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        if (d.as_canonical_u32() != RV64_REGISTER_AS)
            || (e.as_canonical_u32() != RV64_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = AddSubPreCompute {
            c: c.as_canonical_u32(),
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)) {
                BaseAluOpcode::ADD => $execute_impl::<_, _, AddOp>,
                BaseAluOpcode::SUB => $execute_impl::<_, _, SubOp>,
                _ => unreachable!("AddSubExecutor received non-ADD/SUB opcode"),
            },
        )
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for AddSubExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AddSubPreCompute>()
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
        let data: &mut AddSubPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, inst.opcode, self.offset)
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
        let data: &mut AddSubPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, inst.opcode, self.offset)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for AddSubExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AddSubPreCompute>>()
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
        let data: &mut E2PreCompute<AddSubPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, inst.opcode, self.offset)
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
        let data: &mut E2PreCompute<AddSubPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, inst.opcode, self.offset)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for AddSubExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        true
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let mut asm_str = String::new();

        let a: i16 = to_i16(inst.a);
        let b: i16 = to_i16(inst.b);
        let c: i16 = to_i16(inst.c);
        let str_reg_a = if RISCV_TO_X86_OVERRIDE_MAP[(a / 4) as usize].is_some() {
            RISCV_TO_X86_OVERRIDE_MAP[(a / 4) as usize].unwrap()
        } else {
            REG_A_W
        };

        let mut asm_opcode = String::new();
        if inst.opcode == BaseAluOpcode::ADD.global_opcode() {
            asm_opcode += "add";
        } else if inst.opcode == BaseAluOpcode::SUB.global_opcode() {
            asm_opcode += "sub";
        }

        if a == c {
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr((c / 4) as u8, REG_C_W, true);
            asm_str += &delta_str_c;
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr((b / 4) as u8, str_reg_a, true);
            asm_str += &delta_str_b;
            asm_str += &format!("   {asm_opcode} {gpr_reg_b}, {gpr_reg_c}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_b, (a / 4) as u8);
        } else {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr((b / 4) as u8, str_reg_a, true);
            asm_str += &delta_str_b;
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr((c / 4) as u8, REG_C_W, false);
            asm_str += &delta_str_c;
            asm_str += &format!("   {asm_opcode} {gpr_reg_b}, {gpr_reg_c}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_b, (a / 4) as u8);
        }

        Ok(asm_str)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotMeteredExecutor<F>
    for AddSubExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
    fn generate_x86_metered_asm(
        &self,
        inst: &Instruction<F>,
        pc: u32,
        chip_idx: usize,
        config: &SystemConfig,
    ) -> Result<String, AotError> {
        let mut asm_str = self.generate_x86_asm(inst, pc)?;
        asm_str += &update_height_change_asm(chip_idx, 1)?;
        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: &AddSubPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read_bytes::<8>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2 = exec_state.vm_read_bytes::<8>(RV64_REGISTER_AS, pre_compute.c);
    let rs1 = u64::from_le_bytes(rs1);
    let rs2 = u64::from_le_bytes(rs2);
    let rd = <OP as AluOp>::compute(rs1, rs2);
    let rd = rd.to_le_bytes();
    exec_state.vm_write_bytes::<8>(RV64_REGISTER_AS, pre_compute.a as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &AddSubPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<AddSubPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<AddSubPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<AddSubPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait AluOp {
    fn compute(rs1: u64, rs2: u64) -> u64;
}
struct AddOp;
struct SubOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_add(rs2)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_sub(rs2)
    }
}
