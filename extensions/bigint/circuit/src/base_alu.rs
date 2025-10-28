use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_bigint_transpiler::Rv32BaseAlu256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::Rv32HeapAdapterExecutor;
use openvm_rv32im_circuit::BaseAluExecutor;
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{bytes_to_u64_array, u64_array_to_bytes},
    Rv32BaseAlu256Executor, INT256_NUM_LIMBS,
};

type AdapterExecutor = Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32BaseAlu256Executor {
    pub fn new(adapter: AdapterExecutor, offset: usize) -> Self {
        Self(BaseAluExecutor::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow)]
struct BaseAluPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            BaseAluOpcode::ADD => $execute_impl::<_, _, AddOp>,
            BaseAluOpcode::SUB => $execute_impl::<_, _, SubOp>,
            BaseAluOpcode::XOR => $execute_impl::<_, _, XorOp>,
            BaseAluOpcode::OR => $execute_impl::<_, _, OrOp>,
            BaseAluOpcode::AND => $execute_impl::<_, _, AndOp>,
        })
    };
}

impl<F: PrimeField32> Executor<F> for Rv32BaseAlu256Executor {
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
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, local_opcode)
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
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for Rv32BaseAlu256Executor {
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
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, local_opcode)
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
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: &BaseAluPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs1 =
        exec_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 =
        exec_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let rd = <OP as AluOp>::compute(rs1, rs2);
    exec_state.vm_write(RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);
    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BaseAluPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: AluOp>(
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
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, instret, pc, exec_state);
}

impl Rv32BaseAlu256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BaseAluPreCompute,
    ) -> Result<BaseAluOpcode, StaticProgramError> {
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
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BaseAluPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode =
            BaseAluOpcode::from_usize(opcode.local_opcode_idx(Rv32BaseAlu256Opcode::CLASS_OFFSET));
        Ok(local_opcode)
    }
}

trait AluOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS];
}
struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; 4];
        let (res, mut carry) = rs1_u64[0].overflowing_add(rs2_u64[0]);
        rd_u64[0] = res;
        for i in 1..4 {
            let (res1, c1) = rs1_u64[i].overflowing_add(rs2_u64[i]);
            let (res2, c2) = res1.overflowing_add(carry as u64);
            carry = c1 || c2;
            rd_u64[i] = res2;
        }
        u64_array_to_bytes(rd_u64)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; 4];
        let (res, mut borrow) = rs1_u64[0].overflowing_sub(rs2_u64[0]);
        rd_u64[0] = res;
        for i in 1..4 {
            let (res1, c1) = rs1_u64[i].overflowing_sub(rs2_u64[i]);
            let (res2, c2) = res1.overflowing_sub(borrow as u64);
            borrow = c1 || c2;
            rd_u64[i] = res2;
        }
        u64_array_to_bytes(rd_u64)
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; 4];
        // Compiler will expand this loop.
        for i in 0..4 {
            rd_u64[i] = rs1_u64[i] ^ rs2_u64[i];
        }
        u64_array_to_bytes(rd_u64)
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; 4];
        // Compiler will expand this loop.
        for i in 0..4 {
            rd_u64[i] = rs1_u64[i] | rs2_u64[i];
        }
        u64_array_to_bytes(rd_u64)
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; 4];
        // Compiler will expand this loop.
        for i in 0..4 {
            rd_u64[i] = rs1_u64[i] & rs2_u64[i];
        }
        u64_array_to_bytes(rd_u64)
    }
}
