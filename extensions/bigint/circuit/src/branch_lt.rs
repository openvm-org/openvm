use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_bigint_transpiler::Rv32BranchLessThan256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::Rv32HeapBranchAdapterExecutor;
use openvm_rv32im_circuit::BranchLessThanExecutor;
use openvm_rv32im_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{i256_lt, u256_lt},
    Rv32BranchLessThan256Executor, INT256_NUM_LIMBS,
};

type AdapterExecutor = Rv32HeapBranchAdapterExecutor<2, INT256_NUM_LIMBS>;

impl Rv32BranchLessThan256Executor {
    pub fn new(adapter: AdapterExecutor, offset: usize) -> Self {
        Self(BranchLessThanExecutor::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchLtPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            BranchLessThanOpcode::BLT => $execute_impl::<_, _, BltOp>,
            BranchLessThanOpcode::BLTU => $execute_impl::<_, _, BltuOp>,
            BranchLessThanOpcode::BGE => $execute_impl::<_, _, BgeOp>,
            BranchLessThanOpcode::BGEU => $execute_impl::<_, _, BgeuOp>,
        })
    };
}

impl<F: PrimeField32> Executor<F> for Rv32BranchLessThan256Executor {
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchLtPreCompute>()
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
        let data: &mut BranchLtPreCompute = data.borrow_mut();
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
        let data: &mut BranchLtPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for Rv32BranchLessThan256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchLtPreCompute>>()
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
        let data: &mut E2PreCompute<BranchLtPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<BranchLtPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: BranchLessThanOp>(
    pre_compute: &BranchLtPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 =
        exec_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 =
        exec_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let cmp_result = OP::compute(rs1, rs2);
    if cmp_result {
        *pc = (*pc as isize + pre_compute.imm) as u32;
    } else {
        *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    }
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: BranchLessThanOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BranchLtPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: BranchLessThanOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BranchLtPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, instret, pc, exec_state);
}

impl Rv32BranchLessThan256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BranchLtPreCompute,
    ) -> Result<BranchLessThanOpcode, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BranchLtPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        let local_opcode = BranchLessThanOpcode::from_usize(
            opcode.local_opcode_idx(Rv32BranchLessThan256Opcode::CLASS_OFFSET),
        );
        Ok(local_opcode)
    }
}

trait BranchLessThanOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool;
}
struct BltOp;
struct BltuOp;
struct BgeOp;
struct BgeuOp;

impl BranchLessThanOp for BltOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        i256_lt(rs1, rs2)
    }
}
impl BranchLessThanOp for BltuOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        u256_lt(rs1, rs2)
    }
}
impl BranchLessThanOp for BgeOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        !i256_lt(rs1, rs2)
    }
}
impl BranchLessThanOp for BgeuOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        !u256_lt(rs1, rs2)
    }
}
