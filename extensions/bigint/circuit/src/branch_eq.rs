use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32BranchEqual256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::Rv32HeapBranchAdapterExecutor;
use openvm_rv32im_circuit::BranchEqualExecutor;
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{common::bytes_to_u64_array, Rv32BranchEqual256Executor, INT256_NUM_LIMBS};

type AdapterExecutor = Rv32HeapBranchAdapterExecutor<2, INT256_NUM_LIMBS>;

impl Rv32BranchEqual256Executor {
    pub fn new(adapter_step: AdapterExecutor, offset: usize, pc_step: u32) -> Self {
        Self(BranchEqualExecutor::new(adapter_step, offset, pc_step))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchEqPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            BranchEqualOpcode::BEQ => Ok($execute_impl::<_, _, false>),
            BranchEqualOpcode::BNE => Ok($execute_impl::<_, _, true>),
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv32BranchEqual256Executor {
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchEqPreCompute>()
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
        let data: &mut BranchEqPreCompute = data.borrow_mut();
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
        let data: &mut BranchEqPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv32BranchEqual256Executor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv32BranchEqual256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchEqPreCompute>>()
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
        let data: &mut E2PreCompute<BranchEqPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<BranchEqPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv32BranchEqual256Executor {}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &BranchEqPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let mut pc = exec_state.pc();
    let rs1_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 =
        exec_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 =
        exec_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let cmp_result = u256_eq(rs1, rs2);
    if cmp_result ^ IS_NE {
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
    let pre_compute: &BranchEqPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BranchEqPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_NE>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BranchEqPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<BranchEqPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_NE>(&pre_compute.data, exec_state);
}

impl Rv32BranchEqual256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BranchEqPreCompute,
    ) -> Result<BranchEqualOpcode, StaticProgramError> {
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
        *data = BranchEqPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        let local_opcode = BranchEqualOpcode::from_usize(
            opcode.local_opcode_idx(Rv32BranchEqual256Opcode::CLASS_OFFSET),
        );
        Ok(local_opcode)
    }
}

fn u256_eq(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
    for i in 0..4 {
        if rs1_u64[i] != rs2_u64[i] {
            return false;
        }
    }
    true
}
