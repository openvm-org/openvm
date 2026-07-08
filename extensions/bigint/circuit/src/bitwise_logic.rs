use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_bigint_transpiler::Rv64BaseAlu256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_circuit::{adapters::rv64_bytes_to_u32, BitwiseLogicExecutor};
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{bytes_to_u64_array, read_int256, u64_array_to_bytes, write_int256},
    AluAdapterExecutor, Rv64BitwiseLogic256Executor, INT256_NUM_U64_LIMBS, INT256_NUM_U8_LIMBS,
};

impl Rv64BitwiseLogic256Executor {
    pub fn new(adapter: AluAdapterExecutor, offset: usize) -> Self {
        Self(BitwiseLogicExecutor::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow)]
struct BitwiseLogicPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            BaseAluOpcode::XOR => $execute_impl::<F, _, XorOp>,
            BaseAluOpcode::OR => $execute_impl::<F, _, OrOp>,
            BaseAluOpcode::AND => $execute_impl::<F, _, AndOp>,
            _ => unreachable!("Rv64BitwiseLogic256Executor received non-XOR/OR/AND opcode"),
        })
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64BitwiseLogic256Executor {
    fn pre_compute_size(&self) -> usize {
        size_of::<BitwiseLogicPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BitwiseLogicPreCompute = data.borrow_mut();
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
        let data: &mut BitwiseLogicPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv64BitwiseLogic256Executor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64BitwiseLogic256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BitwiseLogicPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<BitwiseLogicPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<BitwiseLogicPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, local_opcode)
    }
}
#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64BitwiseLogic256Executor {}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: &BitwiseLogicPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1_ptr =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.c as u32);
    let rd_ptr =
        exec_state.vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.a as u32);
    let rs1 = read_int256(exec_state, RV64_MEMORY_AS, rv64_bytes_to_u32(rs1_ptr));
    let rs2 = read_int256(exec_state, RV64_MEMORY_AS, rv64_bytes_to_u32(rs2_ptr));
    let rd = <OP as AluOp>::compute(rs1, rs2);
    write_int256(exec_state, RV64_MEMORY_AS, rv64_bytes_to_u32(rd_ptr), &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &BitwiseLogicPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BitwiseLogicPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BitwiseLogicPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BitwiseLogicPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

impl Rv64BitwiseLogic256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BitwiseLogicPreCompute,
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
        if d.as_canonical_u32() != RV64_REGISTER_AS || e_u32 != RV64_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BitwiseLogicPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode =
            BaseAluOpcode::from_usize(opcode.local_opcode_idx(Rv64BaseAlu256Opcode::CLASS_OFFSET));
        Ok(local_opcode)
    }
}

trait AluOp {
    fn compute(
        rs1: [u8; INT256_NUM_U8_LIMBS],
        rs2: [u8; INT256_NUM_U8_LIMBS],
    ) -> [u8; INT256_NUM_U8_LIMBS];
}
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(
        rs1: [u8; INT256_NUM_U8_LIMBS],
        rs2: [u8; INT256_NUM_U8_LIMBS],
    ) -> [u8; INT256_NUM_U8_LIMBS] {
        let rs1_u64 = bytes_to_u64_array(rs1);
        let rs2_u64 = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; INT256_NUM_U64_LIMBS];
        // Compiler will expand this loop.
        for i in 0..INT256_NUM_U64_LIMBS {
            rd_u64[i] = rs1_u64[i] ^ rs2_u64[i];
        }
        u64_array_to_bytes(rd_u64)
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(
        rs1: [u8; INT256_NUM_U8_LIMBS],
        rs2: [u8; INT256_NUM_U8_LIMBS],
    ) -> [u8; INT256_NUM_U8_LIMBS] {
        let rs1_u64 = bytes_to_u64_array(rs1);
        let rs2_u64 = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; INT256_NUM_U64_LIMBS];
        // Compiler will expand this loop.
        for i in 0..INT256_NUM_U64_LIMBS {
            rd_u64[i] = rs1_u64[i] | rs2_u64[i];
        }
        u64_array_to_bytes(rd_u64)
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(
        rs1: [u8; INT256_NUM_U8_LIMBS],
        rs2: [u8; INT256_NUM_U8_LIMBS],
    ) -> [u8; INT256_NUM_U8_LIMBS] {
        let rs1_u64 = bytes_to_u64_array(rs1);
        let rs2_u64 = bytes_to_u64_array(rs2);
        let mut rd_u64 = [0u64; INT256_NUM_U64_LIMBS];
        // Compiler will expand this loop.
        for i in 0..INT256_NUM_U64_LIMBS {
            rd_u64[i] = rs1_u64[i] & rs2_u64[i];
        }
        u64_array_to_bytes(rd_u64)
    }
}
