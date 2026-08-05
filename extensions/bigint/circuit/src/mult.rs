use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Mul256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::bytes_to_u32;
use openvm_riscv_transpiler::MulOpcode;

use crate::{
    common::{bytes_to_u32_array, read_int256, u32_array_to_bytes, write_int256},
    Multiplication256Executor, INT256_NUM_U32_LIMBS, INT256_NUM_U8_LIMBS,
};

impl Multiplication256Executor {
    pub fn new() -> Self {
        Self
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MultPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl InterpreterExecutor for Multiplication256Executor {
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            MulOpcode::from_usize(opcode - Mul256Opcode::CLASS_OFFSET)
        )
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<MultPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut MultPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut MultPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_>)
    }
}

impl InterpreterMeteredExecutor for Multiplication256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MultPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<MultPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<_>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<MultPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<_>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait>(
    pre_compute: &MultPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let rs1_ptr = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.c as u32);
    let rd_ptr = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.a as u32);
    let rs1 = read_int256(exec_state, MEMORY_AS, bytes_to_u32(rs1_ptr))?;
    let rs2 = read_int256(exec_state, MEMORY_AS, bytes_to_u32(rs2_ptr))?;
    let rd = u256_mul(rs1, rs2);
    write_int256(exec_state, MEMORY_AS, bytes_to_u32(rd_ptr), &rd)?;

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &MultPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<MultPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<MultPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<MultPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state)
}

impl Multiplication256Executor {
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut MultPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_u32();
        if d.as_u32() != REGISTER_AS || e_u32 != MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode =
            MulOpcode::from_usize(opcode.local_opcode_idx(Mul256Opcode::CLASS_OFFSET));
        if local_opcode != MulOpcode::MUL {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = MultPreCompute {
            a: a.as_u32() as u8,
            b: b.as_u32() as u8,
            c: c.as_u32() as u8,
        };
        Ok(())
    }
}

#[inline(always)]
pub(crate) fn u256_mul(
    rs1: [u8; INT256_NUM_U8_LIMBS],
    rs2: [u8; INT256_NUM_U8_LIMBS],
) -> [u8; INT256_NUM_U8_LIMBS] {
    let rs1_u32 = bytes_to_u32_array(rs1);
    let rs2_u32 = bytes_to_u32_array(rs2);
    let mut rd = [0u32; INT256_NUM_U32_LIMBS];
    for i in 0..INT256_NUM_U32_LIMBS {
        let mut carry = 0u64;
        for j in 0..(INT256_NUM_U32_LIMBS - i) {
            let res = rs1_u32[i] as u64 * rs2_u32[j] as u64 + rd[i + j] as u64 + carry;
            rd[i + j] = res as u32;
            carry = res >> u32::BITS;
        }
    }
    u32_array_to_bytes(rd)
}

#[cfg(test)]
mod tests {
    use alloy_primitives::U256;
    use rand::{prelude::StdRng, Rng, SeedableRng};

    use crate::{
        common::u64_array_to_bytes, mult::u256_mul, INT256_NUM_U64_LIMBS, INT256_NUM_U8_LIMBS,
    };

    #[test]
    fn test_u256_mul() {
        let mut rng = StdRng::from_seed([42; 32]);
        for _ in 0..10000 {
            let limbs_a: [u64; INT256_NUM_U64_LIMBS] = rng.random();
            let limbs_b: [u64; INT256_NUM_U64_LIMBS] = rng.random();
            let a = U256::from_limbs(limbs_a);
            let b = U256::from_limbs(limbs_b);
            let a_u8: [u8; INT256_NUM_U8_LIMBS] = u64_array_to_bytes(limbs_a);
            let b_u8: [u8; INT256_NUM_U8_LIMBS] = u64_array_to_bytes(limbs_b);
            assert_eq!(U256::from_le_bytes(u256_mul(a_u8, b_u8)), a.wrapping_mul(b));
        }
    }
}
