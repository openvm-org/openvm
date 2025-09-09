use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32Mul256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::Rv32HeapAdapterExecutor;
use openvm_rv32im_circuit::MultiplicationExecutor;
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{bytes_to_u32_array, u32_array_to_bytes},
    Rv32Multiplication256Executor, INT256_NUM_LIMBS,
};

type AdapterExecutor = Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32Multiplication256Executor {
    pub fn new(adapter: AdapterExecutor, offset: usize) -> Self {
        Self(MultiplicationExecutor::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MultPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> Executor<F> for Rv32Multiplication256Executor {
    fn pre_compute_size(&self) -> usize {
        size_of::<MultPreCompute>()
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
        let data: &mut MultPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
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
        let data: &mut MultPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for Rv32Multiplication256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MultPreCompute>>()
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
        let data: &mut E2PreCompute<MultPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
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
        let data: &mut E2PreCompute<MultPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &MultPreCompute,
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
    let rd = u256_mul(rs1, rs2);
    exec_state.vm_write(RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);

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
    let pre_compute: &MultPreCompute = pre_compute.borrow();
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
    let pre_compute: &E2PreCompute<MultPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, instret, pc, exec_state);
}

impl Rv32Multiplication256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
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
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode =
            MulOpcode::from_usize(opcode.local_opcode_idx(Rv32Mul256Opcode::CLASS_OFFSET));
        assert_eq!(local_opcode, MulOpcode::MUL);
        *data = MultPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

#[inline(always)]
pub(crate) fn u256_mul(
    rs1: [u8; INT256_NUM_LIMBS],
    rs2: [u8; INT256_NUM_LIMBS],
) -> [u8; INT256_NUM_LIMBS] {
    let rs1_u64: [u32; 8] = bytes_to_u32_array(rs1);
    let rs2_u64: [u32; 8] = bytes_to_u32_array(rs2);
    let mut rd = [0u32; 8];
    for i in 0..8 {
        let mut carry = 0u64;
        for j in 0..(8 - i) {
            let res = rs1_u64[i] as u64 * rs2_u64[j] as u64 + rd[i + j] as u64 + carry;
            rd[i + j] = res as u32;
            carry = res >> 32;
        }
    }
    u32_array_to_bytes(rd)
}

#[cfg(test)]
mod tests {
    use alloy_primitives::U256;
    use rand::{prelude::StdRng, Rng, SeedableRng};

    use crate::{common::u64_array_to_bytes, mult::u256_mul, INT256_NUM_LIMBS};

    #[test]
    fn test_u256_mul() {
        let mut rng = StdRng::from_seed([42; 32]);
        for _ in 0..10000 {
            let limbs_a: [u64; 4] = rng.gen();
            let limbs_b: [u64; 4] = rng.gen();
            let a = U256::from_limbs(limbs_a);
            let b = U256::from_limbs(limbs_b);
            let a_u8: [u8; INT256_NUM_LIMBS] = u64_array_to_bytes(limbs_a);
            let b_u8: [u8; INT256_NUM_LIMBS] = u64_array_to_bytes(limbs_b);
            assert_eq!(U256::from_le_bytes(u256_mul(a_u8, b_u8)), a.wrapping_mul(b));
        }
    }
}
