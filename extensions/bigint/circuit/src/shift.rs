use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_bigint_transpiler::Rv32Shift256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::Rv32HeapAdapterExecutor;
use openvm_rv32im_circuit::ShiftExecutor;
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{bytes_to_u64_array, u64_array_to_bytes},
    Rv32Shift256Executor, INT256_NUM_LIMBS,
};

type AdapterExecutor = Rv32HeapAdapterExecutor<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32Shift256Executor {
    pub fn new(adapter: AdapterExecutor, offset: usize) -> Self {
        Self(ShiftExecutor::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            ShiftOpcode::SLL => $execute_impl::<_, _, SllOp>,
            ShiftOpcode::SRA => $execute_impl::<_, _, SraOp>,
            ShiftOpcode::SRL => $execute_impl::<_, _, SrlOp>,
        })
    };
}

impl<F: PrimeField32> Executor<F> for Rv32Shift256Executor {
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftPreCompute>()
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
        let data: &mut ShiftPreCompute = data.borrow_mut();
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
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for Rv32Shift256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftPreCompute>>()
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
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: &ShiftPreCompute,
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
    let rd = OP::compute(rs1, rs2);
    exec_state.vm_write(RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);
    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ShiftPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, instret, pc, exec_state);
}

impl Rv32Shift256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftPreCompute,
    ) -> Result<ShiftOpcode, StaticProgramError> {
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
        *data = ShiftPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode =
            ShiftOpcode::from_usize(opcode.local_opcode_idx(Rv32Shift256Opcode::CLASS_OFFSET));
        Ok(local_opcode)
    }
}

trait ShiftOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS];
}
struct SllOp;
struct SrlOp;
struct SraOp;
impl ShiftOp for SllOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd = [0u64; 4];
        // Only use the first 8 bits.
        let shift = (rs2_u64[0] & 0xff) as u32;
        let index_offset = (shift / u64::BITS) as usize;
        let bit_offset = shift % u64::BITS;
        let mut carry = 0u64;
        for i in index_offset..4 {
            let curr = rs1_u64[i - index_offset];
            rd[i] = (curr << bit_offset) + carry;
            if bit_offset > 0 {
                carry = curr >> (u64::BITS - bit_offset);
            }
        }
        u64_array_to_bytes(rd)
    }
}
impl ShiftOp for SrlOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        // Logical right shift - fill with 0
        shift_right(rs1, rs2, 0)
    }
}
impl ShiftOp for SraOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        // Arithmetic right shift - fill with sign bit
        if rs1[INT256_NUM_LIMBS - 1] & 0x80 > 0 {
            shift_right(rs1, rs2, u64::MAX)
        } else {
            shift_right(rs1, rs2, 0)
        }
    }
}

#[inline(always)]
fn shift_right(
    rs1: [u8; INT256_NUM_LIMBS],
    rs2: [u8; INT256_NUM_LIMBS],
    init_value: u64,
) -> [u8; INT256_NUM_LIMBS] {
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
    let mut rd = [init_value; 4];
    let shift = (rs2_u64[0] & 0xff) as u32;
    let index_offset = (shift / u64::BITS) as usize;
    let bit_offset = shift % u64::BITS;
    let mut carry = if bit_offset > 0 {
        init_value << (u64::BITS - bit_offset)
    } else {
        0
    };
    for i in (index_offset..4).rev() {
        let curr = rs1_u64[i];
        rd[i - index_offset] = (curr >> bit_offset) + carry;
        if bit_offset > 0 {
            carry = curr << (u64::BITS - bit_offset);
        }
    }
    u64_array_to_bytes(rd)
}

#[cfg(test)]
mod tests {
    use alloy_primitives::U256;
    use rand::{prelude::StdRng, Rng, SeedableRng};

    use crate::{
        shift::{ShiftOp, SllOp, SraOp, SrlOp},
        INT256_NUM_LIMBS,
    };

    #[test]
    fn test_shift_op() {
        let mut rng = StdRng::from_seed([42; 32]);
        for _ in 0..10000 {
            let limbs_a: [u8; INT256_NUM_LIMBS] = rng.gen();
            let mut limbs_b: [u8; INT256_NUM_LIMBS] = [0; INT256_NUM_LIMBS];
            let shift: u8 = rng.gen();
            limbs_b[0] = shift;
            let a = U256::from_le_bytes(limbs_a);
            {
                let res = SllOp::compute(limbs_a, limbs_b);
                assert_eq!(U256::from_le_bytes(res), a << shift);
            }
            {
                let res = SraOp::compute(limbs_a, limbs_b);
                assert_eq!(U256::from_le_bytes(res), a.arithmetic_shr(shift as usize));
            }
            {
                let res = SrlOp::compute(limbs_a, limbs_b);
                assert_eq!(U256::from_le_bytes(res), a >> shift);
            }
        }
    }
}
