use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::LoadSignExtendExecutor;
use crate::adapters::{rv64_address_add_imm, rv64_bytes_to_u32};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadSignExtendPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

impl<A, const LIMB_BITS: usize> LoadSignExtendExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    /// Return (local_opcode, enabled)
    fn pre_compute_impl_rv64<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadSignExtendPreCompute,
    ) -> Result<(Rv64LoadStoreOpcode, bool), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;

        let local_opcode = Rv64LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV64_REGISTER_AS || e_u32 != RV64_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        debug_assert!(
            matches!(local_opcode, LOADB | LOADH | LOADW),
            "LoadSignExtendExecutor should only handle LOADB/LOADH/LOADW opcodes"
        );

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        *data = LoadSignExtendPreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        let enabled = !f.is_zero();
        Ok((local_opcode, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $enabled:ident) => {
        match ($local_opcode, $enabled) {
            (LOADB, true) => Ok($execute_impl::<_, _, LoadBOp, true>),
            (LOADB, false) => Ok($execute_impl::<_, _, LoadBOp, false>),
            (LOADH, true) => Ok($execute_impl::<_, _, LoadHOp, true>),
            (LOADH, false) => Ok($execute_impl::<_, _, LoadHOp, false>),
            (LOADW, true) => Ok($execute_impl::<_, _, LoadWOp, true>),
            (LOADW, false) => Ok($execute_impl::<_, _, LoadWOp, false>),
            _ => unreachable!(),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for LoadSignExtendExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadSignExtendPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (local_opcode, enabled) = self.pre_compute_impl_rv64(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
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
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (local_opcode, enabled) = self.pre_compute_impl_rv64(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for LoadSignExtendExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled) =
            self.pre_compute_impl_rv64(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled)
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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled) =
            self.pre_compute_impl_rv64(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled)
    }
}

/// Sign-extends the `WIDTH` bytes starting at the byte shift in the concatenation of the two
/// read blocks. Any byte shift is supported.
trait LoadSignExtOp {
    const WIDTH: usize;

    fn compute_write_data(
        src: &[u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS];
}

struct LoadBOp;
struct LoadHOp;
struct LoadWOp;

impl LoadSignExtOp for LoadBOp {
    const WIDTH: usize = 1;

    #[inline(always)]
    fn compute_write_data(
        src: &[u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        ((src[shift_amount] as i8) as i64).to_le_bytes()
    }
}

impl LoadSignExtOp for LoadHOp {
    const WIDTH: usize = 2;

    #[inline(always)]
    fn compute_write_data(
        src: &[u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let half: [u8; 2] = array::from_fn(|i| src[shift_amount + i]);
        (i16::from_le_bytes(half) as i64).to_le_bytes()
    }
}

impl LoadSignExtOp for LoadWOp {
    const WIDTH: usize = 4;

    #[inline(always)]
    fn compute_write_data(
        src: &[u8; 2 * RV64_REGISTER_NUM_LIMBS],
        shift_amount: usize,
    ) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let word: [u8; 4] = array::from_fn(|i| src[shift_amount + i]);
        (i32::from_le_bytes(word) as i64).to_le_bytes()
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: LoadSignExtOp,
    const ENABLED: bool,
>(
    pre_compute: &LoadSignExtendPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = rv64_bytes_to_u32(rs1_bytes);
    let addr = rv64_address_add_imm(rs1_val, pre_compute.imm_extended);
    debug_assert!((addr as usize) < RV64_MEMORY_BYTES);
    let ptr_val = addr as u32;

    let shift_amount = (ptr_val as usize) % RV64_REGISTER_NUM_LIMBS;
    let aligned_ptr = ptr_val - shift_amount as u32;
    // Whether the access spans two adjacent 8-byte blocks.
    let crosses = shift_amount + OP::WIDTH > RV64_REGISTER_NUM_LIMBS;
    debug_assert!(
        !crosses || (aligned_ptr as usize) + 2 * RV64_REGISTER_NUM_LIMBS <= RV64_MEMORY_BYTES
    );

    let mut src = [0u8; 2 * RV64_REGISTER_NUM_LIMBS];
    let block0: [u8; RV64_REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(pre_compute.e as u32, aligned_ptr);
    src[..RV64_REGISTER_NUM_LIMBS].copy_from_slice(&block0);
    if crosses {
        let block1: [u8; RV64_REGISTER_NUM_LIMBS] = exec_state.vm_read_bytes(
            pre_compute.e as u32,
            aligned_ptr + RV64_REGISTER_NUM_LIMBS as u32,
        );
        src[RV64_REGISTER_NUM_LIMBS..].copy_from_slice(&block1);
    }

    if ENABLED {
        let write_data = OP::compute_write_data(&src, shift_amount);
        exec_state.vm_write_bytes(RV64_REGISTER_AS, pre_compute.a as u32, &write_data);
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: LoadSignExtOp,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &LoadSignExtendPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<LoadSignExtendPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    OP: LoadSignExtOp,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<LoadSignExtendPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP, ENABLED>(&pre_compute.data, exec_state)
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::StaticProgramError;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode, DEFERRAL_AS,
    };
    use openvm_riscv_transpiler::Rv64LoadStoreOpcode::LOADW;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::LoadSignExtendPreCompute;
    use crate::{adapters::Rv64LoadStoreAdapterExecutor, Rv64LoadSignExtendExecutor};

    #[test]
    fn precompute_enforces_address_space_domain() {
        const PC: u32 = 0x100;

        let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreAdapterExecutor::new(29));
        let inst = Instruction::<BabyBear>::from_usize(
            LOADW.global_opcode(),
            [
                8,
                16,
                0,
                RV64_REGISTER_AS as usize,
                DEFERRAL_AS as usize,
                1,
                0,
            ],
        );
        let mut data = LoadSignExtendPreCompute {
            imm_extended: 0,
            a: 0,
            b: 0,
            e: 0,
        };

        let err = executor
            .pre_compute_impl_rv64(PC, &inst, &mut data)
            .expect_err("load address-space domain should be enforced");

        assert!(matches!(err, StaticProgramError::InvalidInstruction(PC)));
    }
}
