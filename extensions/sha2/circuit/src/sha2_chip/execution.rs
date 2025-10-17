use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_sha256_air::{get_sha256_num_blocks, SHA256_ROWS_PER_BLOCK};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{sha256_solve, Sha256VmExecutor, SHA256_NUM_READ_ROWS, SHA256_READ_SIZE};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShaPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> Executor<F> for Sha256VmExecutor {
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
        let data: &mut ShaPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_, _>)
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<ShaPreCompute>()
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
        let data: &mut ShaPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for Sha256VmExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShaPreCompute>>()
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
        let data: &mut E2PreCompute<ShaPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
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
        let data: &mut E2PreCompute<ShaPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<_, _>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &ShaPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let dst = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let src = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let len = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let src_u32 = u32::from_le_bytes(src);
    let len_u32 = u32::from_le_bytes(len);

    let (output, height) = if IS_E1 {
        // SAFETY: RV32_MEMORY_AS is memory address space of type u8
        let message = exec_state.vm_read_slice(RV32_MEMORY_AS, src_u32, len_u32 as usize);
        let output = sha256_solve(message);
        (output, 0)
    } else {
        let num_blocks = get_sha256_num_blocks(len_u32);
        let mut message = Vec::with_capacity(len_u32 as usize);
        for block_idx in 0..num_blocks as usize {
            // Reads happen on the first 4 rows of each block
            for row in 0..SHA256_NUM_READ_ROWS {
                let read_idx = block_idx * SHA256_NUM_READ_ROWS + row;
                let row_input: [u8; SHA256_READ_SIZE] = exec_state.vm_read(
                    RV32_MEMORY_AS,
                    src_u32 + (read_idx * SHA256_READ_SIZE) as u32,
                );
                message.extend_from_slice(&row_input);
            }
        }
        let output = sha256_solve(&message[..len_u32 as usize]);
        let height = num_blocks * SHA256_ROWS_PER_BLOCK as u32;
        (output, height)
    };
    exec_state.vm_write(RV32_MEMORY_AS, dst_u32, &output);

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    height
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
    let pre_compute: &ShaPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, instret, pc, exec_state);
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
    let pre_compute: &E2PreCompute<ShaPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX, false>(&pre_compute.data, instret, pc, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

impl Sha256VmExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShaPreCompute,
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
        *data = ShaPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        assert_eq!(&Rv32Sha256Opcode::SHA256.global_opcode(), opcode);
        Ok(())
    }
}
