use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv64im_transpiler::{
    Rv64HintStoreOpcode::{self, *},
    MAX_HINT_BUFFER_DWORDS,
};
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64HintStorePreCompute {
    a: u8,
    b: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64HintStoreExecutor {
    pub offset: usize,
}

impl Rv64HintStoreExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64HintStorePreCompute,
    ) -> Result<Rv64HintStoreOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, d, e, ..
        } = inst;
        if d.as_canonical_u32() != RV32_REGISTER_AS || e.as_canonical_u32() != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = Rv64HintStorePreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };

        Ok(Rv64HintStoreOpcode::from_usize(
            opcode.local_opcode_idx(self.offset),
        ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            HINT_STORED => $execute_impl::<_, _, true>,
            HINT_BUFFER => $execute_impl::<_, _, false>,
        })
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64HintStoreExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64HintStorePreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut Rv64HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
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
        let pre_compute: &mut Rv64HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64HintStoreExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64HintStorePreCompute>>()
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
        let pre_compute: &mut E2PreCompute<Rv64HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
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
        let pre_compute: &mut E2PreCompute<Rv64HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64HintStoreExecutor {
    fn get_opcode_name(&self, _opcode: usize) -> String {
        panic!("not yet implemented")
    }

    fn execute(
        &self,
        _state: VmStateMut<F, openvm_circuit::system::memory::online::TracingMemory, RA>,
        _instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        panic!("not yet implemented")
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv64HintStoreExecutor {}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64HintStoreExecutor {}

const RV64_NUM_LIMBS: usize = 8;

/// Return the number of consumed dwords for E2 height accounting.
#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_HINT_STORED: bool>(
    pre_compute: &Rv64HintStorePreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<u32, ExecutionError> {
    let pc = exec_state.pc();

    let mem_ptr_bytes: [u8; RV64_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let mem_ptr = u64::from_le_bytes(mem_ptr_bytes);

    let num_dwords = if IS_HINT_STORED {
        1u32
    } else {
        let num_dwords_bytes: [u8; RV64_NUM_LIMBS] =
            exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
        u32::from_le_bytes([
            num_dwords_bytes[0],
            num_dwords_bytes[1],
            num_dwords_bytes[2],
            num_dwords_bytes[3],
        ])
    };
    if num_dwords > MAX_HINT_BUFFER_DWORDS as u32 {
        return Err(ExecutionError::HintBufferTooLarge {
            pc,
            num_words: num_dwords,
            max_hint_buffer_words: MAX_HINT_BUFFER_DWORDS as u32,
        });
    }

    if exec_state.streams.hint_stream.len() < RV64_NUM_LIMBS * num_dwords as usize {
        return Err(ExecutionError::HintOutOfBounds { pc });
    }

    for dword_index in 0..num_dwords as u64 {
        let data: [u8; RV64_NUM_LIMBS] = std::array::from_fn(|_| {
            exec_state
                .streams
                .hint_stream
                .pop_front()
                .unwrap()
                .as_canonical_u32() as u8
        });
        exec_state.vm_write(
            RV32_MEMORY_AS,
            (mem_ptr + RV64_NUM_LIMBS as u64 * dword_index) as u32,
            &data,
        );
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    Ok(num_dwords)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_HINT_STORED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &Rv64HintStorePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64HintStorePreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_HINT_STORED>(pre_compute, exec_state)?;
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_HINT_STORED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<Rv64HintStorePreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64HintStorePreCompute>>(),
    )
    .borrow();
    let height_delta = execute_e12_impl::<F, CTX, IS_HINT_STORED>(&pre_compute.data, exec_state)?;
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height_delta);
    Ok(())
}
