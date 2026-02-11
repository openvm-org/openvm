use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv64im_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64LoadSignExtendPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64LoadSignExtendExecutor {
    pub offset: usize,
}

impl Rv64LoadSignExtendExecutor {
    /// Return (load_type, enabled)
    /// load_type: 0=LOADB, 1=LOADH, 2=LOADW
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64LoadSignExtendPreCompute,
    ) -> Result<(u8, bool), StaticProgramError> {
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

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let load_type: u8 = match local_opcode {
            LOADB => 0,
            LOADH => 1,
            LOADW => 2,
            _ => unreachable!(
                "Rv64LoadSignExtendExecutor should only handle LOADB/LOADH/LOADW opcodes"
            ),
        };

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;

        *data = Rv64LoadSignExtendPreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        let enabled = !f.is_zero();
        Ok((load_type, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $load_type:ident, $enabled:ident) => {
        Ok(match ($load_type, $enabled) {
            (0, true) => $execute_impl::<_, _, 0, true>,
            (0, false) => $execute_impl::<_, _, 0, false>,
            (1, true) => $execute_impl::<_, _, 1, true>,
            (1, false) => $execute_impl::<_, _, 1, false>,
            (2, true) => $execute_impl::<_, _, 2, true>,
            (2, false) => $execute_impl::<_, _, 2, false>,
            _ => unreachable!(),
        })
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64LoadSignExtendExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64LoadSignExtendPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut Rv64LoadSignExtendPreCompute = data.borrow_mut();
        let (load_type, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, load_type, enabled)
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
        let pre_compute: &mut Rv64LoadSignExtendPreCompute = data.borrow_mut();
        let (load_type, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, load_type, enabled)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64LoadSignExtendExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64LoadSignExtendPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<Rv64LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (load_type, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, load_type, enabled)
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
        let pre_compute: &mut E2PreCompute<Rv64LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (load_type, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, load_type, enabled)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64LoadSignExtendExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64LoadSignExtendExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64LoadSignExtendExecutor {
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_metered_asm(
        &self,
        _inst: &Instruction<F>,
        _pc: u32,
        _chip_idx: usize,
        _config: &SystemConfig,
    ) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

const RV64_NUM_LIMBS: usize = 8;

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const LOAD_TYPE: u8,
    const ENABLED: bool,
>(
    pre_compute: &Rv64LoadSignExtendPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1_val = u32::from_le_bytes([rs1_bytes[0], rs1_bytes[1], rs1_bytes[2], rs1_bytes[3]]);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));

    let shift_amount = ptr_val % 8;
    let aligned_ptr = ptr_val - shift_amount;

    let read_data: [u8; RV64_NUM_LIMBS] = exec_state.vm_read(pre_compute.e as u32, aligned_ptr);

    let write_data: [u8; RV64_NUM_LIMBS] = match LOAD_TYPE {
        0 => {
            // LOADB: sign-extend byte to i64
            let byte = read_data[shift_amount as usize];
            let sign_extended = (byte as i8) as i64;
            sign_extended.to_le_bytes()
        }
        1 => {
            // LOADH: sign-extend halfword to i64
            if shift_amount % 2 != 0 || shift_amount > 6 {
                return Err(ExecutionError::Fail {
                    pc,
                    msg: "LoadSignExtend LOADH invalid shift amount",
                });
            }
            let half: [u8; 2] = array::from_fn(|i| read_data[shift_amount as usize + i]);
            (i16::from_le_bytes(half) as i64).to_le_bytes()
        }
        2 => {
            // LOADW: sign-extend word to i64 (new for RV64)
            if shift_amount != 0 && shift_amount != 4 {
                return Err(ExecutionError::Fail {
                    pc,
                    msg: "LoadSignExtend LOADW invalid shift amount",
                });
            }
            let word: [u8; 4] = array::from_fn(|i| read_data[shift_amount as usize + i]);
            (i32::from_le_bytes(word) as i64).to_le_bytes()
        }
        _ => unreachable!(),
    };

    if ENABLED {
        exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &write_data);
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const LOAD_TYPE: u8,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &Rv64LoadSignExtendPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64LoadSignExtendPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, LOAD_TYPE, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const LOAD_TYPE: u8,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<Rv64LoadSignExtendPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64LoadSignExtendPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, LOAD_TYPE, ENABLED>(&pre_compute.data, exec_state)
}
