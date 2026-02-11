use std::{
    borrow::{Borrow, BorrowMut},
    fmt::Debug,
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
    LocalOpcode, NATIVE_AS,
};
use openvm_rv64im_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64LoadStorePreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
    e: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64LoadStoreExecutor {
    pub offset: usize,
}

impl Rv64LoadStoreExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64LoadStorePreCompute,
    ) -> Result<(Rv64LoadStoreOpcode, bool, bool), StaticProgramError> {
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
        let enabled = !f.is_zero();

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        match local_opcode {
            LOADD | LOADWU | LOADBU | LOADHU => {}
            STORED | STOREW | STOREH | STOREB => {
                if !enabled {
                    return Err(StaticProgramError::InvalidInstruction(pc));
                }
            }
            _ => unreachable!("Rv64LoadStoreExecutor should not handle LOADB/LOADH/LOADW opcodes"),
        }

        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        let imm_extended = imm + imm_sign * 0xffff0000;
        let is_native_store = e_u32 == NATIVE_AS;

        *data = Rv64LoadStorePreCompute {
            imm_extended,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            e: e_u32 as u8,
        };
        Ok((local_opcode, enabled, is_native_store))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $enabled:ident, $is_native_store:ident) => {
        match ($local_opcode, $enabled, $is_native_store) {
            (LOADD, true, _) => Ok($execute_impl::<_, _, U8, LoadDOp, true>),
            (LOADD, false, _) => Ok($execute_impl::<_, _, U8, LoadDOp, false>),
            (LOADWU, true, _) => Ok($execute_impl::<_, _, U8, LoadWUOp, true>),
            (LOADWU, false, _) => Ok($execute_impl::<_, _, U8, LoadWUOp, false>),
            (LOADHU, true, _) => Ok($execute_impl::<_, _, U8, LoadHUOp, true>),
            (LOADHU, false, _) => Ok($execute_impl::<_, _, U8, LoadHUOp, false>),
            (LOADBU, true, _) => Ok($execute_impl::<_, _, U8, LoadBUOp, true>),
            (LOADBU, false, _) => Ok($execute_impl::<_, _, U8, LoadBUOp, false>),
            (STORED, true, false) => Ok($execute_impl::<_, _, U8, StoreDOp, true>),
            (STORED, false, false) => Ok($execute_impl::<_, _, U8, StoreDOp, false>),
            (STORED, true, true) => Ok($execute_impl::<_, _, F, StoreDOp, true>),
            (STORED, false, true) => Ok($execute_impl::<_, _, F, StoreDOp, false>),
            (STOREW, true, false) => Ok($execute_impl::<_, _, U8, StoreWOp, true>),
            (STOREW, false, false) => Ok($execute_impl::<_, _, U8, StoreWOp, false>),
            (STOREW, true, true) => Ok($execute_impl::<_, _, F, StoreWOp, true>),
            (STOREW, false, true) => Ok($execute_impl::<_, _, F, StoreWOp, false>),
            (STOREH, true, false) => Ok($execute_impl::<_, _, U8, StoreHOp, true>),
            (STOREH, false, false) => Ok($execute_impl::<_, _, U8, StoreHOp, false>),
            (STOREH, true, true) => Ok($execute_impl::<_, _, F, StoreHOp, true>),
            (STOREH, false, true) => Ok($execute_impl::<_, _, F, StoreHOp, false>),
            (STOREB, true, false) => Ok($execute_impl::<_, _, U8, StoreBOp, true>),
            (STOREB, false, false) => Ok($execute_impl::<_, _, U8, StoreBOp, false>),
            (STOREB, true, true) => Ok($execute_impl::<_, _, F, StoreBOp, true>),
            (STOREB, false, true) => Ok($execute_impl::<_, _, F, StoreBOp, false>),
            (_, _, _) => unreachable!(),
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64LoadStoreExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64LoadStorePreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut Rv64LoadStorePreCompute = data.borrow_mut();
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled, is_native_store)
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
        let pre_compute: &mut Rv64LoadStorePreCompute = data.borrow_mut();
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode, enabled, is_native_store)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64LoadStoreExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64LoadStorePreCompute>>()
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
        let pre_compute: &mut E2PreCompute<Rv64LoadStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled, is_native_store)
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
        let pre_compute: &mut E2PreCompute<Rv64LoadStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (local_opcode, enabled, is_native_store) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode, enabled, is_native_store)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64LoadStoreExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64LoadStoreExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64LoadStoreExecutor {
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
    T: Copy + Debug + Default,
    OP: Rv64LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: &Rv64LoadStorePreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1_bytes: [u8; RV64_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    // Take lower 32 bits as pointer (matches RV32 semantics in this codebase)
    let rs1_val = u32::from_le_bytes([rs1_bytes[0], rs1_bytes[1], rs1_bytes[2], rs1_bytes[3]]);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));

    let shift_amount = ptr_val % 8;
    let aligned_ptr = ptr_val - shift_amount;

    let read_data: [u8; RV64_NUM_LIMBS] = if OP::IS_LOAD {
        exec_state.vm_read(pre_compute.e as u32, aligned_ptr)
    } else {
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32)
    };

    let mut write_data: [T; RV64_NUM_LIMBS] = if OP::HOST_READ {
        exec_state.host_read(pre_compute.e as u32, aligned_ptr)
    } else {
        [T::default(); RV64_NUM_LIMBS]
    };

    if !OP::compute_write_data(&mut write_data, read_data, shift_amount as usize) {
        let err = ExecutionError::Fail {
            pc,
            msg: "Invalid Rv64LoadStoreOp",
        };
        return Err(err);
    }

    if ENABLED {
        if OP::IS_LOAD {
            exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &write_data);
        } else {
            exec_state.vm_write(pre_compute.e as u32, aligned_ptr, &write_data);
        }
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    T: Copy + Debug + Default,
    OP: Rv64LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &Rv64LoadStorePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64LoadStorePreCompute>()).borrow();
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    T: Copy + Debug + Default,
    OP: Rv64LoadStoreOp<T>,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<Rv64LoadStorePreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64LoadStorePreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, T, OP, ENABLED>(&pre_compute.data, exec_state)
}

trait Rv64LoadStoreOp<T> {
    const IS_LOAD: bool;
    const HOST_READ: bool;

    fn compute_write_data(
        write_data: &mut [T; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool;
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default)]
struct U8(u8);

struct LoadDOp;
struct LoadWUOp;
struct LoadHUOp;
struct LoadBUOp;
struct StoreDOp;
struct StoreWOp;
struct StoreHOp;
struct StoreBOp;

// --- Load operations (U8 only) ---

impl Rv64LoadStoreOp<U8> for LoadDOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}

impl Rv64LoadStoreOp<U8> for LoadWUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 4 {
            return false;
        }
        write_data[0] = U8(read_data[shift_amount]);
        write_data[1] = U8(read_data[shift_amount + 1]);
        write_data[2] = U8(read_data[shift_amount + 2]);
        write_data[3] = U8(read_data[shift_amount + 3]);
        // Zero-extend to 8 bytes
        write_data[4] = U8(0);
        write_data[5] = U8(0);
        write_data[6] = U8(0);
        write_data[7] = U8(0);
        true
    }
}

impl Rv64LoadStoreOp<U8> for LoadHUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount % 2 != 0 || shift_amount > 6 {
            return false;
        }
        write_data[0] = U8(read_data[shift_amount]);
        write_data[1] = U8(read_data[shift_amount + 1]);
        for i in 2..RV64_NUM_LIMBS {
            write_data[i] = U8(0);
        }
        true
    }
}

impl Rv64LoadStoreOp<U8> for LoadBUOp {
    const IS_LOAD: bool = true;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[0] = U8(read_data[shift_amount]);
        for i in 1..RV64_NUM_LIMBS {
            write_data[i] = U8(0);
        }
        true
    }
}

// --- Store operations (U8 variant) ---

impl Rv64LoadStoreOp<U8> for StoreDOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(U8);
        true
    }
}

impl Rv64LoadStoreOp<U8> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 4 {
            return false;
        }
        write_data[shift_amount] = U8(read_data[0]);
        write_data[shift_amount + 1] = U8(read_data[1]);
        write_data[shift_amount + 2] = U8(read_data[2]);
        write_data[shift_amount + 3] = U8(read_data[3]);
        true
    }
}

impl Rv64LoadStoreOp<U8> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount % 2 != 0 || shift_amount > 6 {
            return false;
        }
        write_data[shift_amount] = U8(read_data[0]);
        write_data[shift_amount + 1] = U8(read_data[1]);
        true
    }
}

impl Rv64LoadStoreOp<U8> for StoreBOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [U8; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = U8(read_data[0]);
        true
    }
}

// --- Store operations (F variant for native AS) ---

impl<F: PrimeField32> Rv64LoadStoreOp<F> for StoreDOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = false;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        _shift_amount: usize,
    ) -> bool {
        *write_data = read_data.map(F::from_canonical_u8);
        true
    }
}

impl<F: PrimeField32> Rv64LoadStoreOp<F> for StoreWOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount != 0 && shift_amount != 4 {
            return false;
        }
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        write_data[shift_amount + 1] = F::from_canonical_u8(read_data[1]);
        write_data[shift_amount + 2] = F::from_canonical_u8(read_data[2]);
        write_data[shift_amount + 3] = F::from_canonical_u8(read_data[3]);
        true
    }
}

impl<F: PrimeField32> Rv64LoadStoreOp<F> for StoreHOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        if shift_amount % 2 != 0 || shift_amount > 6 {
            return false;
        }
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        write_data[shift_amount + 1] = F::from_canonical_u8(read_data[1]);
        true
    }
}

impl<F: PrimeField32> Rv64LoadStoreOp<F> for StoreBOp {
    const IS_LOAD: bool = false;
    const HOST_READ: bool = true;

    #[inline(always)]
    fn compute_write_data(
        write_data: &mut [F; RV64_NUM_LIMBS],
        read_data: [u8; RV64_NUM_LIMBS],
        shift_amount: usize,
    ) -> bool {
        write_data[shift_amount] = F::from_canonical_u8(read_data[0]);
        true
    }
}
