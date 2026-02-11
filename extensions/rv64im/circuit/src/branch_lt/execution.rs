use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::{GuestMemory, TracingMemory}};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv64im_transpiler::Rv64BranchLessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64BranchLtPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64BranchLessThanExecutor {
    pub offset: usize,
}

impl Rv64BranchLessThanExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64BranchLtPreCompute,
    ) -> Result<Rv64BranchLessThanOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode =
            Rv64BranchLessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = Rv64BranchLtPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            Rv64BranchLessThanOpcode::BLT => Ok($execute_impl::<_, _, BltOp64>),
            Rv64BranchLessThanOpcode::BLTU => Ok($execute_impl::<_, _, BltuOp64>),
            Rv64BranchLessThanOpcode::BGE => Ok($execute_impl::<_, _, BgeOp64>),
            Rv64BranchLessThanOpcode::BGEU => Ok($execute_impl::<_, _, BgeuOp64>),
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64BranchLessThanExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64BranchLtPreCompute>()
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
        let data: &mut Rv64BranchLtPreCompute = data.borrow_mut();
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
        let data: &mut Rv64BranchLtPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64BranchLessThanExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64BranchLtPreCompute>>()
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
        let data: &mut E2PreCompute<Rv64BranchLtPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<Rv64BranchLtPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64BranchLessThanExecutor {
    fn get_opcode_name(&self, _opcode: usize) -> String {
        panic!("not yet implemented")
    }

    fn execute(
        &self,
        _state: VmStateMut<F, TracingMemory, RA>,
        _instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        panic!("not yet implemented")
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv64BranchLessThanExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64BranchLessThanExecutor {
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

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: BranchLtOp64>(
    pre_compute: &Rv64BranchLtPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let mut pc = exec_state.pc();
    let rs1 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.b as u32);
    let jmp = <OP as BranchLtOp64>::compute(u64::from_le_bytes(rs1), u64::from_le_bytes(rs2));
    if jmp {
        pc = (pc as isize + pre_compute.imm) as u32;
    } else {
        pc = pc.wrapping_add(DEFAULT_PC_STEP);
    }
    exec_state.set_pc(pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: BranchLtOp64>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64BranchLtPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64BranchLtPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: BranchLtOp64>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64BranchLtPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64BranchLtPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait BranchLtOp64 {
    fn compute(rs1: u64, rs2: u64) -> bool;
}
struct BltOp64;
struct BltuOp64;
struct BgeOp64;
struct BgeuOp64;

impl BranchLtOp64 for BltOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> bool {
        (rs1 as i64) < (rs2 as i64)
    }
}
impl BranchLtOp64 for BltuOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> bool {
        rs1 < rs2
    }
}
impl BranchLtOp64 for BgeOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> bool {
        (rs1 as i64) >= (rs2 as i64)
    }
}
impl BranchLtOp64 for BgeuOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> bool {
        rs1 >= rs2
    }
}
