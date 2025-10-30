use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
#[cfg(feature = "aot")]
use openvm_instructions::VmOpcode;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{run_auipc, Rv32AuipcExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct AuiPcPreCompute {
    imm: u32,
    a: u8,
}

impl<A> Rv32AuipcExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut AuiPcPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, c: imm, d, .. } = inst;
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let imm = imm.as_canonical_u32();
        let data: &mut AuiPcPreCompute = data.borrow_mut();
        *data = AuiPcPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

impl<F, A> InterpreterExecutor<F> for Rv32AuipcExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AuiPcPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut AuiPcPreCompute = data.borrow_mut();
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
        let data: &mut AuiPcPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }

    #[cfg(feature = "aot")]
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> String {
        use openvm_instructions::riscv::RV32_CELL_BITS;

        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let mut asm_str = String::new();
        let a: i16 = to_i16(inst.a);
        let c: i16 = to_i16(inst.c);
        let d: i16 = to_i16(inst.d);
        let rd = pc.wrapping_add((c as u32) << RV32_CELL_BITS);

        let xmm_map_reg_a = a / 8;
        // TODO: this should return an error instead.
        assert_eq!(d as u32, RV32_REGISTER_AS);
        asm_str += &format!("   mov eax, {}\n", rd);

        if (a / 4) % 2 == 0 {
            // write eax to the [0:32) bits of xmm_map_reg_a
            asm_str += &format!("   pinsrd xmm{}, eax, 0\n", xmm_map_reg_a);
        } else {
            // write eax to the [32:64) bits of xmm_map_reg_a
            asm_str += &format!("   pinsrd xmm{}, eax, 1\n", xmm_map_reg_a);
        }

        // pc += DEFAULT_PC_STEP
        asm_str += &format!("   add r13, {}\n", DEFAULT_PC_STEP);
        // instret += 1
        asm_str += &format!("   add r14, 1\n");

        asm_str
    }

    #[cfg(feature = "aot")]
    fn supports_aot_for_opcode(&self, _opcode: VmOpcode) -> bool {
        true
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for Rv32AuipcExecutor<A> where F: PrimeField32 {}

impl<F, A> MeteredExecutor<F> for Rv32AuipcExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AuiPcPreCompute>>()
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
        let data: &mut E2PreCompute<AuiPcPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<AuiPcPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &AuiPcPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rd = run_auipc(*pc, pre_compute.imm);
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
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
    let pre_compute: &AuiPcPreCompute = pre_compute.borrow();
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
    let pre_compute: &E2PreCompute<AuiPcPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, instret, pc, exec_state);
}
