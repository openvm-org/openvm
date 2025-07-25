use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
    utils::{transmute_field_to_u32, transmute_u32_to_field},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_IMM_AS, LocalOpcode, NATIVE_AS,
};
use openvm_native_compiler::NativeBranchEqualOpcode;
use openvm_rv32im_circuit::BranchEqualCoreCols;
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct NativeBranchEqualCoreRecord<F> {
    pub a: F,
    pub b: F,
    pub imm: F,
    pub is_beq: bool,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct NativeBranchEqualStep<A> {
    adapter: A,
    pub offset: usize,
    pub pc_step: u32,
}

#[derive(derive_new::new)]
pub struct NativeBranchEqualFiller<A> {
    adapter: A,
}

impl<F, A, RA> InstructionExecutor<F, RA> for NativeBranchEqualStep<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceStep<F, ReadData: Into<[F; 2]>, WriteData = ()>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut NativeBranchEqualCoreRecord<F>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            NativeBranchEqualOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, c: imm, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.a, core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let cmp_result = core_record.a == core_record.b;

        core_record.imm = imm;
        core_record.is_beq =
            opcode.local_opcode_idx(self.offset) == BranchEqualOpcode::BEQ as usize;

        if cmp_result == core_record.is_beq {
            *state.pc = (F::from_canonical_u32(*state.pc) + imm).as_canonical_u32();
        } else {
            *state.pc = state.pc.wrapping_add(self.pc_step);
        }

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for NativeBranchEqualFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &NativeBranchEqualCoreRecord<F> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BranchEqualCoreCols<F, 1> = core_row.borrow_mut();
        let (cmp_result, diff_inv_val) = run_eq(record.is_beq, record.a, record.b);

        // Writing in reverse order to avoid overwriting the `record`
        core_row.diff_inv_marker[0] = diff_inv_val;

        core_row.opcode_bne_flag = F::from_bool(!record.is_beq);
        core_row.opcode_beq_flag = F::from_bool(record.is_beq);

        core_row.imm = record.imm;
        core_row.cmp_result = F::from_bool(cmp_result);

        core_row.b = [record.b];
        core_row.a = [record.a];
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct NativeBranchEqualPreCompute {
    imm: isize,
    a_or_imm: u32,
    b_or_imm: u32,
}

impl<A> NativeBranchEqualStep<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut NativeBranchEqualPreCompute,
    ) -> Result<(bool, bool, bool), StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let local_opcode = BranchEqualOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();

        let a_is_imm = d == RV32_IMM_AS;
        let b_is_imm = e == RV32_IMM_AS;

        let a_or_imm = if a_is_imm {
            transmute_field_to_u32(&a)
        } else {
            a.as_canonical_u32()
        };
        let b_or_imm = if b_is_imm {
            transmute_field_to_u32(&b)
        } else {
            b.as_canonical_u32()
        };

        *data = NativeBranchEqualPreCompute {
            imm,
            a_or_imm,
            b_or_imm,
        };

        let is_bne = local_opcode == BranchEqualOpcode::BNE;

        Ok((a_is_imm, b_is_imm, is_bne))
    }
}

impl<F, A> InsExecutorE1<F> for NativeBranchEqualStep<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<NativeBranchEqualPreCompute>()
    }

    #[inline(always)]
    fn pre_compute_e1<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut NativeBranchEqualPreCompute = data.borrow_mut();

        let (a_is_imm, b_is_imm, is_bne) = self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = match (a_is_imm, b_is_imm, is_bne) {
            (true, true, true) => execute_e1_impl::<_, _, true, true, true>,
            (true, true, false) => execute_e1_impl::<_, _, true, true, false>,
            (true, false, true) => execute_e1_impl::<_, _, true, false, true>,
            (true, false, false) => execute_e1_impl::<_, _, true, false, false>,
            (false, true, true) => execute_e1_impl::<_, _, false, true, true>,
            (false, true, false) => execute_e1_impl::<_, _, false, true, false>,
            (false, false, true) => execute_e1_impl::<_, _, false, false, true>,
            (false, false, false) => execute_e1_impl::<_, _, false, false, false>,
        };

        Ok(fn_ptr)
    }
}

impl<F, A> InsExecutorE2<F> for NativeBranchEqualStep<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<NativeBranchEqualPreCompute>>()
    }

    #[inline(always)]
    fn pre_compute_e2<Ctx: E2ExecutionCtx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<NativeBranchEqualPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (a_is_imm, b_is_imm, is_bne) =
            self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = match (a_is_imm, b_is_imm, is_bne) {
            (true, true, true) => execute_e2_impl::<_, _, true, true, true>,
            (true, true, false) => execute_e2_impl::<_, _, true, true, false>,
            (true, false, true) => execute_e2_impl::<_, _, true, false, true>,
            (true, false, false) => execute_e2_impl::<_, _, true, false, false>,
            (false, true, true) => execute_e2_impl::<_, _, false, true, true>,
            (false, true, false) => execute_e2_impl::<_, _, false, true, false>,
            (false, false, true) => execute_e2_impl::<_, _, false, false, true>,
            (false, false, false) => execute_e2_impl::<_, _, false, false, false>,
        };

        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const IS_NE: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &NativeBranchEqualPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, A_IS_IMM, B_IS_IMM, IS_NE>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const IS_NE: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<NativeBranchEqualPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, A_IS_IMM, B_IS_IMM, IS_NE>(&pre_compute.data, vm_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const A_IS_IMM: bool,
    const B_IS_IMM: bool,
    const IS_NE: bool,
>(
    pre_compute: &NativeBranchEqualPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs1 = if A_IS_IMM {
        transmute_u32_to_field(&pre_compute.a_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(NATIVE_AS, pre_compute.a_or_imm)[0]
    };
    let rs2 = if B_IS_IMM {
        transmute_u32_to_field(&pre_compute.b_or_imm)
    } else {
        vm_state.vm_read::<F, 1>(NATIVE_AS, pre_compute.b_or_imm)[0]
    };
    if (rs1 == rs2) ^ IS_NE {
        vm_state.pc = (vm_state.pc as isize + pre_compute.imm) as u32;
    } else {
        vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    }
    vm_state.instret += 1;
}

// Returns (cmp_result, diff_idx, x[diff_idx] - y[diff_idx])
#[inline(always)]
pub(super) fn run_eq<F>(is_beq: bool, x: F, y: F) -> (bool, F)
where
    F: PrimeField32,
{
    if x != y {
        return (!is_beq, (x - y).inverse());
    }
    (is_beq, F::ZERO)
}
