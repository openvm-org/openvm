use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_MEMORY_AS, LocalOpcode,
};
use openvm_native_compiler::{conversion::AS, CastfOpcode};
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::CASTF_MAX_BITS;

// LIMB_BITS is the size of the limbs in bits.
pub(crate) const LIMB_BITS: usize = 8;
// the final limb has only 6 bits
pub(crate) const FINAL_LIMB_BITS: usize = 6;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct CastFCoreCols<T> {
    pub in_val: T,
    pub out_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_valid: T,
}

#[derive(derive_new::new, Copy, Clone, Debug)]
pub struct CastFCoreAir {
    pub bus: VariableRangeCheckerBus, /* to communicate with the range checker that checks that
                                       * all limbs are < 2^LIMB_BITS */
}

impl<F: Field> BaseAir<F> for CastFCoreAir {
    fn width(&self) -> usize {
        CastFCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for CastFCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for CastFCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 1]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &CastFCoreCols<_> = local_core.borrow();

        builder.assert_bool(cols.is_valid);

        let intermed_val = cols
            .out_val
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &limb)| {
                acc + limb * AB::Expr::from_canonical_u32(1 << (i * LIMB_BITS))
            });

        for i in 0..4 {
            self.bus
                .range_check(
                    cols.out_val[i],
                    match i {
                        0..=2 => LIMB_BITS,
                        3 => FINAL_LIMB_BITS,
                        _ => unreachable!(),
                    },
                )
                .eval(builder, cols.is_valid);
        }

        AdapterAirContext {
            to_pc: None,
            reads: [[intermed_val]].into(),
            writes: [cols.out_val.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_canonical_usize(
                    CastfOpcode::CASTF.global_opcode().as_usize(),
                ),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        CastfOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct CastFCoreRecord {
    pub val: u32,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct CastFCoreExecutor<A> {
    adapter: A,
}

#[derive(derive_new::new)]
pub struct CastFCoreFiller<A> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for CastFCoreExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData = [F; 1], WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut CastFCoreRecord),
    >,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", CastfOpcode::CASTF)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        core_record.val = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)[0]
            .as_canonical_u32();

        let x = run_castf(core_record.val);

        self.adapter
            .write(state.memory, instruction, x, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for CastFCoreFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &CastFCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut CastFCoreCols<_> = core_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`
        let out = run_castf(record.val);
        for (i, &limb) in out.iter().enumerate() {
            let limb_bits = if i == out.len() - 1 {
                FINAL_LIMB_BITS
            } else {
                LIMB_BITS
            };
            self.range_checker_chip.add_count(limb as u32, limb_bits);
        }
        core_row.is_valid = F::ONE;
        core_row.out_val = out.map(F::from_canonical_u8);
        core_row.in_val = F::from_canonical_u32(record.val);
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct CastFPreCompute {
    a: u32,
    b: u32,
}

impl<A> CastFCoreExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut CastFPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            a, b, d, e, opcode, ..
        } = inst;

        if opcode.local_opcode_idx(CastfOpcode::CLASS_OFFSET) != CastfOpcode::CASTF as usize {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if d.as_canonical_u32() != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if e.as_canonical_u32() != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        *data = CastFPreCompute { a, b };

        Ok(())
    }
}

impl<F, A> Executor<F> for CastFCoreExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<CastFPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut CastFPreCompute = data.borrow_mut();

        self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = execute_e1_impl::<_, _>;

        Ok(fn_ptr)
    }
}

impl<F, A> MeteredExecutor<F> for CastFCoreExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<CastFPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<CastFPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = execute_e2_impl::<_, _>;

        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &CastFPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<CastFPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, vm_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &CastFPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let y = vm_state.vm_read::<F, 1>(AS::Native as u32, pre_compute.b)[0];
    let x = run_castf(y.as_canonical_u32());

    vm_state.vm_write::<u8, RV32_REGISTER_NUM_LIMBS>(RV32_MEMORY_AS, pre_compute.a, &x);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[inline(always)]
pub(super) fn run_castf(y: u32) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    debug_assert!(y < 1 << CASTF_MAX_BITS);
    y.to_le_bytes()
}
