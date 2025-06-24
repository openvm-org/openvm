use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, EmptyAdapterCoreLayout, MinimalInstruction, RecordArena, Result,
        StepExecutorE1, TraceFiller, TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
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
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::CastfOpcode;
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

#[derive(derive_new::new)]
pub struct CastFCoreStep<A> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, CTX, A> TraceStep<F, CTX> for CastFCoreStep<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceStep<F, CTX, ReadData = [F; 1], WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
{
    type RecordLayout = EmptyAdapterCoreLayout<F, A>;
    type RecordMut<'a> = (A::RecordMut<'a>, &'a mut CastFCoreRecord);

    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", CastfOpcode::CASTF)
    }

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let (mut adapter_record, core_record) = arena.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        core_record.val = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)[0]
            .as_canonical_u32();

        let x = run_castf(core_record.val);

        self.adapter
            .write(state.memory, instruction, &x, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, CTX, A> TraceFiller<F, CTX> for CastFCoreStep<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F, CTX>,
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

impl<F, A> StepExecutorE1<F> for CastFCoreStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<F, ReadData = [F; 1], WriteData = [u8; RV32_REGISTER_NUM_LIMBS]>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { opcode, .. } = instruction;

        assert_eq!(
            opcode.local_opcode_idx(CastfOpcode::CLASS_OFFSET),
            CastfOpcode::CASTF as usize
        );

        let [y] = self.adapter.read(state, instruction);

        let x = run_castf(y.as_canonical_u32());

        self.adapter.write(state, instruction, &x);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

#[inline(always)]
pub(super) fn run_castf(y: u32) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    debug_assert!(y < 1 << CASTF_MAX_BITS);
    y.to_le_bytes()
}
