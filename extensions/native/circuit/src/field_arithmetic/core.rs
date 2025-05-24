use std::borrow::{Borrow, BorrowMut};

use itertools::izip;
use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, MinimalInstruction, Result,
        StepExecutorE1, TraceStep, VmAdapterInterface, VmCoreAir, VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::FieldArithmeticOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FieldArithmeticCoreCols<T> {
    pub a: T,
    pub b: T,
    pub c: T,

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,
}

#[derive(derive_new::new, Copy, Clone, Debug)]
pub struct FieldArithmeticCoreAir {}

impl<F: Field> BaseAir<F> for FieldArithmeticCoreAir {
    fn width(&self) -> usize {
        FieldArithmeticCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldArithmeticCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FieldArithmeticCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 2]>,
    I::Writes: From<[[AB::Expr; 1]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FieldArithmeticCoreCols<_> = local_core.borrow();

        let a = cols.a;
        let b = cols.b;
        let c = cols.c;

        let flags = [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div];
        let opcodes = [ADD, SUB, MUL, DIV];
        let results = [b + c, b - c, b * c, b * cols.divisor_inv];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results` equals `io.z`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the multiplicative
        //   inverse of `io.y`.

        let mut is_valid = AB::Expr::ZERO;
        let mut expected_opcode = AB::Expr::ZERO;
        let mut expected_result = AB::Expr::ZERO;
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            is_valid += flag.into();
            expected_opcode += flag * AB::Expr::from_canonical_u32(opcode as u32);
            expected_result += flag * result;
        }
        builder.assert_eq(a, expected_result);
        builder.assert_bool(is_valid.clone());
        builder.assert_eq(cols.is_div, c * cols.divisor_inv);

        AdapterAirContext {
            to_pc: None,
            reads: [[cols.b.into()], [cols.c.into()]].into(),
            writes: [[cols.a.into()]].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        FieldArithmeticOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct FieldArithmeticRecord<F> {
    pub opcode: FieldArithmeticOpcode,
    pub a: F,
    pub b: F,
    pub c: F,
}

#[derive(derive_new::new)]
pub struct FieldArithmeticCoreStep<A> {
    adapter: A,
}

impl<F, CTX, A> TraceStep<F, CTX> for FieldArithmeticCoreStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = [F; 2],
            WriteData = [F; 1],
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldArithmeticOpcode::from_usize(opcode - FieldArithmeticOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let &Instruction { opcode, .. } = instruction;
        let local_opcode = FieldArithmeticOpcode::from_usize(
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET),
        );

        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);

        let [b_val, c_val] = self.adapter.read(state.memory, instruction, adapter_row);

        let a_val = FieldArithmetic::run_field_arithmetic(local_opcode, b_val, c_val).unwrap();

        let core_row: &mut FieldArithmeticCoreCols<_> = core_row.borrow_mut();
        core_row.a = a_val;
        core_row.b = b_val;
        core_row.c = c_val;

        core_row.is_add = F::from_bool(local_opcode == FieldArithmeticOpcode::ADD);
        core_row.is_sub = F::from_bool(local_opcode == FieldArithmeticOpcode::SUB);
        core_row.is_mul = F::from_bool(local_opcode == FieldArithmeticOpcode::MUL);
        core_row.is_div = F::from_bool(local_opcode == FieldArithmeticOpcode::DIV);

        self.adapter
            .write(state.memory, instruction, adapter_row, &[a_val]);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, (), adapter_row);

        let core_row: &mut FieldArithmeticCoreCols<_> = core_row.borrow_mut();

        core_row.divisor_inv = if core_row.is_div.is_zero() {
            F::ZERO
        } else {
            core_row.c.inverse()
        };
    }
}

impl<F, A> StepExecutorE1<F> for FieldArithmeticCoreStep<A>
where
    F: PrimeField32,
    A: 'static + for<'a> AdapterExecutorE1<F, ReadData = [F; 2], WriteData = [F; 1]>,
{
    fn execute_e1<Ctx>(
        &mut self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = FieldArithmeticOpcode::from_usize(
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET),
        );

        let [b_val, c_val] = self.adapter.read(state, instruction);
        let a_val = FieldArithmetic::run_field_arithmetic(local_opcode, b_val, c_val).unwrap();

        self.adapter.write(state, instruction, &[a_val]);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &mut self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

pub struct FieldArithmetic;
impl FieldArithmetic {
    pub(super) fn run_field_arithmetic<F: Field>(
        opcode: FieldArithmeticOpcode,
        b: F,
        c: F,
    ) -> Option<F> {
        match opcode {
            FieldArithmeticOpcode::ADD => Some(b + c),
            FieldArithmeticOpcode::SUB => Some(b - c),
            FieldArithmeticOpcode::MUL => Some(b * c),
            FieldArithmeticOpcode::DIV => {
                if c.is_zero() {
                    None
                } else {
                    Some(b * c.inverse())
                }
            }
        }
    }
}
