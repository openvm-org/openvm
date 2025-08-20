use std::borrow::{Borrow, BorrowMut};

use itertools::izip;
use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::FieldArithmeticOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

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
#[derive(AlignedBytesBorrow, Debug)]
pub struct FieldArithmeticRecord<F> {
    pub b: F,
    pub c: F,
    pub local_opcode: u8,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct FieldArithmeticCoreExecutor<A> {
    adapter: A,
}

#[derive(derive_new::new)]
pub struct FieldArithmeticCoreFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for FieldArithmeticCoreExecutor<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData = [F; 2], WriteData = [F; 1]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FieldArithmeticRecord<F>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldArithmeticOpcode::from_usize(opcode - FieldArithmeticOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        core_record.local_opcode =
            opcode.local_opcode_idx(FieldArithmeticOpcode::CLASS_OFFSET) as u8;

        let opcode = FieldArithmeticOpcode::from_usize(core_record.local_opcode as usize);
        let a_val = run_field_arithmetic(opcode, core_record.b, core_record.c);

        self.adapter
            .write(state.memory, instruction, [a_val], &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for FieldArithmeticCoreFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // FieldArithmeticCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid FieldArithmeticRecord written by the executor
        // during trace generation
        let record: &FieldArithmeticRecord<F> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FieldArithmeticCoreCols<_> = core_row.borrow_mut();

        let opcode = FieldArithmeticOpcode::from_usize(record.local_opcode as usize);
        let result = run_field_arithmetic(opcode, record.b, record.c);

        // Writing in reverse order to avoid overwriting the `record`
        core_row.divisor_inv = if opcode == FieldArithmeticOpcode::DIV {
            record.c.inverse()
        } else {
            F::ZERO
        };

        core_row.is_div = F::from_bool(opcode == FieldArithmeticOpcode::DIV);
        core_row.is_mul = F::from_bool(opcode == FieldArithmeticOpcode::MUL);
        core_row.is_sub = F::from_bool(opcode == FieldArithmeticOpcode::SUB);
        core_row.is_add = F::from_bool(opcode == FieldArithmeticOpcode::ADD);

        core_row.c = record.c;
        core_row.b = record.b;
        core_row.a = result;
    }
}

pub(super) fn run_field_arithmetic<F: Field>(opcode: FieldArithmeticOpcode, b: F, c: F) -> F {
    match opcode {
        FieldArithmeticOpcode::ADD => b + c,
        FieldArithmeticOpcode::SUB => b - c,
        FieldArithmeticOpcode::MUL => b * c,
        FieldArithmeticOpcode::DIV => {
            assert!(!c.is_zero(), "Division by zero");
            b * c.inverse()
        }
    }
}
