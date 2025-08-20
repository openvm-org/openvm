use std::{
    array,
    borrow::{Borrow, BorrowMut},
    ops::{Add, Mul, Sub},
};

use itertools::izip;
use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::FieldExtensionOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

pub const BETA: usize = 11;
pub const EXT_DEG: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FieldExtensionCoreCols<T> {
    pub x: [T; EXT_DEG],
    pub y: [T; EXT_DEG],
    pub z: [T; EXT_DEG],

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is z.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: [T; EXT_DEG],
}

#[derive(derive_new::new, Copy, Clone, Debug)]
pub struct FieldExtensionCoreAir {}

impl<F: Field> BaseAir<F> for FieldExtensionCoreAir {
    fn width(&self) -> usize {
        FieldExtensionCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldExtensionCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for FieldExtensionCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; EXT_DEG]; 2]>,
    I::Writes: From<[[AB::Expr; EXT_DEG]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &FieldExtensionCoreCols<_> = local_core.borrow();

        let flags = [cols.is_add, cols.is_sub, cols.is_mul, cols.is_div];
        let opcodes = [FE4ADD, FE4SUB, BBE4MUL, BBE4DIV];
        let results = [
            FieldExtension::add(cols.y, cols.z),
            FieldExtension::subtract(cols.y, cols.z),
            FieldExtension::multiply(cols.y, cols.z),
            FieldExtension::multiply(cols.y, cols.divisor_inv),
        ];

        // Imposing the following constraints:
        // - Each flag in `flags` is a boolean.
        // - Exactly one flag in `flags` is true.
        // - The inner product of the `flags` and `opcodes` equals `io.opcode`.
        // - The inner product of the `flags` and `results[:,j]` equals `io.x[j]` for each `j`.
        // - If `is_div` is true, then `aux.divisor_inv` correctly represents the inverse of `io.z`.

        let mut is_valid = AB::Expr::ZERO;
        let mut expected_opcode = AB::Expr::ZERO;
        let mut expected_result = [
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        for (flag, opcode, result) in izip!(flags, opcodes, results) {
            builder.assert_bool(flag);

            is_valid += flag.into();
            expected_opcode += flag * AB::F::from_canonical_usize(opcode.local_usize());

            for (j, result_part) in result.into_iter().enumerate() {
                expected_result[j] += flag * result_part;
            }
        }

        for (x_j, expected_result_j) in izip!(cols.x, expected_result) {
            builder.assert_eq(x_j, expected_result_j);
        }
        builder.assert_bool(is_valid.clone());

        // constrain aux.divisor_inv: z * z^(-1) = 1
        let z_times_z_inv = FieldExtension::multiply(cols.z, cols.divisor_inv);
        for (i, prod_i) in z_times_z_inv.into_iter().enumerate() {
            if i == 0 {
                builder.assert_eq(cols.is_div, prod_i);
            } else {
                builder.assert_zero(prod_i);
            }
        }

        AdapterAirContext {
            to_pc: None,
            reads: [cols.y.map(Into::into), cols.z.map(Into::into)].into(),
            writes: [cols.x.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        FieldExtensionOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct FieldExtensionRecord<F> {
    pub y: [F; EXT_DEG],
    pub z: [F; EXT_DEG],
    pub local_opcode: u8,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct FieldExtensionCoreExecutor<A> {
    adapter: A,
}

#[derive(derive_new::new)]
pub struct FieldExtensionCoreFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for FieldExtensionCoreExecutor<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData = [[F; EXT_DEG]; 2], WriteData = [F; EXT_DEG]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut FieldExtensionRecord<F>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldExtensionOpcode::from_usize(opcode - FieldExtensionOpcode::CLASS_OFFSET)
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

        core_record.local_opcode =
            opcode.local_opcode_idx(FieldExtensionOpcode::CLASS_OFFSET) as u8;

        [core_record.y, core_record.z] =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        let x = run_field_extension(
            FieldExtensionOpcode::from_usize(core_record.local_opcode as usize),
            core_record.y,
            core_record.z,
        );

        self.adapter
            .write(state.memory, instruction, x, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for FieldExtensionCoreFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // FieldExtensionCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid FieldExtensionRecord written by the executor
        // during trace generation
        let record: &FieldExtensionRecord<F> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut FieldExtensionCoreCols<_> = core_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`
        let opcode = FieldExtensionOpcode::from_usize(record.local_opcode as usize);
        if opcode == FieldExtensionOpcode::BBE4DIV {
            core_row.divisor_inv = FieldExtension::invert(record.z);
        } else {
            core_row.divisor_inv = [F::ZERO; EXT_DEG];
        }

        core_row.is_div = F::from_bool(opcode == FieldExtensionOpcode::BBE4DIV);
        core_row.is_mul = F::from_bool(opcode == FieldExtensionOpcode::BBE4MUL);
        core_row.is_sub = F::from_bool(opcode == FieldExtensionOpcode::FE4SUB);
        core_row.is_add = F::from_bool(opcode == FieldExtensionOpcode::FE4ADD);

        core_row.z = record.z;
        core_row.y = record.y;
        core_row.x = run_field_extension(opcode, core_row.y, core_row.z);
    }
}

// Returns the result of the field extension operation.
// Will panic if divide by zero.
pub(super) fn run_field_extension<F: Field>(
    opcode: FieldExtensionOpcode,
    y: [F; EXT_DEG],
    z: [F; EXT_DEG],
) -> [F; EXT_DEG] {
    match opcode {
        FieldExtensionOpcode::FE4ADD => FieldExtension::add(y, z),
        FieldExtensionOpcode::FE4SUB => FieldExtension::subtract(y, z),
        FieldExtensionOpcode::BBE4MUL => FieldExtension::multiply(y, z),
        FieldExtensionOpcode::BBE4DIV => FieldExtension::divide(y, z),
    }
}

pub(crate) struct FieldExtension;

impl FieldExtension {
    pub(crate) fn add<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        V: Copy,
        V: Add<V, Output = E>,
    {
        array::from_fn(|i| x[i] + y[i])
    }

    pub(crate) fn subtract<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        V: Copy,
        V: Sub<V, Output = E>,
    {
        array::from_fn(|i| x[i] - y[i])
    }

    pub(crate) fn multiply<V, E>(x: [V; EXT_DEG], y: [V; EXT_DEG]) -> [E; EXT_DEG]
    where
        E: FieldAlgebra,
        V: Copy,
        V: Mul<V, Output = E>,
        E: Mul<V, Output = E>,
        V: Add<V, Output = E>,
        E: Add<V, Output = E>,
    {
        let [x0, x1, x2, x3] = x;
        let [y0, y1, y2, y3] = y;
        [
            x0 * y0 + (x1 * y3 + x2 * y2 + x3 * y1) * E::from_canonical_usize(BETA),
            x0 * y1 + x1 * y0 + (x2 * y3 + x3 * y2) * E::from_canonical_usize(BETA),
            x0 * y2 + x1 * y1 + x2 * y0 + (x3 * y3) * E::from_canonical_usize(BETA),
            x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
        ]
    }

    pub(crate) fn divide<F: Field>(x: [F; EXT_DEG], y: [F; EXT_DEG]) -> [F; EXT_DEG] {
        Self::multiply(x, Self::invert(y))
    }

    pub(crate) fn invert<F: Field>(a: [F; EXT_DEG]) -> [F; EXT_DEG] {
        // Let a = (a0, a1, a2, a3) represent the element we want to invert.
        // Define a' = (a0, -a1, a2, -a3).  By construction, the product b = a * a' will have zero
        // degree-1 and degree-3 coefficients.
        // Let b = (b0, 0, b2, 0) and define b' = (b0, 0, -b2, 0).
        // Note that c = b * b' = b0^2 - BETA * b2^2, which is an element of the base field.
        // Therefore, the inverse of a is 1 / a = a' / (a * a') = a' * b' / (b * b') = a' * b' / c.

        let [a0, a1, a2, a3] = a;

        let beta = F::from_canonical_usize(BETA);

        let mut b0 = a0 * a0 - beta * (F::TWO * a1 * a3 - a2 * a2);
        let mut b2 = F::TWO * a0 * a2 - a1 * a1 - beta * a3 * a3;

        let c = b0 * b0 - beta * b2 * b2;
        let inv_c = c.inverse();

        b0 *= inv_c;
        b2 *= inv_c;

        [
            a0 * b0 - a2 * b2 * beta,
            -a1 * b0 + a3 * b2 * beta,
            -a0 * b2 + a2 * b0,
            a1 * b2 - a3 * b0,
        ]
    }
}
